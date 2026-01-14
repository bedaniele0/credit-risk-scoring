"""
Credit Risk Scoring API - FastAPI Application

Proyecto: Credit Risk Scoring - UCI Taiwan Dataset
Fase DVP-PRO: F8 - Productización
Autor: Ing. Daniel Varela Pérez
Email: bedaniele0@gmail.com
Tel: +52 55 4189 3428
Fecha: 2025-11-18
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import time
from scipy import stats

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Cargar configuración
OPTIMAL_THRESHOLD = 0.12  # Threshold optimizado en F7

# =====================================================
# METRICAS PROMETHEUS
# =====================================================

REQUEST_COUNT = Counter(
    "credit_api_requests_total",
    "Total de requests HTTP procesadas",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "credit_api_request_latency_seconds",
    "Latencia de requests HTTP",
    ["endpoint"]
)
PREDICTIONS_TOTAL = Counter(
    "credit_api_predictions_total",
    "Total de predicciones realizadas",
    ["mode"]
)
MODEL_LOADED_GAUGE = Gauge(
    "credit_api_model_loaded",
    "Estado de carga del modelo (1=cargado, 0=no cargado)"
)

# API Key (opcional, activada si se define API_KEY env)
API_KEY = os.getenv("API_KEY")


class PrometheusRoute(APIRoute):
    """Route handler instrumentado con Prometheus."""

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            start_time = time.perf_counter()
            response: Response = await original_route_handler(request)
            elapsed = time.perf_counter() - start_time

            endpoint = request.url.path
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=response.status_code
            ).inc()
            return response

        return custom_route_handler


# =====================================================
# Inicializar FastAPI con lifespan
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga de recursos en startup/shutdown."""
    global model, baseline_scores

    logger.info("Iniciando app (lifespan)")
    success = load_model()
    MODEL_LOADED_GAUGE.set(1 if success else 0)

    if success:
        try:
            baseline_scores = compute_reference_scores()
            if baseline_scores is not None:
                logger.info("Scores de referencia cargados para PSI/KS")
        except Exception as e:
            logger.warning(f"No se pudieron calcular scores de referencia: {e}")

    yield

    # Shutdown hooks (si aplica)
    logger.info("Apagando app (lifespan)")


app = FastAPI(
    title="Credit Risk Scoring API",
    description="""
    API para predicción de riesgo crediticio basada en el dataset UCI Taiwan.

    ## Funcionalidades
    - **Predicción individual**: Probabilidad de default y banda de riesgo
    - **Predicción batch**: Procesa hasta 100 solicitudes
    - **Health/Métricas**: Estado del modelo y performance

    ## Metodología
    DVP-PRO por Ing. Daniel Varela Pérez.
    """,
    version="1.0.0",
    contact={
        "name": "Ing. Daniel Varela Pérez",
        "email": "bedaniele0@gmail.com"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.router.route_class = PrometheusRoute

# =====================================================
# MODELOS PYDANTIC
# =====================================================

class CreditApplication(BaseModel):
    """Schema para una solicitud de crédito individual."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 50000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 35,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 40000,
                "BILL_AMT2": 38000,
                "BILL_AMT3": 35000,
                "BILL_AMT4": 32000,
                "BILL_AMT5": 30000,
                "BILL_AMT6": 28000,
                "PAY_AMT1": 2000,
                "PAY_AMT2": 2000,
                "PAY_AMT3": 2000,
                "PAY_AMT4": 2000,
                "PAY_AMT5": 2000,
                "PAY_AMT6": 2000
            }
        }
    )

    # Variables demográficas
    LIMIT_BAL: float = Field(..., description="Monto del crédito otorgado (NT$)", ge=0)
    SEX: int = Field(..., description="Género (1=masculino, 2=femenino)", ge=1, le=2)
    EDUCATION: int = Field(..., description="Educación (1=posgrado, 2=universidad, 3=preparatoria, 4=otros)", ge=1, le=4)
    MARRIAGE: int = Field(..., description="Estado civil (1=casado, 2=soltero, 3=otros)", ge=1, le=3)
    AGE: int = Field(..., description="Edad en años", ge=18, le=100)

    # Historial de pagos (PAY_0 a PAY_6)
    # -1=pago a tiempo, 1=retraso 1 mes, 2=retraso 2 meses, etc.
    PAY_0: int = Field(..., description="Estado de pago en septiembre", ge=-2, le=8)
    PAY_2: int = Field(..., description="Estado de pago en agosto", ge=-2, le=8)
    PAY_3: int = Field(..., description="Estado de pago en julio", ge=-2, le=8)
    PAY_4: int = Field(..., description="Estado de pago en junio", ge=-2, le=8)
    PAY_5: int = Field(..., description="Estado de pago en mayo", ge=-2, le=8)
    PAY_6: int = Field(..., description="Estado de pago en abril", ge=-2, le=8)

    # Montos de factura (BILL_AMT1 a BILL_AMT6)
    BILL_AMT1: float = Field(..., description="Monto facturado en septiembre (NT$)")
    BILL_AMT2: float = Field(..., description="Monto facturado en agosto (NT$)")
    BILL_AMT3: float = Field(..., description="Monto facturado en julio (NT$)")
    BILL_AMT4: float = Field(..., description="Monto facturado en junio (NT$)")
    BILL_AMT5: float = Field(..., description="Monto facturado en mayo (NT$)")
    BILL_AMT6: float = Field(..., description="Monto facturado en abril (NT$)")

    # Montos de pago (PAY_AMT1 a PAY_AMT6)
    PAY_AMT1: float = Field(..., description="Monto pagado en septiembre (NT$)", ge=0)
    PAY_AMT2: float = Field(..., description="Monto pagado en agosto (NT$)", ge=0)
    PAY_AMT3: float = Field(..., description="Monto pagado en julio (NT$)", ge=0)
    PAY_AMT4: float = Field(..., description="Monto pagado en junio (NT$)", ge=0)
    PAY_AMT5: float = Field(..., description="Monto pagado en mayo (NT$)", ge=0)
    PAY_AMT6: float = Field(..., description="Monto pagado en abril (NT$)", ge=0)


class PredictionResponse(BaseModel):
    """Schema para respuesta de predicción."""

    probability: float = Field(..., description="Probabilidad de default (0-1)")
    prediction: str = Field(..., description="Clasificación: DEFAULT o NO_DEFAULT")
    risk_band: str = Field(..., description="Banda de riesgo: APROBADO, REVISION, RECHAZO")
    threshold_used: float = Field(..., description="Threshold usado para clasificación")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    model_version: str = Field(..., description="Versión del modelo")


class BatchPredictionRequest(BaseModel):
    """Schema para predicción batch."""
    applications: List[CreditApplication]


class BatchPredictionResponse(BaseModel):
    """Schema para respuesta de predicción batch."""
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Schema para health check."""
    status: str
    model_loaded: bool
    model_version: str
    threshold: float
    timestamp: str


class MetricsResponse(BaseModel):
    """Schema para métricas del modelo."""
    model_version: str
    threshold: float
    metrics: dict
    timestamp: str


class ScoreDriftRequest(BaseModel):
    """Payload para monitoreo de drift basado en scores."""
    scores: List[float] = Field(..., description="Scores actuales (probabilidades de default)")


# =====================================================
# CARGA DEL MODELO
# =====================================================

# Variables globales para el modelo
model = None
feature_names = None
model_metadata = None
baseline_scores = None


def load_model():
    """Carga el modelo y sus metadatos."""
    global model, feature_names, model_metadata

    try:
        # Cargar modelo
        model_path = MODELS_DIR / "final_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

        model = joblib.load(model_path)
        logger.info(f"Modelo cargado: {type(model).__name__}")
        MODEL_LOADED_GAUGE.set(1)

        # Cargar nombres de features
        features_path = MODELS_DIR / "feature_names.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Features cargadas: {len(feature_names)} variables")
        else:
            logger.warning("feature_names.json no encontrado, usando nombres por defecto")
            feature_names = None

        # Cargar metadata
        metadata_path = MODELS_DIR / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Metadata cargada: versión {model_metadata.get('version', 'unknown')}")
        else:
            model_metadata = {"version": "1.0.0"}

        return True

    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        MODEL_LOADED_GAUGE.set(0)
        return False


def compute_reference_scores():
    """
    Genera scores de referencia para monitoreo (PSI/KS) usando el dataset procesado.
    """
    if model is None:
        return None

    data_path = BASE_DIR / "data" / "processed" / "credit_data_processed.csv"
    if not data_path.exists():
        logger.warning("Dataset de referencia no encontrado para PSI/KS")
        return None

    df = pd.read_csv(data_path)

    # Remover targets si existen
    for target_col in ["default.payment.next.month", "default_flag"]:
        if target_col in df.columns:
            df = df.drop(columns=[target_col])

    if feature_names:
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0
        df = df[feature_names]

    scores = model.predict_proba(df)[:, 1]
    return scores


#
# Carga temprana del modelo (para que los tests y el arranque tengan modelo listo).
# Si falla, el evento de startup intentará nuevamente y dejará trazas en los logs.
#
if model is None:
    if not load_model():
        logger.warning("Cargado inicial de modelo fallido; se reintentará en startup")


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def engineer_features(data: dict) -> pd.DataFrame:
    """
    Aplica el mismo feature engineering usado en entrenamiento.

    Args:
        data: Diccionario con los datos de entrada

    Returns:
        DataFrame con features procesadas
    """
    df = pd.DataFrame([data])

    # Feature engineering básico
    # 1. Utilization ratio
    if df['LIMIT_BAL'].iloc[0] > 0:
        df['utilization_1'] = df['BILL_AMT1'] / df['LIMIT_BAL']
    else:
        df['utilization_1'] = 0

    # 2. Payment ratios
    for i in range(1, 7):
        bill_col = f'BILL_AMT{i}'
        pay_col = f'PAY_AMT{i}'
        ratio_col = f'payment_ratio_{i}'

        if df[bill_col].iloc[0] > 0:
            df[ratio_col] = df[pay_col] / df[bill_col]
        else:
            df[ratio_col] = 0

    # 3. Age bins (one-hot encoding)
    age = df['AGE'].iloc[0]
    df['AGE_bin_26-35'] = 1 if 26 <= age <= 35 else 0
    df['AGE_bin_36-45'] = 1 if 36 <= age <= 45 else 0
    df['AGE_bin_46-60'] = 1 if 46 <= age <= 60 else 0
    df['AGE_bin_60+'] = 1 if age > 60 else 0

    # 4. Education grouped
    education = df['EDUCATION'].iloc[0]
    if education in [0, 4, 5, 6]:
        df['EDUCATION_grouped'] = 4  # Otros
    else:
        df['EDUCATION_grouped'] = education

    # 5. Marriage grouped
    marriage = df['MARRIAGE'].iloc[0]
    if marriage == 0:
        df['MARRIAGE_grouped'] = 3  # Otros
    else:
        df['MARRIAGE_grouped'] = marriage

    # Seleccionar features en el orden correcto
    if feature_names:
        # Asegurar que todas las features existen
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0
        df = df[feature_names]

    return df


def get_risk_band(probability: float) -> str:
    """
    Clasifica la probabilidad en bandas de riesgo.

    Args:
        probability: Probabilidad de default

    Returns:
        Banda de riesgo
    """
    if probability < 0.20:
        return "APROBADO"
    elif probability < 0.50:
        return "REVISION"
    else:
        return "RECHAZO"


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Calcula PSI entre distribuciones esperada (baseline) y actual.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    expected_pct = (expected_counts + 1e-6) / len(expected)
    actual_pct = (actual_counts + 1e-6) / len(actual)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def require_api_key(request: Request):
    """
    Autenticación opcional via API Key.

    Si API_KEY está definida en entorno, se requiere header X-API-Key coincidente.
    """
    if API_KEY:
        key = request.headers.get("X-API-Key")
        if key != API_KEY:
            raise HTTPException(status_code=401, detail="API key inválida o ausente")
    return True


# =====================================================
# ENDPOINTS
# =====================================================


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información básica."""
    return {
        "message": "Credit Risk Scoring API",
        "version": "1.0.0",
        "author": "Ing. Daniel Varela Pérez",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Verifica el estado del servicio y el modelo.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_metadata.get("version", "unknown") if model_metadata else "unknown",
        threshold=OPTIMAL_THRESHOLD,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(application: CreditApplication, _: bool = Depends(require_api_key)):
    """
    Predice la probabilidad de default para una solicitud de crédito.

    - **probability**: Probabilidad de default (0-1)
    - **prediction**: DEFAULT o NO_DEFAULT según threshold
    - **risk_band**: APROBADO (<20%), REVISION (20-50%), RECHAZO (>50%)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        # Convertir a dict y aplicar feature engineering
        data = application.model_dump()
        features = engineer_features(data)

        # Predecir
        probability = model.predict_proba(features)[0, 1]

        # Clasificar
        prediction = "DEFAULT" if probability >= OPTIMAL_THRESHOLD else "NO_DEFAULT"
        risk_band = get_risk_band(probability)

        # Log para monitoreo
        logger.info(f"Prediction: prob={probability:.4f}, pred={prediction}, risk={risk_band}")
        PREDICTIONS_TOTAL.labels(mode="single").inc()

        return PredictionResponse(
            probability=round(float(probability), 4),
            prediction=prediction,
            risk_band=risk_band,
            threshold_used=OPTIMAL_THRESHOLD,
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get("version", "1.0.0") if model_metadata else "1.0.0"
        )

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest, _: bool = Depends(require_api_key)):
    """
    Predice para múltiples solicitudes de crédito.

    Máximo 100 solicitudes por batch.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    if len(request.applications) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 solicitudes por batch")

    try:
        predictions = []

        for app in request.applications:
            data = app.model_dump()
            features = engineer_features(data)

            probability = model.predict_proba(features)[0, 1]
            prediction = "DEFAULT" if probability >= OPTIMAL_THRESHOLD else "NO_DEFAULT"
            risk_band = get_risk_band(probability)

            predictions.append(PredictionResponse(
                probability=round(float(probability), 4),
                prediction=prediction,
                risk_band=risk_band,
                threshold_used=OPTIMAL_THRESHOLD,
                timestamp=datetime.now().isoformat(),
                model_version=model_metadata.get("version", "1.0.0") if model_metadata else "1.0.0"
            ))

        PREDICTIONS_TOTAL.labels(mode="batch").inc(len(predictions))

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics(_: bool = Depends(require_api_key)):
    """
    Obtiene las métricas de rendimiento del modelo.
    """
    try:
        # Cargar métricas guardadas
        metrics_path = REPORTS_DIR / "metrics" / "validation_results.json"

        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                validation_results = json.load(f)

            metrics = {
                "optimal_threshold": validation_results.get("optimal_threshold"),
                "auc_roc": validation_results.get("metrics_optimal", {}).get("auc_roc"),
                "recall": validation_results.get("metrics_optimal", {}).get("recall"),
                "precision": validation_results.get("metrics_optimal", {}).get("precision"),
                "f1": validation_results.get("metrics_optimal", {}).get("f1"),
                "ks": validation_results.get("metrics_optimal", {}).get("ks"),
                "brier": validation_results.get("metrics_optimal", {}).get("brier"),
                "cost_savings": validation_results.get("cost_savings")
            }
        else:
            metrics = {"message": "Métricas no disponibles"}

        return MetricsResponse(
            model_version=model_metadata.get("version", "1.0.0") if model_metadata else "1.0.0",
            threshold=OPTIMAL_THRESHOLD,
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")


@app.get("/prometheus", tags=["Monitoring"])
async def prometheus_metrics():
    """
    Exposición de métricas Prometheus.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/monitoring/drift", tags=["Monitoring"])
async def monitor_drift(request: ScoreDriftRequest, _: bool = Depends(require_api_key)):
    """
    Calcula PSI y KS contra los scores de referencia (baseline).
    """
    global baseline_scores
    if (baseline_scores is None or len(baseline_scores) == 0) and model is not None:
        baseline_scores = compute_reference_scores()

    if baseline_scores is None or len(baseline_scores) == 0:
        raise HTTPException(status_code=503, detail="Scores de referencia no disponibles")

    current_scores = np.asarray(request.scores, dtype=float)
    if current_scores.size == 0:
        raise HTTPException(status_code=400, detail="Se requieren scores actuales")

    psi_value = compute_psi(baseline_scores, current_scores)
    ks_stat = stats.ks_2samp(baseline_scores, current_scores).statistic

    status = "ok"
    if psi_value >= 0.25:
        status = "critical"
    elif psi_value >= 0.10:
        status = "warning"

    return {
        "status": status,
        "psi": round(psi_value, 4),
        "ks_statistic": round(float(ks_stat), 4),
        "thresholds": {
            "psi_warning": 0.10,
            "psi_critical": 0.25
        },
        "reference_size": len(baseline_scores),
        "current_size": int(current_scores.size),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Información detallada del modelo.
    """
    return {
        "model_type": type(model).__name__ if model else "Not loaded",
        "version": model_metadata.get("version", "unknown") if model_metadata else "unknown",
        "threshold": OPTIMAL_THRESHOLD,
        "features_count": len(feature_names) if feature_names else "unknown",
        "risk_bands": {
            "APROBADO": "PD < 20%",
            "REVISION": "20% <= PD < 50%",
            "RECHAZO": "PD >= 50%"
        },
        "training_date": model_metadata.get("training_date", "unknown") if model_metadata else "unknown"
    }


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
