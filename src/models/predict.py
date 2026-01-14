"""
============================================================================
predict.py - Prediction Pipeline para Credit Risk Scoring
============================================================================
Script de predicción/inferencia para nuevos clientes con manejo de bandas
de riesgo (APROBADO/REVISION/RECHAZO) y exportación de resultados.

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Tel: +52 55 4189 3428
Metodología: DVP-PRO
============================================================================
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/predict.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# RISK BANDS CLASSIFICATION
# ============================================================================


def classify_risk_band(
    probability: float, threshold_aprobado: float = 0.20, threshold_revision: float = 0.50
) -> str:
    """
    Clasifica cliente en bandas de riesgo basado en probabilidad de default.

    Bandas:
    - APROBADO: PD < 20% (bajo riesgo)
    - REVISION: 20% <= PD < 50% (riesgo medio, requiere análisis manual)
    - RECHAZO: PD >= 50% (alto riesgo)

    Args:
        probability: Probabilidad de default (0-1)
        threshold_aprobado: Threshold para clasificar como APROBADO
        threshold_revision: Threshold para clasificar como RECHAZO

    Returns:
        Banda de riesgo: 'APROBADO', 'REVISION', o 'RECHAZO'
    """
    if probability < threshold_aprobado:
        return "APROBADO"
    elif probability < threshold_revision:
        return "REVISION"
    else:
        return "RECHAZO"


def calculate_credit_limit(
    probability: float, base_limit: float = 50000, risk_band: str = None
) -> float:
    """
    Calcula límite de crédito sugerido basado en riesgo.

    Estrategia:
    - APROBADO: 80-100% del límite base
    - REVISION: 40-60% del límite base
    - RECHAZO: 0%

    Args:
        probability: Probabilidad de default
        base_limit: Límite base de crédito (MXN)
        risk_band: Banda de riesgo (opcional, se calcula si no se provee)

    Returns:
        Límite de crédito sugerido (MXN)
    """
    if risk_band is None:
        risk_band = classify_risk_band(probability)

    if risk_band == "APROBADO":
        # Límite alto para bajo riesgo
        factor = 1.0 - (probability / 0.20) * 0.2  # 0.8 - 1.0
        return base_limit * factor
    elif risk_band == "REVISION":
        # Límite reducido para riesgo medio
        factor = 0.6 - ((probability - 0.20) / 0.30) * 0.2  # 0.4 - 0.6
        return base_limit * factor
    else:  # RECHAZO
        return 0.0


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================


def load_model(model_path: str) -> BaseEstimator:
    """
    Carga modelo desde archivo joblib.

    Args:
        model_path: Path al modelo serializado

    Returns:
        Modelo cargado
    """
    logger.info(f"Loading model from: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    logger.info("Model loaded successfully")

    return model


def predict_single(
    model: BaseEstimator,
    features: pd.DataFrame,
    optimal_threshold: float = 0.12,
    threshold_aprobado: float = 0.20,
    threshold_revision: float = 0.50,
) -> Dict:
    """
    Genera predicción para un solo cliente.

    Args:
        model: Modelo entrenado
        features: DataFrame con features del cliente (1 fila)
        optimal_threshold: Threshold óptimo para clasificación binaria
        threshold_aprobado: Threshold para banda APROBADO
        threshold_revision: Threshold para banda RECHAZO

    Returns:
        Diccionario con resultados de predicción
    """
    # Probabilidad de default
    proba = model.predict_proba(features)[0, 1]

    # Clasificación binaria con threshold óptimo
    default_prediction = int(proba >= optimal_threshold)

    # Banda de riesgo
    risk_band = classify_risk_band(proba, threshold_aprobado, threshold_revision)

    # Límite de crédito sugerido
    credit_limit = calculate_credit_limit(proba, base_limit=50000, risk_band=risk_band)

    result = {
        "default_probability": float(proba),
        "default_prediction": default_prediction,
        "risk_band": risk_band,
        "suggested_credit_limit_mxn": float(credit_limit),
        "optimal_threshold_used": optimal_threshold,
    }

    return result


def predict_batch(
    model: BaseEstimator,
    data: pd.DataFrame,
    optimal_threshold: float = 0.12,
    threshold_aprobado: float = 0.20,
    threshold_revision: float = 0.50,
    include_features: bool = False,
) -> pd.DataFrame:
    """
    Genera predicciones para múltiples clientes.

    Args:
        model: Modelo entrenado
        data: DataFrame con features de clientes
        optimal_threshold: Threshold óptimo para clasificación binaria
        threshold_aprobado: Threshold para banda APROBADO
        threshold_revision: Threshold para banda RECHAZO
        include_features: Si incluir features originales en output

    Returns:
        DataFrame con predicciones
    """
    logger.info(f"Generating predictions for {len(data)} samples...")

    # Probabilidades
    probabilities = model.predict_proba(data)[:, 1]

    # Predicciones binarias
    predictions = (probabilities >= optimal_threshold).astype(int)

    # Bandas de riesgo
    risk_bands = [
        classify_risk_band(p, threshold_aprobado, threshold_revision) for p in probabilities
    ]

    # Límites de crédito
    credit_limits = [
        calculate_credit_limit(p, base_limit=50000, risk_band=rb)
        for p, rb in zip(probabilities, risk_bands)
    ]

    # Crear DataFrame de resultados
    results = pd.DataFrame(
        {
            "default_probability": probabilities,
            "default_prediction": predictions,
            "risk_band": risk_bands,
            "suggested_credit_limit_mxn": credit_limits,
        }
    )

    # Agregar timestamp
    results["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Agregar features originales si se solicita
    if include_features:
        results = pd.concat([data.reset_index(drop=True), results], axis=1)

    logger.info("Predictions completed successfully")

    return results


def generate_summary_stats(predictions: pd.DataFrame) -> Dict:
    """
    Genera estadísticas resumidas de las predicciones.

    Args:
        predictions: DataFrame con predicciones

    Returns:
        Diccionario con estadísticas
    """
    stats = {
        "total_predictions": len(predictions),
        "avg_default_probability": float(predictions["default_probability"].mean()),
        "median_default_probability": float(predictions["default_probability"].median()),
        "std_default_probability": float(predictions["default_probability"].std()),
        "total_defaults_predicted": int(predictions["default_prediction"].sum()),
        "default_rate": float(predictions["default_prediction"].mean()),
        # Distribución por bandas
        "count_aprobado": int((predictions["risk_band"] == "APROBADO").sum()),
        "count_revision": int((predictions["risk_band"] == "REVISION").sum()),
        "count_rechazo": int((predictions["risk_band"] == "RECHAZO").sum()),
        "pct_aprobado": float((predictions["risk_band"] == "APROBADO").mean() * 100),
        "pct_revision": float((predictions["risk_band"] == "REVISION").mean() * 100),
        "pct_rechazo": float((predictions["risk_band"] == "RECHAZO").mean() * 100),
        # Límites de crédito
        "avg_credit_limit": float(predictions["suggested_credit_limit_mxn"].mean()),
        "total_credit_exposure": float(predictions["suggested_credit_limit_mxn"].sum()),
    }

    return stats


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================


def save_predictions(
    predictions: pd.DataFrame, output_path: str, save_summary: bool = True
) -> None:
    """
    Guarda predicciones a archivo CSV.

    Args:
        predictions: DataFrame con predicciones
        output_path: Path de salida
        save_summary: Si guardar también archivo de resumen
    """
    # Crear directorio si no existe
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar predicciones
    predictions.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to: {output_path}")

    # Guardar resumen
    if save_summary:
        stats = generate_summary_stats(predictions)

        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CREDIT RISK PREDICTIONS - SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Predictions: {stats['total_predictions']:,}\n")
            f.write(
                f"Average Default Probability: {stats['avg_default_probability']:.2%}\n"
            )
            f.write(f"Total Defaults Predicted: {stats['total_defaults_predicted']:,}\n")
            f.write(f"Default Rate: {stats['default_rate']:.2%}\n\n")

            f.write("RISK BANDS DISTRIBUTION:\n")
            f.write(f"  - APROBADO: {stats['count_aprobado']:,} ({stats['pct_aprobado']:.1f}%)\n")
            f.write(f"  - REVISION: {stats['count_revision']:,} ({stats['pct_revision']:.1f}%)\n")
            f.write(f"  - RECHAZO:  {stats['count_rechazo']:,} ({stats['pct_rechazo']:.1f}%)\n\n")

            f.write(f"Average Credit Limit: ${stats['avg_credit_limit']:,.0f} MXN\n")
            f.write(f"Total Credit Exposure: ${stats['total_credit_exposure']:,.0f} MXN\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Summary saved to: {summary_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Entry point para CLI."""
    parser = argparse.ArgumentParser(
        description="Credit Risk Scoring - Prediction Pipeline (DVP-PRO)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path al dataset con features de clientes",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/final_model.joblib",
        help="Path al modelo entrenado",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="data/predictions/predictions_latest.csv",
        help="Path de salida para predicciones",
    )

    parser.add_argument(
        "--optimal-threshold",
        type=float,
        default=0.12,
        help="Threshold óptimo para clasificación binaria",
    )

    parser.add_argument(
        "--threshold-aprobado",
        type=float,
        default=0.20,
        help="Threshold para clasificar como APROBADO",
    )

    parser.add_argument(
        "--threshold-revision",
        type=float,
        default=0.50,
        help="Threshold para clasificar como RECHAZO",
    )

    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Incluir features originales en output",
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="No generar archivo de resumen",
    )

    args = parser.parse_args()

    # Crear directorio de logs
    Path("logs").mkdir(parents=True, exist_ok=True)

    try:
        logger.info("=" * 80)
        logger.info("CREDIT RISK SCORING - PREDICTION PIPELINE")
        logger.info("Metodología: DVP-PRO")
        logger.info("=" * 80)

        # Cargar modelo
        model = load_model(args.model_path)

        # Cargar datos
        logger.info(f"Loading data from: {args.data_path}")
        data = pd.read_csv(args.data_path)

        # Remover target si existe
        if "default.payment.next.month" in data.columns:
            logger.info("Removing target column from features")
            data = data.drop(columns=["default.payment.next.month"])

        logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")

        # Generar predicciones
        predictions = predict_batch(
            model=model,
            data=data,
            optimal_threshold=args.optimal_threshold,
            threshold_aprobado=args.threshold_aprobado,
            threshold_revision=args.threshold_revision,
            include_features=args.include_features,
        )

        # Guardar resultados
        save_predictions(predictions, args.output_path, save_summary=not args.no_summary)

        # Mostrar resumen en consola
        stats = generate_summary_stats(predictions)

        logger.info("=" * 80)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Predictions: {stats['total_predictions']:,}")
        logger.info(f"Default Rate: {stats['default_rate']:.2%}")
        logger.info(
            f"Risk Bands - APROBADO: {stats['pct_aprobado']:.1f}% | "
            f"REVISION: {stats['pct_revision']:.1f}% | "
            f"RECHAZO: {stats['pct_rechazo']:.1f}%"
        )
        logger.info(f"Total Credit Exposure: ${stats['total_credit_exposure']:,.0f} MXN")
        logger.info("=" * 80)

        logger.info("Prediction pipeline completed successfully! ✓")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
