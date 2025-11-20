# Credit Risk Scoring - UCI Taiwan Dataset

**Proyecto de Machine Learning para predicción de riesgo crediticio**

Desarrollado siguiendo la metodología **DVP-PRO** (Daniel Varela Pérez - Professional)

---

## Descripción

Sistema de scoring crediticio que predice la probabilidad de default en tarjetas de crédito, basado en el dataset UCI "Default of Credit Card Clients" de Taiwan (2005).

### Características Principales

- **Modelo**: LightGBM con calibración isotónica
- **API REST**: FastAPI para servir predicciones
- **Monitoreo**: PSI y KS drift detection
- **Threshold optimizado**: 0.12 (basado en costos de negocio)

---

## Métricas del Modelo

| Métrica | Valor | Meta | Estado |
|---------|-------|------|--------|
| AUC-ROC | 0.7813 | ≥ 0.80 | ⚠️ |
| KS | 0.4251 | ≥ 0.30 | ✅ |
| Recall | 0.8704 | ≥ 0.70 | ✅ |
| Precision | 0.3107 | ≥ 0.30 | ✅ |
| Brier Score | 0.1349 | ≤ 0.20 | ✅ |

**Ahorro estimado**: $5,466,000 vs threshold por defecto (0.5)

---

## Estructura del Proyecto

```
credit-risk-scoring/
├── config/                  # Configuraciones
├── data/
│   ├── raw/                # Datos originales
│   └── processed/          # Datos procesados
├── docs/                    # Documentación
│   └── deployment_guide.md # Guía de deployment
├── models/                  # Modelos serializados
│   ├── final_model.joblib
│   ├── feature_names.json
│   └── model_metadata.json
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_validation.ipynb
├── reports/
│   ├── figures/            # Visualizaciones
│   ├── metrics/            # Métricas JSON
│   └── monitoring/         # Reportes de drift
├── src/
│   ├── api/               # FastAPI application
│   │   └── main.py
│   └── monitoring/        # Scripts de monitoreo
│       └── drift_monitor.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Instalación

### Requisitos

- Python 3.11+
- pip

### Configuración

```bash
# Clonar repositorio
git clone <repository-url>
cd credit-risk-scoring

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### API REST

```bash
# Iniciar servidor
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Documentación interactiva
open http://localhost:8000/docs
```

### Ejemplo de Predicción

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Respuesta**:
```json
{
  "probability": 0.1353,
  "prediction": "DEFAULT",
  "risk_band": "APROBADO",
  "threshold_used": 0.12,
  "model_version": "1.0.0"
}
```

### Bandas de Riesgo

| Banda | Probabilidad | Acción |
|-------|--------------|--------|
| APROBADO | < 20% | Aprobar solicitud |
| REVISION | 20% - 50% | Revisión manual |
| RECHAZO | ≥ 50% | Rechazar solicitud |

---

## Docker

```bash
# Build
docker build -t credit-risk-api:1.0.0 .

# Run
docker run -d -p 8000:8000 credit-risk-api:1.0.0

# Docker Compose
docker-compose up -d
```

---

## Monitoreo

```bash
# Ejecutar monitoreo de drift
python src/monitoring/drift_monitor.py
```

### Métricas Monitoreadas

- **PSI** (Population Stability Index): Drift en distribución de scores
- **KS Decay**: Degradación del poder discriminativo
- **CSI** (Characteristic Stability Index): Drift por feature

### Thresholds de Alerta

| Métrica | OK | Warning | Critical |
|---------|----|---------|---------|
| PSI | < 0.10 | 0.10 - 0.25 | ≥ 0.25 |
| KS Decay | < 10% | - | ≥ 10% |

---

## Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Información básica |
| GET | `/health` | Health check |
| POST | `/predict` | Predicción individual |
| POST | `/predict/batch` | Predicción batch (max 100) |
| GET | `/metrics` | Métricas del modelo |
| GET | `/model/info` | Información del modelo |

---

## Metodología DVP-PRO

Este proyecto sigue la metodología DVP-PRO de 10 fases:

| Fase | Descripción | Estado |
|------|-------------|--------|
| F0 | Definición del Problema | ✅ |
| F1 | Adquisición de Datos | ✅ |
| F2 | Exploración de Datos (EDA) | ✅ |
| F3 | Preparación de Datos | ✅ |
| F4 | Feature Engineering | ✅ |
| F5 | Modelado y Experimentación | ✅ |
| F6 | Selección del Modelo | ✅ |
| F7 | Evaluación y Validación | ✅ |
| F8 | Productización | ✅ |
| F9 | Monitoreo y Mantenimiento | ✅ |

---

## Dataset

**Fuente**: UCI Machine Learning Repository

**Referencia**: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

**URL**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

### Variables

- **Demográficas**: SEX, EDUCATION, MARRIAGE, AGE
- **Financieras**: LIMIT_BAL, BILL_AMT1-6, PAY_AMT1-6
- **Historial de pago**: PAY_0, PAY_2-6
- **Target**: default.payment.next.month (0/1)

---

## Tecnologías

- **ML**: LightGBM, XGBoost, Scikit-learn
- **Optimización**: Optuna
- **API**: FastAPI, Uvicorn
- **Tracking**: MLflow
- **Interpretabilidad**: SHAP
- **Containerización**: Docker

---

## Autor

**Ing. Daniel Varela Pérez**

- Email: bedaniele0@gmail.com
- Tel: +52 55 4189 3428

---

## Licencia

Este proyecto es para fines educativos y de demostración.

---

## Documentación Adicional

- [Guía de Deployment](docs/deployment_guide.md)
- [Model Card](reports/model_card.md)
- [Reporte de Validación](notebooks/validation_report.md)

---

**Versión**: 1.0.0
**Fecha**: 2025-11-18
**Metodología**: DVP-PRO v2.0
