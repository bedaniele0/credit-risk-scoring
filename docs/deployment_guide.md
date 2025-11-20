# Guía de Deployment - Credit Risk Scoring API

**Proyecto:** Credit Risk Scoring - UCI Taiwan Dataset
**Fase DVP-PRO:** F8 - Productización
**Autor:** Ing. Daniel Varela Pérez
**Email:** bedaniele0@gmail.com
**Tel:** +52 55 4189 3428
**Fecha:** 2025-11-18
**Versión:** 1.0.0

---

## 1. Descripción General

Esta guía describe el proceso de deployment del modelo de Credit Risk Scoring desarrollado con la metodología DVP-PRO.

### Arquitectura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Cliente       │────▶│   FastAPI       │────▶│   LightGBM      │
│   (HTTP/REST)   │     │   (Uvicorn)     │     │   (Calibrado)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   Monitoring    │
                        │   (PSI/KS)      │
                        └─────────────────┘
```

### Componentes

- **API REST**: FastAPI con Uvicorn
- **Modelo**: LightGBM + Calibración Isotónica
- **Monitoreo**: PSI y KS drift detection
- **Containerización**: Docker

---

## 2. Requisitos Previos

### Software

- Python 3.11+
- Docker 20.10+
- Docker Compose 2.0+

### Dependencias Python

```bash
pip install fastapi uvicorn joblib pandas numpy scikit-learn lightgbm
```

### Archivos Requeridos

```
models/
├── final_model.joblib       # Modelo serializado
├── feature_names.json       # Lista de features
└── model_metadata.json      # Metadata del modelo

data/processed/
├── X_train.csv              # Datos de referencia
└── X_test.csv               # Datos de test

reports/metrics/
└── validation_results.json  # Métricas de validación
```

---

## 3. Instalación Local

### 3.1 Clonar Repositorio

```bash
git clone <repository-url>
cd credit-risk-scoring
```

### 3.2 Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3.3 Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3.4 Ejecutar API Localmente

```bash
# Opción 1: Directo
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Opción 2: Python
python src/api/main.py
```

### 3.5 Verificar Instalación

```bash
# Health check
curl http://localhost:8000/health

# Documentación
open http://localhost:8000/docs
```

---

## 4. Deployment con Docker

### 4.1 Build de la Imagen

```bash
docker build -t credit-risk-api:1.0.0 .
```

### 4.2 Ejecutar Container

```bash
docker run -d \
  --name credit-risk-api \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/reports/monitoring:/app/reports/monitoring \
  credit-risk-api:1.0.0
```

### 4.3 Docker Compose

```bash
# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f api

# Detener servicios
docker-compose down
```

### 4.4 Verificar Container

```bash
# Estado del container
docker ps

# Health check
curl http://localhost:8000/health

# Logs
docker logs credit-risk-api
```

---

## 5. Configuración de la API

### 5.1 Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Nivel de logging | INFO |
| `MODEL_PATH` | Ruta al modelo | models/final_model.joblib |
| `THRESHOLD` | Threshold de clasificación | 0.12 |

### 5.2 Threshold de Decisión

El threshold óptimo es **0.12** (optimizado en F7). Para modificar:

```python
# src/api/main.py
OPTIMAL_THRESHOLD = 0.12  # Cambiar según necesidad
```

### 5.3 Bandas de Riesgo

| Banda | Probabilidad | Acción |
|-------|--------------|--------|
| APROBADO | < 20% | Aprobar solicitud |
| REVISION | 20% - 50% | Revisión manual |
| RECHAZO | ≥ 50% | Rechazar solicitud |

---

## 6. Uso de la API

### 6.1 Endpoints Disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Información básica |
| GET | `/health` | Health check |
| POST | `/predict` | Predicción individual |
| POST | `/predict/batch` | Predicción batch |
| GET | `/metrics` | Métricas del modelo |
| GET | `/model/info` | Información del modelo |

### 6.2 Ejemplo de Predicción Individual

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

**Respuesta:**

```json
{
  "probability": 0.1523,
  "prediction": "DEFAULT",
  "risk_band": "APROBADO",
  "threshold_used": 0.12,
  "timestamp": "2025-11-18T15:30:00",
  "model_version": "1.0.0"
}
```

### 6.3 Ejemplo de Predicción Batch

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "applications": [
      { /* aplicación 1 */ },
      { /* aplicación 2 */ }
    ]
  }'
```

### 6.4 Python Client

```python
import requests

def predict_credit_risk(data: dict) -> dict:
    """Predice riesgo crediticio via API."""
    response = requests.post(
        "http://localhost:8000/predict",
        json=data
    )
    response.raise_for_status()
    return response.json()

# Ejemplo de uso
result = predict_credit_risk({
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    # ... resto de campos
})

print(f"Probabilidad: {result['probability']}")
print(f"Decisión: {result['prediction']}")
print(f"Banda: {result['risk_band']}")
```

---

## 7. Monitoreo

### 7.1 Ejecutar Monitor de Drift

```bash
python src/monitoring/drift_monitor.py
```

### 7.2 Métricas Monitoreadas

| Métrica | Descripción | Threshold |
|---------|-------------|-----------|
| PSI | Population Stability Index | < 0.10 OK, < 0.25 Warning, ≥ 0.25 Critical |
| KS Decay | Degradación del estadístico KS | < 10% |
| CSI | Characteristic Stability Index (por feature) | Mismo que PSI |

### 7.3 Interpretar Resultados

**PSI < 0.10**: Distribución estable, no hay acción requerida
**0.10 ≤ PSI < 0.25**: Cambio moderado, investigar causa
**PSI ≥ 0.25**: Cambio significativo, considerar reentrenamiento

### 7.4 Automatizar Monitoreo

```bash
# Cron job (Linux/Mac)
0 6 * * * /path/to/venv/bin/python /path/to/src/monitoring/drift_monitor.py

# Task Scheduler (Windows)
schtasks /create /tn "CreditRiskMonitor" /tr "python src/monitoring/drift_monitor.py" /sc daily /st 06:00
```

---

## 8. Métricas del Modelo

### 8.1 Rendimiento (Test Set)

| Métrica | Valor | Meta | Estado |
|---------|-------|------|--------|
| AUC-ROC | 0.7813 | ≥ 0.80 | ⚠️ |
| KS | 0.4251 | ≥ 0.30 | ✅ |
| Recall | 0.8704 | ≥ 0.70 | ✅ |
| Precision | 0.3107 | ≥ 0.30 | ✅ |
| Brier | 0.1349 | ≤ 0.20 | ✅ |

### 8.2 Intervalos de Confianza (Bootstrap 95%)

- **Recall**: 0.8708 [0.8523, 0.8874]
- **Precision**: 0.3106 [0.2961, 0.3257]
- **AUC-ROC**: 0.7816 [0.7662, 0.7961]

---

## 9. Troubleshooting

### 9.1 Errores Comunes

**Error: Modelo no encontrado**
```
FileNotFoundError: models/final_model.joblib
```
Solución: Verificar que el archivo existe y la ruta es correcta.

**Error: Features no coinciden**
```
ValueError: Feature names mismatch
```
Solución: Verificar que `feature_names.json` contiene las mismas features usadas en training.

**Error: Puerto ocupado**
```
Address already in use
```
Solución: Cambiar puerto o matar proceso anterior:
```bash
lsof -i :8000 | grep LISTEN
kill -9 <PID>
```

### 9.2 Logs

```bash
# Ver logs del container
docker logs credit-risk-api -f

# Logs de la aplicación
tail -f logs/app.log
```

---

## 10. Seguridad

### 10.1 Recomendaciones

1. **Autenticación**: Implementar API keys o OAuth2
2. **Rate Limiting**: Limitar requests por IP
3. **HTTPS**: Usar certificado SSL en producción
4. **Secrets**: No hardcodear credentials

### 10.2 Ejemplo con API Key

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
```

---

## 11. CI/CD Pipeline

### 11.1 GitHub Actions (Ejemplo)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t credit-risk-api:${{ github.sha }} .
      - name: Push to Registry
        run: |
          docker tag credit-risk-api:${{ github.sha }} registry/credit-risk-api:latest
          docker push registry/credit-risk-api:latest
```

---

## 12. Mantenimiento

### 12.1 Calendario de Mantenimiento

| Frecuencia | Tarea |
|------------|-------|
| Diario | Revisar logs y alertas |
| Semanal | Monitoreo de drift (PSI, KS) |
| Mensual | Análisis de rendimiento |
| Trimestral | Evaluar necesidad de reentrenamiento |

### 12.2 Triggers de Reentrenamiento

1. PSI > 0.25
2. KS decay > 10%
3. Recall < 0.60
4. Cambio significativo en distribución de datos

---

## 13. Contacto y Soporte

**Desarrollador:** Ing. Daniel Varela Pérez
**Email:** bedaniele0@gmail.com
**Tel:** +52 55 4189 3428
**Metodología:** DVP-PRO v2.0

---

## Apéndice A: Estructura del Proyecto

```
credit-risk-scoring/
├── config/                  # Configuraciones
├── data/
│   ├── raw/                # Datos originales
│   └── processed/          # Datos procesados
├── docs/                    # Documentación
├── models/                  # Modelos serializados
├── notebooks/              # Jupyter notebooks
├── reports/
│   ├── figures/            # Visualizaciones
│   ├── metrics/            # Métricas
│   └── monitoring/         # Reportes de drift
├── src/
│   ├── api/               # FastAPI application
│   └── monitoring/        # Scripts de monitoreo
├── tests/                  # Tests unitarios
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

**Documento generado automáticamente como parte de la Fase F8 (Productización) de la metodología DVP-PRO.**
