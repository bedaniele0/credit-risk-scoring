# F1 - Setup del Proyecto - Credit Risk Scoring

**Autor:** Ing. Daniel Varela Pérez
**Email:** bedaniele0@gmail.com
**Tel:** +52 55 4189 3428
**Metodología:** DVP-PRO (Fase 1)

## 1. Objetivo de la Fase

Configurar el entorno de desarrollo, instalar dependencias y validar que todos los componentes del proyecto Credit Risk Scoring están listos para uso.

## 2. Requisitos del Sistema

### Hardware Mínimo
- CPU: 2 cores
- RAM: 4 GB mínimo (8 GB recomendado)
- Disco: 2 GB disponibles

### Software Requerido
- **Python 3.10+** (compatible con 3.11, 3.12, 3.14)
- pip (gestor de paquetes)
- Git (opcional, para control de versiones)

## 3. Instalación del Entorno

### Paso a Paso

```bash
# 1. Navegar al directorio del proyecto
cd ~/Desktop/credit-risk-scoring

# 2. Crear entorno virtual con Python 3.10+
python3 -m venv venv

# 3. Activar entorno virtual
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 4. Actualizar pip
pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

**ML/Data Science:**
- scikit-learn==1.7.2 - Pipelines y modelo (versión fija)
- imbalanced-learn - SMOTE para balanceo de clases
- pandas - Manipulación de datos
- numpy - Operaciones numéricas

**API & Web:**
- fastapi - Framework API REST
- uvicorn - ASGI server
- streamlit - Dashboard interactivo
- pydantic - Validación de datos

**Visualización:**
- plotly - Gráficos interactivos
- matplotlib - Gráficos estáticos
- seaborn - Visualizaciones estadísticas

**MLOps:**
- mlflow - Experiment tracking y model registry

**Testing:**
- pytest - Framework de testing
- pytest-cov - Coverage reporting
- pytest-asyncio - Tests asíncronos

**Utils:**
- joblib - Serialización de modelos
- pyyaml - Configuración (config.yaml)

## 4. Validación de Setup

### Validación Manual

```bash
# 1. Verificar versión de Python
python3 --version
# Output esperado: Python 3.10.x o superior

# 2. Verificar dependencias clave
python3 -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python3 -c "import pandas; print(f'pandas: {pandas.__version__}')"

# 3. Verificar modelos entrenados
ls -lh models/*.joblib
# Output esperado:
# final_model.joblib (~11 MB)
```

### Ejecutar Tests

```bash
# Ejecutar suite completa de tests
pytest tests/ -v

# Output esperado: Todos los tests passing
```

## 5. Estructura del Proyecto Creada

```
credit-risk-scoring/
├── config/
│   └── config.yaml               # Configuración centralizada
├── data/
│   ├── raw/
│   │   ├── default of credit card clients.csv  # Dataset UCI (CSV limpio)
│   │   └── default of credit card clients.xls  # Dataset UCI original
│   └── processed/                # Datos procesados (train/test)
│       ├── credit_data_processed.csv
│       ├── X_train.csv, X_test.csv
│       ├── y_train.csv, y_test.csv
│       └── *.parquet             # Versiones Parquet
├── models/
│   ├── final_model.joblib        # Modelo CalibratedClassifierCV (~11 MB)
│   ├── model_metadata.json       # Metadatos del modelo
│   ├── feature_names.json        # Features utilizadas
│   └── final_metrics.json        # Métricas del modelo
├── reports/
│   └── metrics/
│       └── validation_results.json   # Resultados de validación
├── src/
│   ├── api/
│   │   └── main.py               # API REST endpoints
│   ├── data/
│   │   └── make_dataset.py       # Procesamiento de datos
│   ├── features/
│   │   └── build_features.py     # Feature engineering
│   ├── models/
│   │   ├── train_credit.py       # Entrenamiento de modelos
│   │   ├── evaluate.py           # Evaluación de modelos
│   │   ├── predict.py            # Predicciones
│   │   └── mlflow_utils.py       # Utilidades MLflow
│   ├── monitoring/
│   │   └── drift_monitor.py      # Monitoreo de drift
│   └── visualization/
│       └── dashboard.py          # Dashboard Streamlit
├── tests/
│   ├── api/test_endpoints.py     # Tests de API
│   ├── unit/test_feature_engineering.py  # Tests de features
│   ├── unit/test_monitoring.py   # Tests de monitoreo
│   └── integration/test_api_integration.py  # Tests de integración
├── notebooks/
│   ├── 01_eda.ipynb              # Análisis exploratorio
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── docs/                         # Documentación DVP-PRO
│   ├── F0_problem_statement.md
│   ├── F1_setup.md               # Este documento
│   ├── F2_architecture.md
│   └── F7_deployment.md
├── mlruns/                       # MLflow experiment tracking
├── requirements.txt              # Dependencias Python
├── Dockerfile                    # Docker build
├── docker-compose.yml            # Orquestación Docker
└── README.md                     # Documentación principal
```

## 6. Ejecución de Componentes

### Entrenamiento del Modelo

```bash
# Entrenamiento estándar
python3 -m src.models.train_credit --data_path data/processed/credit_data_processed.csv

# Output esperado:
# Training model...
# Best parameters found: {...}
# Validation AUC: 0.7813
# Model saved: models/final_model.joblib
```

### API REST

```bash
# Terminal 1 - Levantar API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Output esperado:
# INFO: Uvicorn running on http://0.0.0.0:8000
# INFO: Model loaded successfully
# INFO: Application startup complete
```

**Acceder a la documentación:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Dashboard Streamlit

```bash
# Terminal 2 - Levantar Dashboard
streamlit run src/visualization/dashboard.py

# Output esperado:
# Local URL: http://localhost:8501
# INFO: Model loaded successfully
```

**Acceder al dashboard:**
- URL: http://localhost:8501

### MLflow UI

```bash
# Terminal 3 - MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Output esperado:
# INFO: Listening at: http://127.0.0.1:5000
```

**Acceder a MLflow:**
- URL: http://localhost:5000

### Monitoreo de Drift

```bash
# Ejecutar monitoreo de drift
python3 src/monitoring/drift_monitor.py

# Output esperado:
# PSI/KS scores calculated
# Drift report saved: reports/monitoring/drift_report_YYYYMMDD.json
```

## 7. Testing

### Ejecutar Suite Completa

```bash
# Tests con reporte verbose
pytest tests/ -v

# Tests con coverage
pytest tests/ -v --cov=src

# Tests específicos
pytest tests/api/test_endpoints.py -v
pytest tests/unit/test_feature_engineering.py -v
pytest tests/unit/test_monitoring.py -v
pytest tests/integration/test_api_integration.py -v
```

### Resultados Esperados

```
============================================================
test session starts
============================================================
platform darwin -- Python 3.13.x, pytest-9.0.2
plugins: anyio-4.12.0, asyncio-1.3.0, cov-7.0.0

tests/api/test_endpoints.py::... PASSED
tests/unit/test_feature_engineering.py::... PASSED
tests/unit/test_monitoring.py::... PASSED
tests/integration/test_api_integration.py::... PASSED

============================================================
All tests passed
============================================================
```

## 8. Configuración de Variables (config.yaml)

El proyecto usa `config/config.yaml` para centralizar parámetros:

```yaml
project:
  name: "credit-risk-scoring"
  version: "0.1.0"
  author: "Ing. Daniel Varela Perez"
  email: "bedaniele0@gmail.com"

paths:
  raw_data: "data/raw/default of credit card clients.csv"
  processed_data: "data/processed"
  models: "models"
  reports: "reports"

model:
  random_state: 42
  test_size: 0.2
  cv_folds: 5

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

target_variable: "default.payment.next.month"
```

## 9. Comandos Básicos del Proyecto

### Ciclo de Vida del Modelo

```bash
# 1. Procesar datos raw
python3 -m src.data.make_dataset

# 2. Construir features
python3 -m src.features.build_features

# 3. Entrenar modelo
python3 -m src.models.train_credit --data_path data/processed/credit_data_processed.csv

# 4. Evaluar modelo
python3 -m src.models.evaluate

# 5. Levantar API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 6. Levantar Dashboard
streamlit run src/visualization/dashboard.py

# 7. Monitorear drift
python3 src/monitoring/drift_monitor.py
```

## 10. Troubleshooting

### Error: "Python version mismatch"

**Problema:** Usando versión incompatible de Python

**Solución:**
```bash
# Desactivar venv actual
deactivate

# Crear nuevo venv con Python 3.10+
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "scikit-learn version mismatch"

**Problema:** Versión de scikit-learn incompatible con modelos guardados

**Solución:**
```bash
# Reinstalar con versión específica
pip install scikit-learn==1.7.2 --force-reinstall
```

### Error: "Module not found"

**Problema:** Dependencia no instalada

**Solución:**
```bash
# Reinstalar todas las dependencias
pip install -r requirements.txt --force-reinstall
```

### Error: "Port already in use"

**Problema:** Puerto 8000, 8501 o 5000 ocupado

**Solución API:**
```bash
# Cambiar puerto
uvicorn src.api.main:app --host 0.0.0.0 --port 8010 --reload
```

**Solución Dashboard:**
```bash
streamlit run src/visualization/dashboard.py --server.port 8510
```

**Solución MLflow:**
```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```

### Error: "Model file not found"

**Problema:** Modelos no entrenados

**Solución:**
```bash
# Entrenar modelos
python3 -m src.models.train_credit --data_path data/processed/credit_data_processed.csv

# Verificar creación
ls -lh models/*.joblib
```

### Warning: "Matplotlib cache not writable"

**Problema:** Warnings de Matplotlib en sandbox

**Solución:**
```bash
# Exportar variable de entorno
export MPLCONFIGDIR=/tmp/matplotlib

# Y ejecutar comando
python3 -m src.models.train_credit --data_path data/processed/credit_data_processed.csv
```

## 11. Docker Deployment

### Build de la Imagen

```bash
# Build de imagen Docker
docker build -t credit-risk-api:1.0.0 .
```

### Ejecutar Contenedor

```bash
# Run contenedor
docker run -d -p 8000:8000 credit-risk-api:1.0.0
```

### Docker Compose (API + Prometheus)

```bash
# Levantar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

**Servicios expuestos:**
- API: http://localhost:8000
- Prometheus: http://localhost:9090

## 12. Checklist de Setup Completado

- [x] Python 3.10+ instalado y verificado
- [x] Entorno virtual creado y activado
- [x] Dependencias instaladas (todas)
- [x] Tests ejecutados (todos pasando)
- [x] Modelos pre-entrenados verificados
- [x] API funcional (http://localhost:8000)
- [x] Dashboard funcional (http://localhost:8501)
- [x] MLflow UI funcional (http://localhost:5000)
- [x] Monitoreo de drift ejecutable

## 13. Métricas del Modelo (Referencia Rápida)

**Modelo:** CalibratedClassifierCV
**Threshold Óptimo:** 0.12

| Métrica | Threshold=0.12 | Threshold=0.50 | Meta |
|---------|----------------|----------------|------|
| **AUC-ROC** | 0.7813 | 0.7813 | ≥0.80 ⚠️ |
| **KS** | 0.4251 | 0.4251 | ≥0.30 ✅ |
| **Recall** | 0.8704 | 0.3715 | ≥0.70 ✅ |
| **Precision** | 0.3107 | 0.6591 | ≥0.30 ✅ |
| **F1-Score** | 0.4579 | 0.4752 | - |
| **Brier Score** | 0.1349 | 0.1349 | ≤0.20 ✅ |

**Cost Savings:** $5,466,000 MXN (ver `reports/metrics/validation_results.json`)

## 14. Próximos Pasos

Una vez completado el setup:
1. Revisar F0 (Problem Statement)
2. Revisar F2 (Architecture)
3. Explorar F3 (Data Quality Report) - *Pendiente de crear*
4. Ejecutar notebooks de análisis
5. Explorar dashboard y API
6. Revisar F7 (Deployment)

## 15. Recursos Adicionales

- **README.md** - Guía rápida de inicio
- **config/config.yaml** - Parámetros configurables
- **Notebooks** - Análisis exploratorios en `notebooks/`
- **API Docs** - http://localhost:8000/docs (Swagger)

---

**Setup completado por:**
**Ing. Daniel Varela Pérez**
Senior Data Scientist & ML Engineer
📧 bedaniele0@gmail.com | 📱 +52 55 4189 3428

**Metodología:** DVP-PRO
**Fecha:** Diciembre 2024
**Versión:** 1.0
