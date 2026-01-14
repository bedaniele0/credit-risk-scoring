# F2 – Diseño Arquitectónico: Credit Card Default Risk Scoring (UCI Taiwan)

**Autor:** Ing. Daniel Varela Pérez  
**Versión:** 1.0  
**Fecha:** 2025‑11‑07

---

## 1) Diagrama de Alto Nivel
```mermaid
graph TD
  A[Data Raw (XLS/CSV)] --> B[Ingesta & Validación]
  B --> C[(Feature Store Offline
  Parquet + Catálogo)]
  C --> D[Entrenamiento ML
  (MLflow Tracking)]
  D --> E[(Model Registry)]
  E --> F[API Scoring (FastAPI)]
  F --> G[Apps Originación]
  F --> H[Batch Scoring]
  E --> I[Monitoreo Modelo]
  C --> J[Monitoreo Datos]
  I --> D
  J --> D
```

---

## 2) Componentes y Responsabilidades
- **Ingesta & Validación:** `src/data/make_dataset.py` + Great Expectations (validar esquema, nulos, rangos).
- **Feature Store (offline):** Parquet particionado (`data/processed/`) + `feature_catalog.csv`.
- **Entrenamiento:** scikit‑learn/LightGBM, CV 5‑fold, `mlflow` para parámetros, métricas y artefactos.
- **Registry:** `mlflow` Model Registry (staging → production).
- **Serving:** FastAPI (JSON in/out), p95 < 200 ms; serialización `joblib`.
- **Monitoreo:** Prometheus (latencia, error‑rate), métricas de drift (PSI, KS) y recalibración mensual.

---

## 3) Esquema de Datos Operativo (post‑ingesta)
- **ID** (int)
- **LIMIT_BAL, AGE** (num)
- **SEX, EDUCATION, MARRIAGE** (categorías codificadas)
- **PAY_0..PAY_6** (historial de retraso; ordinal)
- **BILL_AMT1..BILL_AMT6** (facturación mensual)
- **PAY_AMT1..PAY_AMT6** (pagos mensuales)
- **Target:** `default.payment.next.month` → `default_flag` (0/1)

**Transformaciones clave:**
- Imputación simple (mediana/moda), winsorization p1‑p99 en montos.
- Casting de `PAY_*` a categorías ordenadas si mejora.
- Derivadas iniciales: `utilization = BILL_AMT1 / LIMIT_BAL`, `payment_ratio_k = PAY_AMTk / BILL_AMTk`.

---

## 4) MLOps – Ciclo de Entrenamiento
1. `make data` → validaciones + export a Parquet.
2. `make features` → ingeniería + catálogo de features.
3. `make train` → CV 5‑fold, logging a MLflow.
4. `make eval` → métricas holdout, SHAP, curva KS/ROC, Model Card.
5. Registrar a **Staging**; si pasa umbrales, promover a **Production**.

---

## 5) Interfaz del Servicio (contrato API)
**POST /predict**
```json
{
  "LIMIT_BAL": 50000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 35,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": -1,
  "BILL_AMT1": 3913,
  "PAY_AMT1": 0
  // ... resto de campos requeridos
}
```
**Respuesta**
```json
{
  "probability": 0.27,
  "prediction": "DEFAULT",
  "risk_band": "REVISION",
  "threshold_used": 0.12,
  "timestamp": "2025-12-26T18:24:21.352644",
  "model_version": "1.0.0"
}
```

---

## 6) ADRs (resumen)
### ADR‑001: Familia de Modelo – **LightGBM**
- **Decisión:** Usar LightGBM por rendimiento en tabular, manejo de missing y velocidad.
- **Alternativas:** XGBoost, LR regularizada (baseline), CatBoost.
- **Consecuencia:** Ajuste de regularización y class_weight; calibración Platt/Isotónica.

### ADR‑002: Feature Store – **Parquet + Catálogo CSV**
- **Decisión:** Almacenamiento offline en Parquet + `feature_catalog.csv`.
- **Alternativas:** Feast/Hopsworks (overkill para escala S/M).
- **Consecuencia:** Simplicidad, bajo costo y trazabilidad.

### ADR‑003: Monitoreo – **Prometheus + PSI/KS**
- **Decisión:** Latencia/errores con Prometheus; drift con PSI (features) y KS (score).
- **Consecuencia:** Alertas y retraining programado mensual.

---

## 7) Definition of Done (F2)
- Diagrama aprobado y versionado en `docs/`.
- ADR‑001..003 firmados.
- Esquema de dataset y transformaciones mínimas definidos.
- Contrato API especificado.

---

## 8) Siguientes Pasos (F3/F4)
- Implementar `make_dataset.py` con validaciones.
- Definir `feature_catalog.csv` y primeras features derivadas.
- Preparar notebooks EDA y baseline (LR + LightGBM) con MLflow.
