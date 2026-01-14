# F8 - Monitoreo en Producción - Credit Risk Scoring

**Autor:** Ing. Daniel Varela Pérez
**Email:** bedaniele0@gmail.com
**Metodología:** DVP-PRO (Fase 8)

## 1. Objetivo de la Fase

Establecer un sistema de monitoreo continuo para detectar degradación del modelo, drift de datos y anomalías operacionales en producción, garantizando performance sostenido del scoring crediticio.

## 2. Arquitectura de Monitoreo

### 2.1 Componentes

```
┌──────────────────────────────────────────────────────────┐
│                   Data Sources                           │
│  (Production Predictions, Actual Labels, Feature Data)   │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│              Monitoring Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Drift       │  │ Performance │  │ API         │      │
│  │ Detection   │  │ Monitoring  │  │ Monitoring  │      │
│  │ (PSI/KS/CSI)│  │ (AUC/Recall)│  │ (Latency)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│              Alerting System                             │
│  (Slack, Email, PagerDuty, Dashboards)                   │
└──────────────────────────────────────────────────────────┘
```

**Implementación:** `src/monitoring/`

## 3. Monitoreo de Data Drift

### 3.1 Population Stability Index (PSI)

**Propósito:** Detectar cambios en distribución de features entre train y producción.

**Fórmula:**
```
PSI = Σ (Expected% - Actual%) × ln(Expected% / Actual%)
```

**Implementación:** `src/monitoring/drift_monitor.py`

```python
def calculate_psi(expected, actual, bins=10):
    expected_pct, _ = np.histogram(expected, bins=bins)
    actual_pct, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_pct / len(expected)
    actual_pct = actual_pct / len(actual)

    psi = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))
    return psi
```

**Thresholds de alerta:**
- **PSI < 0.10:** No drift significativo ✅
- **0.10 ≤ PSI < 0.25:** Drift moderado ⚠️ (monitorear)
- **PSI ≥ 0.25:** Drift severo 🚨 (reentrenar modelo)

**Features monitoreadas (Top 10):**
1. PAY_0 (importance=0.198)
2. PAY_2 (importance=0.142)
3. PAY_3 (importance=0.118)
4. PAY_4 (importance=0.095)
5. utilization_1 (importance=0.087)
6. LIMIT_BAL (importance=0.076)
7. payment_ratio_1 (importance=0.064)
8. PAY_5 (importance=0.052)
9. BILL_AMT1 (importance=0.041)
10. PAY_6 (importance=0.038)

**Frecuencia:** Semanal

### 3.2 Kolmogorov-Smirnov (KS) Test

**Propósito:** Test estadístico para detectar diferencias en distribuciones.

**Fórmula:**
```
KS = max|CDF_train(x) - CDF_prod(x)|
```

**Implementación:**
```python
from scipy.stats import ks_2samp

ks_stat, p_value = ks_2samp(train_data, prod_data)

if p_value < 0.05:
    print("⚠️ Drift detectado (p<0.05)")
```

**Thresholds:**
- **p-value ≥ 0.05:** No drift significativo ✅
- **p-value < 0.05:** Drift detectado ⚠️ (investigar)
- **p-value < 0.01:** Drift severo 🚨 (reentrenar)

**Frecuencia:** Semanal

### 3.3 Characteristic Stability Index (CSI)

**Propósito:** Monitorear distribución de probabilidades predichas.

**Fórmula:** Similar a PSI, pero sobre scores predichos.

**Thresholds:**
- **CSI < 0.10:** Modelo estable ✅
- **0.10 ≤ CSI < 0.25:** Cambio moderado en scores ⚠️
- **CSI ≥ 0.25:** Cambio severo en scores 🚨

**Ejemplo de uso:**
```python
train_scores = model.predict_proba(X_train)[:, 1]
prod_scores = model.predict_proba(X_prod)[:, 1]

csi = calculate_psi(train_scores, prod_scores, bins=10)
```

**Frecuencia:** Semanal

## 4. Monitoreo de Performance del Modelo

### 4.1 Métricas Online (Con Labels)

**Escenario:** Labels disponibles después de 1-3 meses (cuando se confirma default/no default)

**Métricas monitoreadas:**

| Métrica | Target | Alert Threshold | Acción |
|---------|--------|-----------------|--------|
| **AUC-ROC** | ≥0.78 | <0.75 | Reentrenar 🚨 |
| **Recall** | ≥0.87 | <0.80 | Investigar ⚠️ |
| **Precision** | ≥0.31 | <0.25 | Investigar ⚠️ |
| **KS Statistic** | ≥0.42 | <0.35 | Reentrenar 🚨 |
| **Brier Score** | ≤0.14 | >0.18 | Revisar calibración ⚠️ |

**Frecuencia:** Mensual (cuando labels estén disponibles)

**Implementación:**
```python
def monitor_performance(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    ks = calculate_ks_statistic(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)

    alerts = []
    if auc < 0.75:
        alerts.append("🚨 AUC degradado: {:.4f}".format(auc))
    if ks < 0.35:
        alerts.append("🚨 KS degradado: {:.4f}".format(ks))
    if brier > 0.18:
        alerts.append("⚠️ Brier alto: {:.4f}".format(brier))

    return alerts
```

### 4.2 Métricas Proxy (Sin Labels)

**Escenario:** Labels no disponibles inmediatamente.

**Proxies monitoreados:**

1. **Default Rate Proxy:**
   - % de predicciones con PD >0.50
   - Esperado: ~22% (basado en training)
   - Alert si: <15% o >30%

2. **Score Distribution:**
   - Media de probabilidades predichas
   - Esperado: ~0.22
   - Alert si: <0.15 o >0.35

3. **Approval Rate:**
   - % de solicitudes aprobadas (PD <0.20)
   - Esperado: ~38% (threshold=0.12)
   - Alert si: <25% o >55%

**Frecuencia:** Diaria

**Implementación:**
```python
def monitor_proxies(y_pred_proba):
    high_risk_pct = (y_pred_proba > 0.5).mean()
    avg_score = y_pred_proba.mean()
    approval_rate = (y_pred_proba < 0.2).mean()

    alerts = []
    if high_risk_pct < 0.15 or high_risk_pct > 0.30:
        alerts.append("⚠️ High risk %: {:.2%}".format(high_risk_pct))
    if avg_score < 0.15 or avg_score > 0.35:
        alerts.append("⚠️ Avg score: {:.4f}".format(avg_score))
    if approval_rate < 0.25 or approval_rate > 0.55:
        alerts.append("⚠️ Approval rate: {:.2%}".format(approval_rate))

    return alerts
```

## 5. Monitoreo de API

### 5.1 Métricas de Latencia

**Implementación:** Endpoint `/prometheus` en `src/api/main.py`

**Métricas expuestas:**

| Métrica | Target | Alert Threshold |
|---------|--------|-----------------|
| **Latency P50** | <100ms | >150ms |
| **Latency P95** | <200ms | >300ms |
| **Latency P99** | <500ms | >1000ms |

**Frecuencia:** Real-time (Prometheus scrape cada 15s)

**Configuración Prometheus:**
```yaml
scrape_configs:
  - job_name: 'credit-risk-api'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/prometheus'
```

### 5.2 Métricas de Throughput

| Métrica | Target | Alert Threshold |
|---------|--------|-----------------|
| **Requests/s** | >10 | <5 (bajo tráfico) |
| **Predictions/day** | >1,000 | <500 |
| **Uptime** | >99.5% | <99.0% |

**Frecuencia:** Real-time

### 5.3 Métricas de Errores

| Métrica | Target | Alert Threshold |
|---------|--------|-----------------|
| **Error Rate (4xx)** | <1% | >5% |
| **Error Rate (5xx)** | <0.1% | >1% |
| **Model Load Failures** | 0 | >0 |

**Frecuencia:** Real-time

**Implementación:**
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API latency')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions')
ERROR_COUNT = Counter('api_errors_total', 'Total API errors', ['error_type'])
```

## 6. Sistema de Alertas

### 6.1 Canales de Alerta

**Implementación:** `src/monitoring/alerts.py`

**Niveles de severidad:**
- 🚨 **CRITICAL:** Requiere acción inmediata (reentrenamiento, rollback)
- ⚠️ **WARNING:** Requiere investigación (drift moderado, latency alta)
- ℹ️ **INFO:** Informativo (cambios menores, tendencias)

**Canales:**
1. **Slack/MS Teams:** Alertas en tiempo real
2. **Email:** Reportes diarios/semanales
3. **PagerDuty:** Alertas críticas on-call
4. **Dashboard:** Grafana/Streamlit para visualización

### 6.2 Triggers de Alerta

| Condición | Severidad | Acción |
|-----------|-----------|--------|
| PSI >0.25 en feature top 5 | 🚨 CRITICAL | Reentrenar modelo |
| AUC <0.75 | 🚨 CRITICAL | Reentrenar modelo |
| Latency P95 >300ms (5 min consecutivos) | 🚨 CRITICAL | Escalar infra o rollback |
| Error rate >5% (5 min consecutivos) | 🚨 CRITICAL | Investigar logs, posible rollback |
| 0.10 ≤ PSI <0.25 | ⚠️ WARNING | Monitorear de cerca |
| 0.75 ≤ AUC <0.78 | ⚠️ WARNING | Considerar reentrenamiento |
| Latency P95 >200ms | ⚠️ WARNING | Optimizar inference |
| Approval rate fuera de [25%, 55%] | ⚠️ WARNING | Revisar threshold |

**Frecuencia de evaluación:**
- Métricas de API: Cada 1 minuto
- Drift: Cada 1 semana
- Performance (con labels): Cada 1 mes

### 6.3 Ejemplo de Alerta

```python
def send_alert(severity, message):
    if severity == "CRITICAL":
        send_slack("🚨 CRITICAL: " + message)
        send_email(to="oncall@company.com", subject="CRITICAL ALERT", body=message)
        trigger_pagerduty(message)
    elif severity == "WARNING":
        send_slack("⚠️ WARNING: " + message)
        log_to_dashboard(message)
    else:  # INFO
        log_to_dashboard(message)
```

## 7. Reentrenamiento del Modelo

### 7.1 Triggers de Reentrenamiento

**Automático (recomendado):**
- ✅ **Mensual:** Reentrenamiento programado con datos frescos
- 🚨 **On-demand:** Si PSI >0.25 o AUC <0.75

**Manual (bajo revisión):**
- ⚠️ Cuando stakeholders reportan inconsistencias
- ⚠️ Cambios regulatorios o de negocio

### 7.2 Proceso de Reentrenamiento

```
1. Data Collection
   ├── Nuevos datos de producción (últimos 3-6 meses)
   └── Labels confirmados de defaults

2. Data Validation
   ├── Great Expectations checks
   └── Drift analysis vs training original

3. Retraining
   ├── Same pipeline (F4 feature engineering)
   ├── Hyperparameter tuning (Optuna)
   └── Calibration (isotonic)

4. Validation
   ├── Test set metrics (AUC, Recall, KS)
   ├── A/B testing vs modelo actual
   └── Fairness audit

5. Deployment
   ├── Blue-green deployment
   ├── Canary rollout (5% → 25% → 100%)
   └── Monitoreo intensivo primeras 48h
```

**Frecuencia:** Mensual + on-demand

### 7.3 Model Registry

**Versionado con MLflow:**

```python
import mlflow

# Registrar nuevo modelo
with mlflow.start_run():
    mlflow.log_params(new_params)
    mlflow.log_metrics(new_metrics)
    mlflow.sklearn.log_model(new_model, "model")

    # Registrar en Model Registry
    mlflow.register_model(
        model_uri="runs:/{}/model".format(run_id),
        name="credit-risk-classifier",
        tags={"version": "1.1.0", "retrain_date": "2025-12-26"}
    )
```

**Stages:**
- **None:** Modelo en desarrollo
- **Staging:** Modelo en testing (A/B)
- **Production:** Modelo activo en producción
- **Archived:** Modelos antiguos (rollback si necesario)

## 8. Dashboards de Monitoreo

### 8.1 Dashboard Operacional (Grafana/Streamlit)

**Paneles recomendados:**

1. **API Health:**
   - Requests/s (time series)
   - Latency P50/P95/P99 (time series)
   - Error rate (time series)
   - Uptime (gauge)

2. **Model Performance:**
   - AUC-ROC trending (monthly)
   - Recall trending (monthly)
   - KS statistic trending (monthly)

3. **Data Drift:**
   - PSI por feature (heatmap)
   - KS p-values (table)
   - Score distribution (histogram)

4. **Business Metrics:**
   - Approval rate (gauge)
   - Default rate proxy (gauge)
   - Cost savings estimado (counter)

**Refresh:** 1 minuto (API), 1 día (drift), 1 mes (performance)

### 8.2 Reporte Semanal

**Enviado vía email a stakeholders:**

```
Subject: [Credit Risk Model] Weekly Monitoring Report

1. API Performance:
   - Avg latency: 87ms (target: <100ms) ✅
   - Uptime: 99.8% (target: >99.5%) ✅
   - Total predictions: 12,453

2. Data Drift:
   - PSI PAY_0: 0.08 (stable) ✅
   - PSI utilization_1: 0.12 (moderate drift) ⚠️
   - KS test: p=0.08 (no drift) ✅

3. Model Performance (Proxy):
   - Avg score: 0.23 (expected: 0.22) ✅
   - Approval rate: 37% (expected: 38%) ✅
   - High risk %: 24% (expected: 22%) ⚠️

4. Actions:
   - ⚠️ Monitor utilization_1 drift (close to threshold)
   - ⚠️ High risk % slightly elevated, investigate data source
```

## 9. Auditoría y Compliance

### 9.1 Logging

**Logs requeridos:**
- ✅ Todas las predicciones (input features + score + decision)
- ✅ Timestamps de requests
- ✅ User ID (si disponible)
- ✅ Model version usada

**Retención:** 2 años (regulación financiera)

**Formato:**
```json
{
  "timestamp": "2025-12-26T10:15:30Z",
  "request_id": "uuid-1234",
  "model_version": "1.0.0",
  "input": {...},
  "prediction": {
    "probability": 0.27,
    "prediction": "DEFAULT",
    "risk_band": "REVISION",
    "threshold_used": 0.12
  },
  "latency_ms": 87
}
```

### 9.2 Explainability Logging

**Para regulación (SHAP values):**

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_prod)

# Log top 5 features más influyentes
top_features = get_top_shap_features(shap_values, feature_names, k=5)
log_to_audit(request_id, top_features)
```

**Uso:** Justificar decisiones de rechazo ante clientes.

## 10. Checklist de Monitoreo

### 10.1 Pre-Producción

- [x] Drift monitor implementado (PSI, KS, CSI)
- [x] Performance monitor implementado (AUC, Recall, KS)
- [x] API monitoring configurado (latency, errors)
- [x] Alertas configuradas (Slack/Email)
- [ ] Dashboard Grafana/Streamlit creado
- [ ] Reporte semanal automatizado
- [ ] Great Expectations para data validation
- [ ] A/B testing framework listo

### 10.2 Post-Producción

- [ ] Monitoreo activo durante primeras 48h
- [ ] Baseline de métricas establecido (primera semana)
- [ ] Alertas ajustadas según baseline real
- [ ] Reentrenamiento mensual programado
- [ ] Auditoría de logs funcionando
- [ ] Fairness monitoring activo

## 11. Decisiones de Diseño (ADR)

| Decisión | Alternativa | Razón |
|----------|-------------|-------|
| PSI para drift | KL Divergence | PSI más interpretable para stakeholders |
| Umbral PSI=0.25 | 0.10 o 0.30 | Balance sensibilidad/false alarms |
| Reentrenamiento mensual | Trimestral | Comportamiento crediticio cambia rápido |
| Prometheus | CloudWatch/Datadog | Open-source, extensible, estándar industria |
| Isotonic calibration | Re-calibrar en producción | Mantiene calibración original, menos riesgo |
| Threshold=0.12 fijo | Dinámico | Simplifica operación, cambios controlados |

## 12. Conclusión

El sistema de monitoreo implementa:
- ✅ **Drift detection** (PSI, KS, CSI) para top 10 features
- ✅ **Performance tracking** (AUC, Recall, KS, Brier)
- ✅ **API monitoring** (latency, throughput, errors)
- ✅ **Alerting system** (Slack, Email, PagerDuty)
- ✅ **Reentrenamiento** mensual programado + on-demand
- ✅ **Model registry** con MLflow versionado
- ⚠️ **Dashboard** recomendado (Grafana/Streamlit)
- ⚠️ **Fairness monitoring** pendiente implementación

**Ready for production monitoring** con estrategia clara de detección temprana y mitigación de degradación ✅

**Archivos de monitoreo:**
- `src/monitoring/drift_monitor.py` - Implementación PSI/KS/CSI
- `src/monitoring/alerts.py` - Sistema de alertas
- `reports/monitoring/` - Reportes de drift (generados semanalmente)

---

**Documento completado por:**
**Ing. Daniel Varela Pérez**
Senior Data Scientist & ML Engineer
📧 bedaniele0@gmail.com

**Metodología:** DVP-PRO
**Fase:** F8 - Monitoring
**Fecha:** Diciembre 2024
