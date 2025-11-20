
# Model Card - Credit Risk Scoring Model

**Proyecto:** Credit Card Default Risk Scoring (UCI Taiwan)  
**Metodología:** DVP-PRO (Daniel Varela Pérez - Professional)  
**Fase:** F5 - Modelado y Experimentación  

---

## 1. Model Details

- **Developer:** Ing. Daniel Varela Pérez
- **Email:** bedaniele0@gmail.com
- **Tel:** +52 55 4189 3428
- **Model Date:** 2025-11-18
- **Model Version:** 1.0.0
- **Model Type:** Binary Classification (Credit Risk)
- **Algorithm:** LightGBM + Calibration (isotonic)
- **Framework:** scikit-learn, LightGBM

---

## 2. Intended Use

### Primary Use Cases
- Predicción de probabilidad de incumplimiento (default) en tarjetas de crédito
- Clasificación de riesgo crediticio en 3 bandas:
  - **Aprobado:** PD < 20%
  - **Revisión:** 20% ≤ PD < 50%
  - **Rechazo:** PD ≥ 50%
- Apoyo a decisiones de originación crediticia

### Primary Users
- Analistas de Riesgo Crediticio
- Equipos de Originación y Cobranza
- Gestores de Cartera

### Out-of-Scope Use Cases
- Decisiones automatizadas sin supervisión humana
- Aplicación a productos crediticios diferentes (hipotecas, préstamos personales)
- Uso sin monitoreo continuo de drift

---

## 3. Training Data

- **Dataset:** Default of Credit Card Clients (Taiwan, 2005)
- **Source:** UCI Machine Learning Repository
- **Total Samples:** 30,000
- **Training Set:** 24,000 (80%)
- **Test Set:** 6,000 (20%)
- **Features:** 36 (post feature engineering + encoding)
- **Target Distribution (Train):** Class 0: 77.88%, Class 1: 22.12%
- **Date Range:** 2005 (snapshot histórico)

### Key Features
- Variables demográficas: AGE, SEX, EDUCATION, MARRIAGE
- Variables financieras: LIMIT_BAL, BILL_AMT1-6, PAY_AMT1-6
- Historial de pago: PAY_0, PAY_2-6
- Features derivadas: utilization_1, payment_ratio_1-6, age_bins, grouped categories

---

## 4. Evaluation Data

- **Test Set Size:** 6,000 samples
- **Validation Strategy:** Stratified 5-fold Cross-Validation
- **Target Distribution (Test):** Class 0: 77.88%, Class 1: 22.12%

---

## 5. Performance Metrics

### Test Set Results

| Métrica | Valor | Meta | Estado |
|---------|-------|------|--------|
| **AUC-ROC** | 0.7813 | ≥ 0.80 | ✗ FAIL |
| **KS Statistic** | 0.4251 | ≥ 0.30 | ✓ PASS |
| **Recall (Default)** | 0.3715 | ≥ 0.70 | ✗ FAIL |
| **Precision (Default)** | 0.6591 | ≥ 0.30 | ✓ PASS |
| **Brier Score** | 0.1349 | ≤ 0.20 | ✓ PASS |
| **Accuracy** | 0.8185 | - | - |
| **F1-Score (Default)** | 0.4752 | - | - |

### Cross-Validation Results
- **AUC (CV mean):** 0.7890

---

## 6. Model Architecture

### Base Model: LightGBM
- **Boosting Type:** Gradient Boosting Decision Tree (GBDT)
- **Optimization:** Optuna (50 trials)
- **Class Handling:** Balanced class weights

### Best Hyperparameters (Optuna)
```python
{
  "num_leaves": 46,
  "learning_rate": 0.012637124713745373,
  "n_estimators": 381,
  "max_depth": 9,
  "min_child_samples": 24,
  "feature_fraction": 0.6521833098780689,
  "bagging_fraction": 0.6018948524228394,
  "bagging_freq": 2,
  "reg_alpha": 0.0001250811528398735,
  "reg_lambda": 2.868622644974536
}
```

### Calibration
- **Method:** Isotonic
- **CV Folds:** 5
- **Purpose:** Mejorar calibración de probabilidades (Brier Score)

---

## 7. Ethical Considerations

### Fairness & Bias
- **Variables sensibles:** SEX, AGE incluidas en el modelo
- **Recomendación:** Realizar análisis de disparate impact por género y edad
- **Mitigación:** Implementar thresholds diferenciados si se detecta sesgo significativo

### Privacy
- Dataset público (UCI Repository)
- No contiene PII identificable
- Datos históricos (2005) - considerar actualización para uso productivo

### Transparency
- Feature importance disponible
- Modelo interpretable vía SHAP values (implementar en F7)
- Decisiones auditables vía MLflow tracking

---

## 8. Caveats and Recommendations

### Limitations
1. **Datos históricos (2005):** El modelo fue entrenado con datos de hace 20 años. Requiere validación con datos recientes antes de deployment.
2. **Contexto geográfico:** Datos de Taiwan. Validar aplicabilidad a otras regiones.
3. **Desbalance de clases:** Target con ~22% de defaults. Calibración es crítica.
4. **Threshold óptimo:** Requiere definición según costo de negocio (F6 - Validación).

### Recommendations
1. **Monitoreo continuo:** Implementar PSI y KS drift monitoring (mensual)
2. **Recalibración:** Programar retraining trimestral o ante drift significativo
3. **Validación de negocio:** Calcular ROI real en piloto antes de rollout completo
4. **XAI:** Implementar SHAP explanations para decisiones individuales
5. **Threshold tuning:** Optimizar según matriz de costos de negocio

---

## 9. Model Governance

### Versioning
- **Model Version:** 1.0.0
- **Tracking System:** MLflow
- **Experiment Name:** credit_risk_scoring_uci_taiwan

### Artifacts
- `final_model.joblib`: Modelo serializado (LightGBM + Calibration)
- `final_metrics.json`: Métricas de evaluación
- `feature_names.json`: Lista de features requeridas
- `model_metadata.json`: Metadata completo del modelo

### Maintenance
- **Owner:** Ing. Daniel Varela Pérez (bedaniele0@gmail.com)
- **Review Frequency:** Trimestral
- **Retraining Trigger:** PSI > 0.25 o KS decay > 10%

---

## 10. References

1. **Dataset:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
2. **UCI Repository:** https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
3. **Metodología DVP-PRO:** Varela Pérez, D. (2025). Metodología Profesional para Proyectos de Data Science.

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-18 15:23:26  
**Contact:** bedaniele0@gmail.com | +52 55 4189 3428
