# F0 – Problem Statement: Credit Card Default Risk Scoring (UCI Taiwan)

## 1. Contexto de Negocio
Una institución emisora de tarjetas de crédito busca mejorar la gestión del riesgo de impago a corto plazo. Actualmente, las aprobaciones se basan en reglas estáticas y la experiencia de analistas, lo que limita la precisión, la escalabilidad y la trazabilidad del proceso.

## 2. Problema
Determinar la probabilidad de que un cliente incumpla el pago mínimo en el próximo mes, utilizando su historial de comportamiento y variables socioeconómicas. El proceso actual no logra anticipar correctamente a los clientes con alto riesgo de impago.

## 3. Objetivo
Desarrollar un modelo de **scoring predictivo** que clasifique a los clientes en tres bandas de riesgo, para mejorar la decisión de otorgamiento de crédito:
- **Aprobado:** PD < 20%
- **Revisión:** 20% ≤ PD < 50%
- **Rechazo:** PD ≥ 50%

## 4. Datos Base
- Dataset: *Default of Credit Card Clients (Taiwan, 2005)*  
- Registros: 30,000  
- Variables: 25 (demográficas, financieras y de comportamiento de pago)  
- Variable objetivo: `default.payment.next.month` (1 = incumple, 0 = cumple)
- Origen: Universidad Nacional de Taiwán / UCI Machine Learning Repository

## 5. Supuestos y Restricciones
- Datos tabulares con variables numéricas y categóricas codificadas.
- No contiene PII sensible.
- *Train/test split* 80/20 y validación cruzada 5-fold.
- Se requiere explicabilidad (SHAP) y evaluación de sesgo por género y educación.

## 6. Métricas de Éxito
| Métrica | Meta Mínima | Justificación |
|-----------|--------------|----------------|
| **AUC-ROC** | ≥ 0.80 | Medida general de discriminación |
| **KS** | ≥ 0.30 | Separación entre buenos y malos pagadores |
| **Recall (Clase 1)** | ≥ 0.70 | Sensibilidad ante incumplidos |
| **Precision (Clase 1)** | ≥ 0.30 | Reducción de falsos positivos |
| **Brier Score** | ≤ 0.20 | Calibración de probabilidades |

## 7. Indicadores de Negocio (KPI)
- Reducción del índice de morosidad en ≥ 12%.
- Incremento del margen ajustado al riesgo en ≥ 5%.
- Tiempo de respuesta del scoring < 200 ms (API).

## 8. Stakeholders
- **Sponsor:** Chief Risk Officer (CRO)
- **Usuarios:** Analistas de Riesgo y Cobranza
- **Soporte Técnico:** Data Science, Data Engineering, MLOps

## 9. Riesgos Identificados
- **Desbalance del target:** aplicar *class weighting* o *SMOTE*.
- **Drift temporal:** monitoreo PSI y recalibración mensual.
- **Aspectos éticos/regulatorios:** exclusión de variables sensibles y reporte XAI documentado.

---
**Resultado Esperado:**
Un modelo de riesgo crediticio calibrado, explicable y con trazabilidad completa bajo la metodología **DVP-PRO**, integrable en el flujo de originación de tarjetas de crédito y monitoreado en producción.

