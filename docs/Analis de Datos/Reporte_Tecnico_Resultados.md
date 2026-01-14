# Reporte Tecnico de Resultados - Credit Risk Scoring

## 1. Resumen tecnico
Este reporte interpreta los resultados del modelo de riesgo crediticio usando el dataset UCI Taiwan. El modelo final es un LightGBM calibrado (isotonic) con umbral optimizado en 0.12, enfocado en maximizar deteccion de incumplimientos bajo una matriz de costos de negocio (FP=$1,000, FN=$10,000 MXN).

## 2. Modelo y datos
- Modelo: CalibratedClassifierCV (LightGBM + calibracion isotonica)
- Features: 36 (23 raw + 13 derivadas)
- Dataset: 30,000 registros (UCI Taiwan, 2005)
- Split: train 24,000 / test 6,000
- Target: default.payment.next.month (alias default_flag post-ingesta)

## 3. Metricas principales (test set)
Fuente: `reports/metrics/validation_results.json`

### 3.1 Umbral optimizado (0.12)
- AUC-ROC: 0.7813
- KS: 0.4251
- Recall: 0.8704
- Precision: 0.3107
- F1: 0.4579
- Brier: 0.1349
- Matriz de confusion (test):
  - TN: 2,110
  - FP: 2,563
  - FN: 172
  - TP: 1,155

### 3.2 Umbral default (0.50)
- Recall: 0.3715
- Precision: 0.6591
- F1: 0.4752
- Accuracy: 0.8185

**Interpretacion tecnica:** el umbral 0.12 aumenta significativamente el recall (deteccion de incumplimientos) a costa de precision. Esto es consistente con el objetivo de minimizar costo total de negocio bajo FN mas costoso que FP.

## 4. Estabilidad del modelo
- CV (5-fold): AUC 0.7816 ± 0.0063 (CV 0.81%)
- Bootstrap 95% CI AUC: [0.7662, 0.7961]

**Interpretacion:** el modelo es estable; la variacion en AUC y recall es baja en validaciones repetidas.

## 5. Calibracion
- Brier score 0.1349
- Calibracion isotonica aplicada

**Interpretacion:** las probabilidades son confiables para decisiones basadas en umbral y bandas de riesgo.

## 6. Ahorro estimado
- Cost Savings reportado: $5,466,000 MXN
- Matriz de costos: FP=$1,000, FN=$10,000

**Interpretacion:** el ahorro depende de la matriz de costos; con los costos definidos, el umbral optimizado reduce el costo esperado frente a baseline.

## 7. Bandas de riesgo
- APROBADO: PD < 0.20
- REVISION: 0.20 <= PD < 0.50
- RECHAZO: PD >= 0.50

**Interpretacion:** la segmentacion permite separar flujos operativos (aprobacion automatica, revision manual, rechazo).

## 8. Monitoreo
Se cuenta con monitoreo de drift (PSI/CSI) y metricas de performance. El ultimo reporte de drift indica estado HEALTHY con CSI muy bajo en features clave.

## 9. Limitaciones
- Dataset historico (2005), requiere validacion con datos actuales para produccion real.
- AUC 0.7813 por debajo del target 0.80, aunque KS y recall cumplen.
- Threshold 0.12 incrementa falsos positivos; requiere capacidad operativa para revision.

## 10. Recomendaciones tecnicas
1. Mantener monitoreo PSI/CSI mensual y alertas de degradacion.
2. Ejecutar fairness audit por SEX y AGE antes de escenarios regulados.
3. Evaluar features externas (burden/ingresos/bureau) para mejorar AUC.
4. Validar con datos recientes y reentrenar si hay drift significativo.

---

Fuente de metricas: `reports/metrics/validation_results.json`
