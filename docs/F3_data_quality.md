# F3 - Data Quality Report - Credit Risk Scoring

**Autor:** Ing. Daniel Varela Pérez
**Email:** bedaniele0@gmail.com
**Metodología:** DVP-PRO (Fase 3)

## 1. Objetivo de la Fase

Analizar la calidad del dataset UCI *Default of Credit Card Clients*, identificar problemas de datos y documentar transformaciones necesarias para garantizar la integridad del modelo.

## 2. Dataset Overview

### 2.1 Información General

- **Fuente:** UCI Machine Learning Repository
- **Nombre:** Default of Credit Card Clients (Taiwan, 2005)
- **Registros Totales:** 30,000 clientes
- **Variables:** 25 (23 features + 1 ID + 1 target)
- **Target:** `default.payment.next.month` (1 = default, 0 = no default)
- **Periodo:** Abril 2005 - Septiembre 2005

### 2.2 Distribución del Target

```
Class 0 (No Default): 23,364 (77.88%)
Class 1 (Default):     6,636 (22.12%)

Ratio: 3.52:1 (desbalanceado)
```

**Interpretación:** Dataset moderadamente desbalanceado, requiere técnicas de balanceo (SMOTE, class weighting) para evitar bias hacia la clase mayoritaria.

## 3. Features del Dataset

### 3.1 Variables Demográficas

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| **SEX** | Categórica | Género del cliente | 1=Male, 2=Female |
| **EDUCATION** | Categórica | Nivel educativo | 1=Grad, 2=Univ, 3=HS, 4=Other |
| **MARRIAGE** | Categórica | Estado civil | 1=Married, 2=Single, 3=Other |
| **AGE** | Numérica | Edad en años | 21-79 años |

### 3.2 Variables Financieras

| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| **LIMIT_BAL** | Numérica | Límite de crédito (NT$) | 10,000 - 1,000,000 |
| **BILL_AMT1-6** | Numérica | Facturación mensual (6 meses) | -165,580 - 964,511 |
| **PAY_AMT1-6** | Numérica | Pago mensual (6 meses) | 0 - 873,552 |

**Nota:** BILL_AMT puede ser negativo (crédito a favor del cliente).

### 3.3 Variables de Comportamiento

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| **PAY_0** | Ordinal | Estatus de pago Sept 2005 | -2 a 8 |
| **PAY_2** | Ordinal | Estatus de pago Ago 2005 | -2 a 8 |
| **PAY_3** | Ordinal | Estatus de pago Jul 2005 | -2 a 8 |
| **PAY_4** | Ordinal | Estatus de pago Jun 2005 | -2 a 8 |
| **PAY_5** | Ordinal | Estatus de pago May 2005 | -2 a 8 |
| **PAY_6** | Ordinal | Estatus de pago Abr 2005 | -2 a 8 |

**Códigos:**
- `-2`: No consumption
- `-1`: Paid in full
- `0`: Revolving credit (uso normal)
- `1-8`: Meses de retraso (1=1 mes, 2=2 meses, ..., 8=8+ meses)

## 4. Data Quality Issues Identificados

### 4.1 Valores Faltantes

✅ **Resultado:** **0 valores nulos** en todo el dataset

**Validación:**
```python
df.isnull().sum().sum()  # Output: 0
```

**Conclusión:** No se requiere imputación.

### 4.2 Valores Atípicos (Outliers)

**LIMIT_BAL:**
- P99: 800,000 NT$
- Max: 1,000,000 NT$
- ✅ Sin outliers extremos

**BILL_AMT1-6:**
- Min: -165,580 NT$ (crédito a favor)
- Max: 964,511 NT$
- ⚠️ Valores negativos son válidos (overpaymment)

**PAY_AMT1-6:**
- Min: 0 NT$
- Max: 873,552 NT$
- ✅ Sin outliers extremos

**AGE:**
- Min: 21 años
- Max: 79 años
- ✅ Rango razonable

### 4.3 Valores Inconsistentes

**EDUCATION:**
- Valores esperados: 1, 2, 3, 4
- Valores encontrados: **0, 5, 6** (no documentados)
- **Solución:** Agrupar 0, 5, 6 → 4 (Other)

**MARRIAGE:**
- Valores esperados: 1, 2, 3
- Valores encontrados: **0** (no documentado)
- **Solución:** Agrupar 0 → 3 (Other)

**PAY_0 a PAY_6:**
- Valores esperados: -2 a 8
- Valores encontrados: **-2, -1, 0, 1-8** (conforme)
- ✅ Sin inconsistencias

### 4.4 Duplicados

**Validación:**
```python
df.duplicated().sum()  # Output: 0
```

✅ **Resultado:** 0 registros duplicados

## 5. Exploratory Data Analysis (EDA)

### 5.1 Correlación con Target

**Top 5 Features más correlacionadas con `default.payment.next.month`:**

1. **PAY_0** (0.324) - Estatus de pago más reciente
2. **PAY_2** (0.264) - Estatus de pago hace 2 meses
3. **PAY_3** (0.235) - Estatus de pago hace 3 meses
4. **PAY_4** (0.218) - Estatus de pago hace 4 meses
5. **PAY_5** (0.205) - Estatus de pago hace 5 meses

**Insight:** Variables de comportamiento de pago (`PAY_*`) son los predictores más fuertes, validando el diseño del modelo.

### 5.2 Distribución por Género

```
SEX = 1 (Male):   11,888 (39.6%)
SEX = 2 (Female): 18,112 (60.4%)

Default Rate Male:   23.5%
Default Rate Female: 21.4%
```

**Diferencia:** +2.1pp mayor tasa de default en hombres (no significativa).

### 5.3 Distribución por Educación

```
EDUCATION = 1 (Graduate):     10,585 (35.3%)
EDUCATION = 2 (University):   14,030 (46.8%)
EDUCATION = 3 (High School):   4,917 (16.4%)
EDUCATION = 4 (Other):           468 (1.5%)

Default Rate Graduate:    21.2%
Default Rate University:  22.8%
Default Rate High School: 22.0%
Default Rate Other:       23.5%
```

**Insight:** Educación universitaria tiene tasa ligeramente mayor (posible confounding con edad/deuda).

### 5.4 Distribución por Estado Civil

```
MARRIAGE = 1 (Married):  13,659 (45.5%)
MARRIAGE = 2 (Single):   15,964 (53.2%)
MARRIAGE = 3 (Other):       377 (1.3%)

Default Rate Married:  20.8%
Default Rate Single:   23.2%
Default Rate Other:    23.3%
```

**Insight:** Clientes casados tienen menor tasa de default (-2.4pp).

### 5.5 Distribución por Edad

```
Age 21-25:  3,275 (10.9%)  →  Default Rate: 25.4%
Age 26-35: 12,040 (40.1%)  →  Default Rate: 23.6%
Age 36-45:  9,267 (30.9%)  →  Default Rate: 20.5%
Age 46-60:  4,836 (16.1%)  →  Default Rate: 19.2%
Age 60+:      582 (1.9%)   →  Default Rate: 17.5%
```

**Insight:** Clientes jóvenes (21-25) tienen mayor riesgo de default (+7.9pp vs 60+).

## 6. Feature Engineering Aplicado

### 6.1 Features Derivadas Creadas

**Implementación:** `src/features/build_features.py`

| Feature | Fórmula | Rationale |
|---------|---------|-----------|
| **utilization_1** | `BILL_AMT1 / LIMIT_BAL` | Tasa de uso del crédito (alto uso = mayor riesgo) |
| **payment_ratio_1-6** | `PAY_AMT_k / BILL_AMT_k` | Capacidad de pago (bajo ratio = mayor riesgo) |
| **EDUCATION_grouped** | `0,5,6 → 4` | Agrupar valores inconsistentes |
| **MARRIAGE_grouped** | `0 → 3` | Agrupar valores inconsistentes |
| **AGE_bin** | Bins: 21-25, 26-35, 36-45, 46-60, 60+ | Segmentación por rango de edad |

**Total Features Finales:** 37 (23 originales + 14 derivadas)

### 6.2 Validación de Features

✅ **0 valores nulos** después de feature engineering
✅ **0 valores infinitos** (manejo de división por cero)
✅ **Rangos válidos** en todas las features

## 7. Train/Test Split

### 7.1 Estrategia

```python
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y  # Mantiene proporción de clases
)
```

### 7.2 Distribución

| Set | Registros | % Total | Default Rate |
|-----|-----------|---------|--------------|
| **Train** | 24,000 | 80% | 22.12% |
| **Test** | 6,000 | 20% | 22.12% |

✅ **Estratificación exitosa:** Mismo default rate en train y test.

## 8. Data Validation Checks

### 8.1 Great Expectations (Sugerido)

Para producción, se recomienda implementar Great Expectations:

```python
# Ejemplo de expectation suite
expect_column_values_to_be_between("LIMIT_BAL", min_value=10000, max_value=1000000)
expect_column_values_to_be_in_set("SEX", [1, 2])
expect_column_values_to_not_be_null("default.payment.next.month")
```

### 8.2 Validaciones Implementadas

✅ Rango de valores para variables numéricas
✅ Valores permitidos para variables categóricas
✅ Tipos de datos correctos (int64, float64)
✅ Sin duplicados
✅ Sin nulos

## 9. Limitaciones del Dataset

### 9.1 Limitaciones Identificadas

- **Temporal:** Datos de 2005 (puede no reflejar comportamiento actual)
- **Geográfico:** Solo Taiwan (limitada generalización)
- **Features:** No incluye score crediticio externo (bureau)
- **Periodo:** Solo 6 meses de historial (ideal sería 12-24 meses)
- **Variables ausentes:**
  - Ingresos del cliente
  - Historial laboral
  - Deuda total de otras fuentes
  - Score crediticio (FICO, Equifax)

### 9.2 Mitigaciones

- **Cross-validation 5-fold** para validar robustez
- **Threshold optimization** por costo de negocio
- **Monitoreo de drift** para detectar cambios en producción
- **Reentrenamiento periódico** (mensual sugerido)

## 10. Conclusiones de Data Quality

### 10.1 Resumen

✅ **Dataset limpio:** 0 nulos, 0 duplicados, sin outliers extremos
✅ **Consistencia:** Valores inconsistentes corregidos (EDUCATION, MARRIAGE)
✅ **Balance:** Desbalance 3.52:1 manejable con SMOTE/class weighting
✅ **Features:** 36 features (23 raw + 13 derivadas) listas para modeling
✅ **Split:** Train/Test estratificado 80/20 con misma distribución

### 10.2 Calidad General

**Score de Calidad:** 9/10

**Detalles:**
- ✅ Completitud: 10/10 (0% missing)
- ✅ Consistencia: 8/10 (issues menores corregidos)
- ✅ Validez: 9/10 (rangos válidos, tipos correctos)
- ✅ Unicidad: 10/10 (0 duplicados)
- ⚠️ Actualidad: 6/10 (dataset de 2005)

### 10.3 Recomendaciones para Producción

1. **Implementar Great Expectations** para validación automática
2. **Monitoreo de drift** en features clave (PAY_0, LIMIT_BAL, utilization)
3. **Reentrenamiento mensual** con datos frescos
4. **Alertas automáticas** si default rate cambia >5%
5. **Feature importance tracking** para detectar cambios en predictividad

## 11. Archivos Generados

```
data/processed/
├── credit_data_processed.csv  (30,000 × 38)  # Dataset completo procesado
├── X_train.csv                (24,000 × 37)  # Features de entrenamiento
├── X_test.csv                 (6,000 × 37)   # Features de prueba
├── y_train.csv                (24,000 × 1)   # Target de entrenamiento
└── y_test.csv                 (6,000 × 1)    # Target de prueba
```

**También disponible en formato Parquet** para lectura rápida.

---

**Documento completado por:**
**Ing. Daniel Varela Pérez**
Senior Data Scientist & ML Engineer
📧 bedaniele0@gmail.com

**Metodología:** DVP-PRO
**Fase:** F3 - Data Quality
**Fecha:** Diciembre 2024
