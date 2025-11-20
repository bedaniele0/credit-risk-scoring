# [Credit Card Default Risk Scoring - Feature Engineering]
## Fase 4: Ingeniería de Características

**Autor:** Ing. Daniel Varela Perez
**Email:** bedaniele0@gmail.com
**Fecha:** 2025-11-07
**Versión:** 1.0

### Objetivos:
1. Cargar el dataset limpio.
2. Crear variables de utilización de crédito.
3. Crear variables de ratios de pago.
4. Agrupar categorías minoritarias en `EDUCATION` y `MARRIAGE`.
5. Guardar el dataset procesado para la fase de modelado.

## 1. Configuración e Importación de Librerías


```python
import pandas as pd
import numpy as np
import yaml
import os
```

## 2. Carga de Datos


```python
def load_config(config_path='../config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
raw_data_path = os.path.join('..', config['paths']['raw_data'])

df = pd.read_csv(raw_data_path)
df_featured = df.copy()

df_featured.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



## 3. Creación de Nuevas Características (Features)

### 3.1 Ratios de Utilización de Crédito

Mide qué porcentaje del límite de crédito está siendo utilizado por el cliente en cada mes. Un alto ratio de utilización puede ser un indicador de estrés financiero.


```python
for i in range(1, 7):
    bill_col = f'BILL_AMT{i}'
    util_col = f'UTIL_RATIO_{i}'
    # Evitar división por cero si LIMIT_BAL es 0
    df_featured[util_col] = df_featured.apply(
        lambda row: row[bill_col] / row['LIMIT_BAL'] if row['LIMIT_BAL'] > 0 else 0, 
        axis=1
    )
```

### 3.2 Ratios de Pago

Mide qué porcentaje de la factura fue pagado. Un ratio bajo o de cero indica dificultades de pago.


```python
for i in range(1, 7):
    pay_col = f'PAY_AMT{i}'
    bill_col = f'BILL_AMT{i}'
    ratio_col = f'PAY_RATIO_{i}'
    # Evitar división por cero si la factura es 0
    df_featured[ratio_col] = df_featured.apply(
        lambda row: row[pay_col] / row[bill_col] if row[bill_col] > 0 else 1, # Si la factura es 0, se asume pago completo (ratio 1)
        axis=1
    )
```

### 3.3 Agrupación de Categorías Minoritarias

Las variables `EDUCATION` y `MARRIAGE` tienen categorías con muy pocas muestras (e.g., 0, 5, 6 en EDUCATION). Agruparlas en una categoría 'Otros' puede ayudar al modelo a generalizar mejor.


```python
# Agrupar categorías en EDUCATION
df_featured['EDUCATION_CAT'] = df_featured['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
# 1=posgrado, 2=universidad, 3=bachillerato, 4=otros

# Agrupar categorías en MARRIAGE
df_featured['MARRIAGE_CAT'] = df_featured['MARRIAGE'].replace({0: 3})
# 1=casado, 2=soltero, 3=otros

print("Nuevos valores para EDUCATION_CAT:")
print(df_featured['EDUCATION_CAT'].value_counts())
print("
Nuevos valores para MARRIAGE_CAT:")
print(df_featured['MARRIAGE_CAT'].value_counts())
```


      Cell In[14], line 11
        print("
              ^
    SyntaxError: unterminated string literal (detected at line 11)



## 4. Guardar Dataset Procesado


```python
processed_data_dir = os.path.join('..', config['paths']['processed_data'])
os.makedirs(processed_data_dir, exist_ok=True)
processed_data_path = os.path.join(processed_data_dir, 'featured_dataset.csv')

df_featured.to_csv(processed_data_path, index=False)

print(f"Dataset con nuevas características guardado en: {processed_data_path}")
print(f"Nuevas dimensiones del dataset: {df_featured.shape}")
```

    Dataset con nuevas características guardado en: ../data/processed/featured_dataset.csv
    Nuevas dimensiones del dataset: (30000, 37)


## 5. Siguientes Pasos

Con el dataset enriquecido, estamos listos para la **Fase 5: Modelado y Experimentación**. El siguiente notebook (`03_modeling.ipynb`) se centrará en:
1. Preparar los datos para el entrenamiento (train/test split).
2. Entrenar un modelo baseline (Regresión Logística).
3. Entrenar nuestro modelo principal (LightGBM) y optimizar hiperparámetros.
4. Registrar todos los experimentos y resultados en MLflow.
