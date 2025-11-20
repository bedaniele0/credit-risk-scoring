# 02 – Feature Engineering (Credit Card Default Risk)
**Objetivo:** crear dataset modelable con variables derivadas y dividir en train/test.  
**Entrada:** `data/processed/train_dataset.parquet` (o CSV/XLSX crudo si no existe).  
**Salida:** `data/processed/features.parquet`, `X_train.parquet`, `X_test.parquet`, `y_train.parquet`, `y_test.parquet`, `src/features/feature_catalog.csv`.



```python

import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

BASE = Path("..").resolve()
RAW = BASE / "data/raw"
PROC = BASE / "data/processed"
CATALOG = BASE / "src/features/feature_catalog.csv"

PROC.mkdir(parents=True, exist_ok=True)
CATALOG.parent.mkdir(parents=True, exist_ok=True)

print("BASE:", BASE)

```

    BASE: /Users/danielevarella/Desktop/gemini_data_science/credit-risk-scoring



```python

def load_data():
    p_parquet = PROC / "train_dataset.parquet"
    if p_parquet.exists():
        df = pd.read_parquet(p_parquet)
        return df

    # fallback: buscar archivos crudos y normalizar columnas
    candidates = [
        RAW/"default_of_credit_card_clients.csv",
        RAW/"default of credit card clients.csv",
        RAW/"default_of_credit_card_clients.xlsx",
        RAW/"default of credit card clients.xlsx",
    ]
    src = None
    for c in candidates:
        if c.exists():
            src = c; break
    if src is None:
        raise FileNotFoundError("No se encontró dataset en data/processed ni data/raw.")

    if src.suffix.lower() == ".csv":
        # algunos CSV exportados desde Numbers requieren header=1
        df0 = pd.read_csv(src, nrows=2)
        df = pd.read_csv(src) if "ID" in df0.columns else pd.read_csv(src, header=1)
    else:
        df = pd.read_excel(src, header=1)

    # estandarizar nombres clave
    df = df.rename(columns={"default.payment.next.month": "default_flag", "PAY_1": "PAY_0"})
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

df = load_data()
df.head()[:3]

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
      <th>default_payment_next_month</th>
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
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>




```python

# Cast básicos
int_like = ["SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
for c in int_like:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# Imputación simple
for c in df.columns:
    if c == "default_flag":
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].astype(float).fillna(df[c].median())
    else:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

print(df.isna().sum().sum(), "nulos tras imputación")        
df.shape

```

    0 nulos tras imputación





    (30000, 25)



### Derivación de variables sugeridas en F3


```python

feat_notes = []

# Utilización de crédito BILL_AMT1 / LIMIT_BAL
if set(["BILL_AMT1","LIMIT_BAL"]).issubset(df.columns):
    df["utilization_1"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["utilization_1"] = df["utilization_1"].fillna(0.0).clip(0, 5)
    feat_notes.append(("utilization_1","BILL_AMT1/LIMIT_BAL recortado a [0,5]"))

# Ratios de pago PAY_AMTk / BILL_AMTk
for k in range(1,7):
    b, p = f"BILL_AMT{k}", f"PAY_AMT{k}"
    if b in df.columns and p in df.columns:
        col = f"payment_ratio_{k}"
        df[col] = df[p] / df[b].replace(0, np.nan)
        df[col] = df[col].fillna(0.0).clip(0, 5)
        feat_notes.append((col, f"{p}/{b} recortado a [0,5]"))

# Buckets de edad
if "AGE" in df.columns:
    df["AGE_bin"] = pd.cut(df["AGE"], bins=[0,25,35,45,60,120], labels=["<=25","26-35","36-45","46-60","60+"], right=True)
    feat_notes.append(("AGE_bin","Edad binned"))

# Agrupar categorías minoritarias en EDUCATION y MARRIAGE
if "EDUCATION" in df.columns:
    df["EDUCATION_grouped"] = df["EDUCATION"].replace({0:4,5:4,6:4})  # 4=otros
    feat_notes.append(("EDUCATION_grouped","Agrupa 0/5/6 en 'otros'"))

if "MARRIAGE" in df.columns:
    df["MARRIAGE_grouped"] = df["MARRIAGE"].replace({0:3})  # 3=otros
    feat_notes.append(("MARRIAGE_grouped","Agrupa 0 en 'otros'"))

len(feat_notes), "features derivadas"

```




    (10, 'features derivadas')




```python

target_candidates = ["default_flag","default_payment_next_month","default_payment_next_month"]
target = None
for t in ["default_flag","default_payment_next_month","default_payment_next_month"]:
    if t in df.columns:
        target = t; break
if target is None:
    raise ValueError("No se encontró la columna de target.")

# Variables base
base_cols = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

bill_cols = [f"BILL_AMT{i}" for i in range(1,7) if f"BILL_AMT{i}" in df.columns]
pay_cols = [f"PAY_AMT{i}" for i in range(1,7) if f"PAY_AMT{i}" in df.columns]
derived = [c for c in df.columns if c.startswith("utilization_") or c.startswith("payment_ratio_") or c.endswith("_grouped") or c=="AGE_bin"]

use_cols = [c for c in base_cols + bill_cols + pay_cols + derived if c in df.columns]

X = df[use_cols].copy()
y = df[target].astype(int).copy()

X.shape, y.shape, target

```




    ((30000, 33), (30000,), 'default_payment_next_month')




```python

cat_cols = [c for c in X.columns if X[c].dtype.name in ("category","object") or c in ["AGE_bin"]]
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X_enc.shape

```




    (30000, 36)




```python

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y
)
X_train.shape, X_test.shape, y_train.mean(), y_test.mean()

```




    ((24000, 36), (6000, 36), 0.22120833333333334, 0.22116666666666668)




```python

# Guardar datasets
(X_train).to_parquet(PROC/"X_train.parquet", index=False)
(X_test).to_parquet(PROC/"X_test.parquet", index=False)
y_train.to_frame("target").to_parquet(PROC/"y_train.parquet", index=False)
y_test.to_frame("target").to_parquet(PROC/"y_test.parquet", index=False)

# Guardar dataset completo con features (para referencia)
full = X_enc.copy()
full[target] = y.values
full.to_parquet(PROC/"features.parquet", index=False)

print("Guardado en:") 
for f in ["X_train.parquet","X_test.parquet","y_train.parquet","y_test.parquet","features.parquet"]:
    print(" -", PROC/f)

```

    Guardado en:
     - /Users/danielevarella/Desktop/gemini_data_science/credit-risk-scoring/data/processed/X_train.parquet
     - /Users/danielevarella/Desktop/gemini_data_science/credit-risk-scoring/data/processed/X_test.parquet
     - /Users/danielevarella/Desktop/gemini_data_science/credit-risk-scoring/data/processed/y_train.parquet
     - /Users/danielevarella/Desktop/gemini_data_science/credit-risk-scoring/data/processed/y_test.parquet
     - /Users/danielevarella/Desktop/gemini_data_science/credit-risk-scoring/data/processed/features.parquet



```python

rows = []
for c in X_enc.columns:
    desc = ""
    if c.startswith("utilization_"):
        desc = "BILL_AMT/LIMIT_BAL recortado"
    elif c.startswith("payment_ratio_"):
        desc = "PAY_AMT/BILL_AMT recortado"
    elif c.startswith("PAY_"):
        desc = "Historial de retraso (mes t)"
    elif c.startswith("BILL_AMT"):
        desc = "Monto facturado mes t"
    elif c.startswith("PAY_AMT"):
        desc = "Pago realizado mes t"
    rows.append({"feature": c, "description": desc})

catalog = pd.DataFrame(rows).sort_values("feature")
catalog.to_csv(CATALOG, index=False)
catalog.head(10)

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
      <th>feature</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>AGE</td>
      <td></td>
    </tr>
    <tr>
      <th>32</th>
      <td>AGE_bin_26-35</td>
      <td></td>
    </tr>
    <tr>
      <th>33</th>
      <td>AGE_bin_36-45</td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td>AGE_bin_46-60</td>
      <td></td>
    </tr>
    <tr>
      <th>35</th>
      <td>AGE_bin_60+</td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>BILL_AMT1</td>
      <td>Monto facturado mes t</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BILL_AMT2</td>
      <td>Monto facturado mes t</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BILL_AMT3</td>
      <td>Monto facturado mes t</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BILL_AMT4</td>
      <td>Monto facturado mes t</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BILL_AMT5</td>
      <td>Monto facturado mes t</td>
    </tr>
  </tbody>
</table>
</div>




**Listo.** Este notebook genera los datasets de entrenamiento/prueba y el catálogo de features.  
Siguiente paso: `03_model_training.ipynb` para baseline (LR, LightGBM) con validación cruzada y métricas (AUC, KS, Brier).

