import os
from pathlib import Path

import pandas as pd

# Carpeta donde están tus parquet (según tu captura)
BASE = Path("data/processed")

ARCHIVOS = [
    "features.parquet",
    "X_train.parquet",
    "X_test.parquet",
    "y_train.parquet",
    "y_test.parquet",
]

for nombre in ARCHIVOS:
    parquet_path = BASE / nombre
    if not parquet_path.exists():
        print(f"⚠️ Archivo NO encontrado: {parquet_path}")
        continue

    print(f"📥 Leyendo: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    csv_path = BASE / (parquet_path.stem + ".csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ Convertido: {parquet_path} → {csv_path} "
          f"({df.shape[0]} filas, {df.shape[1]} columnas)")
