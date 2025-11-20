#!/usr/bin/env python3
"""
make_dataset.py — F3 Ingesta y preparación inicial (DVP-PRO)

Lee el dataset "Default of Credit Card Clients" desde data/raw en formato
CSV o XLS/XLSX, realiza validaciones mínimas, estandariza columnas, crea
`default_flag` y genera un dataset limpio en Parquet/CSV dentro de
`data/processed/`.

Uso:
    python src/data/make_dataset.py

Requisitos:
    - pandas
    - openpyxl (solo si usas .xlsx)

Autor: Ing. Daniel Varela Pérez
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Candidatos de archivo (prioridad: CSV, luego XLSX, luego XLS)
CANDIDATES = [
    "default_of_credit_card_clients.csv",
    "default of credit card clients.csv",
    "default_of_credit_card_clients.xlsx",
    "default of credit card clients.xlsx",
    "default_of_credit_card_clients.xls",
    "default of credit card clients.xls",
]


def find_input_file() -> Path:
    for name in CANDIDATES:
        p = RAW_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No se encontró archivo en {RAW_DIR}. Coloca el CSV/XLSX/XLS con el dataset."
    )


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Algunos CSV exportados desde Numbers llevan encabezado en la primera fila.
        df0 = pd.read_csv(path, nrows=5)
        # Si la primera columna se llama "ID" asumimos header en fila 0; si no, probamos header=1
        if "ID" in df0.columns or "ID" in map(str.upper, df0.columns):
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, header=1)
    elif suffix in {".xlsx", ".xls"}:
        # El archivo UCI original tiene metadatos en la fila 0 y encabezados en la fila 1
        # openpyxl funciona para .xlsx; para .xls se requiere xlrd compatible.
        try:
            df = pd.read_excel(path, header=1)
        except Exception as e:
            raise RuntimeError(
                "Error al leer Excel. Si es .xls, conviértelo a .xlsx desde Numbers/Excel."
            ) from e
    else:
        raise ValueError(f"Formato no soportado: {suffix}")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Renombrar exactamente como en el dataset oficial
    rename_map = {
        "default.payment.next.month": "default_flag",
        "PAY_1": "PAY_0",  # algunas versiones usan PAY_1 en lugar de PAY_0
    }
    # Normaliza nombres
    df = df.rename(columns=rename_map)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def basic_validations(df: pd.DataFrame) -> None:
    required = [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "default_flag",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Chequeo rápido de tamaños y nulos
    n_rows, n_cols = df.shape
    n_null = int(df.isna().sum().sum())
    print(f"[VALIDATION] shape={n_rows}x{n_cols} | total_nulls={n_null}")


def cast_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Asegurar tipo entero donde aplica
    int_like = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        *[f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]],
    ]
    for c in int_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Imputación simple
    for c in df.columns:
        if c == "default_flag":
            continue
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float")
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Derivadas seguras
    if "BILL_AMT1" in df.columns and "LIMIT_BAL" in df.columns:
        df["utilization_1"] = (df["BILL_AMT1"]) / (df["LIMIT_BAL"].replace(0, pd.NA))
        df["utilization_1"] = df["utilization_1"].fillna(0.0)

    for k in range(1, 7):
        b = f"BILL_AMT{k}"
        p = f"PAY_AMT{k}"
        if b in df.columns and p in df.columns:
            df[f"payment_ratio_{k}"] = df[p] / (df[b].replace(0, pd.NA))
            df[f"payment_ratio_{k}"] = df[f"payment_ratio_{k}"].fillna(0.0)

    # Asegurar binaria 0/1 en target si viene como float/int distinta
    df["default_flag"] = pd.to_numeric(df["default_flag"], errors="coerce").fillna(0).astype(int)

    return df


def save_outputs(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = PROCESSED_DIR / "train_dataset.parquet"
    out_csv = PROCESSED_DIR / "train_dataset.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"[OUTPUT] Guardado: {out_parquet} | {out_csv}")


def main():
    try:
        src = find_input_file()
        print(f"[INFO] Archivo detectado: {src}")
        df = read_dataset(src)
        df = standardize_columns(df)

        # Crear default_flag si aún no existe y la columna original está presente
        if "default_flag" not in df.columns and "default.payment.next.month" in df.columns:
            df = df.rename(columns={"default.payment.next.month": "default_flag"})

        basic_validations(df)
        df = cast_and_clean(df)
        basic_validations(df)  # validar de nuevo tras limpieza
        save_outputs(df)
        print("[DONE] Dataset procesado correctamente.")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
