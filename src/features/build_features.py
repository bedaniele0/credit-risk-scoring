import pandas as pd
import numpy as np
import yaml
import os

def build_features():
    """
    Crea nuevas características y guarda el dataset procesado.
    """
    # Construir ruta al config.yaml de forma robusta
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '..', '..', 'config', 'config.yaml')

    # Cargar configuración
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Construir rutas de datos de manera robusta
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    raw_data_path = os.path.join(base_dir, config['paths']['raw_data'])
    
    print(f"Cargando datos desde: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    df_featured = df.copy()

    print("Creando nuevas características...")

    # 1. Ratios de Utilización de Crédito
    for i in range(1, 7):
        bill_col = f'BILL_AMT{i}'
        util_col = f'UTIL_RATIO_{i}'
        df_featured[util_col] = df_featured.apply(
            lambda row: row[bill_col] / row['LIMIT_BAL'] if row['LIMIT_BAL'] > 0 else 0,
            axis=1
        )

    # 2. Ratios de Pago
    for i in range(1, 7):
        pay_col = f'PAY_AMT{i}'
        bill_col = f'BILL_AMT{i}'
        ratio_col = f'PAY_RATIO_{i}'
        df_featured[ratio_col] = df_featured.apply(
            lambda row: row[pay_col] / row[bill_col] if row[bill_col] > 0 else 1,
            axis=1
        )

    # 3. Agrupación de Categorías
    df_featured['EDUCATION_CAT'] = df_featured['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df_featured['MARRIAGE_CAT'] = df_featured['MARRIAGE'].replace({0: 3})

    print("Características creadas:")
    new_cols = [col for col in df_featured.columns if col not in df.columns]
    print(new_cols)

    # Guardar dataset
    processed_data_dir = os.path.join(base_dir, config['paths']['processed_data'])
    os.makedirs(processed_data_dir, exist_ok=True)
    processed_data_path = os.path.join(processed_data_dir, 'featured_dataset.csv')
    
    df_featured.to_csv(processed_data_path, index=False)
    print(f"\nDataset procesado guardado en: {processed_data_path}")
    print(f"Nuevas dimensiones: {df_featured.shape}")

if __name__ == '__main__':
    build_features()
