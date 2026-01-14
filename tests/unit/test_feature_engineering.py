"""
Unit Tests - Feature Engineering
Tests para validar feature engineering de la API

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Metodología: DVP-PRO
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api.main import engineer_features, get_risk_band


class TestFeatureEngineering:
    """Tests para feature engineering."""

    def test_engineer_features_returns_dataframe(self, sample_credit_application):
        """Test que engineer_features retorna un DataFrame."""
        result = engineer_features(sample_credit_application)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_engineer_features_utilization_ratio(self, sample_credit_application):
        """Test cálculo de utilization ratio."""
        result = engineer_features(sample_credit_application)

        expected_utilization = sample_credit_application['BILL_AMT1'] / sample_credit_application['LIMIT_BAL']
        assert 'utilization_1' in result.columns
        assert abs(result['utilization_1'].iloc[0] - expected_utilization) < 0.001

    def test_engineer_features_utilization_zero_limit(self):
        """Test utilization ratio cuando LIMIT_BAL es cero."""
        data = {
            "LIMIT_BAL": 0,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 30,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 1000, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
        }
        result = engineer_features(data)
        assert result['utilization_1'].iloc[0] == 0

    def test_engineer_features_payment_ratios(self, sample_credit_application):
        """Test cálculo de payment ratios."""
        result = engineer_features(sample_credit_application)

        # Verificar que existen las 6 columnas de payment_ratio
        for i in range(1, 7):
            ratio_col = f'payment_ratio_{i}'
            assert ratio_col in result.columns

        # Verificar cálculo específico para payment_ratio_1
        expected_ratio = sample_credit_application['PAY_AMT1'] / sample_credit_application['BILL_AMT1']
        assert abs(result['payment_ratio_1'].iloc[0] - expected_ratio) < 0.001

    def test_engineer_features_payment_ratio_zero_bill(self):
        """Test payment ratio cuando BILL_AMT es cero."""
        data = {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 30,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 0, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 1000, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
        }
        result = engineer_features(data)
        assert result['payment_ratio_1'].iloc[0] == 0

    def test_engineer_features_age_bins_young(self):
        """Test age bins para persona joven (18-25)."""
        data = {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 2,
            "AGE": 22,  # Young
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 1000, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
        }
        result = engineer_features(data)

        assert result['AGE_bin_26-35'].iloc[0] == 0
        assert result['AGE_bin_36-45'].iloc[0] == 0
        assert result['AGE_bin_46-60'].iloc[0] == 0
        assert result['AGE_bin_60+'].iloc[0] == 0

    def test_engineer_features_age_bins_middle_age(self):
        """Test age bins para persona de edad media (26-35)."""
        data = {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 30,  # 26-35
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 1000, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
        }
        result = engineer_features(data)

        assert result['AGE_bin_26-35'].iloc[0] == 1
        assert result['AGE_bin_36-45'].iloc[0] == 0

    def test_engineer_features_age_bins_senior(self):
        """Test age bins para persona mayor (60+)."""
        data = {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 65,  # 60+
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 1000, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
        }
        result = engineer_features(data)

        assert result['AGE_bin_60+'].iloc[0] == 1

    def test_engineer_features_education_grouped_university(self):
        """Test education grouping para universidad (2)."""
        data = {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,  # Universidad
            "MARRIAGE": 1,
            "AGE": 30,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 1000, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
        }
        result = engineer_features(data)

        assert result['EDUCATION_grouped'].iloc[0] == 2

    def test_engineer_features_education_grouped_others(self):
        """Test education grouping para 'otros' (0, 4, 5, 6)."""
        for edu in [0, 4, 5, 6]:
            data = {
                "LIMIT_BAL": 50000,
                "SEX": 1,
                "EDUCATION": edu,
                "MARRIAGE": 1,
                "AGE": 30,
                "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 1000, "BILL_AMT2": 0, "BILL_AMT3": 0,
                "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
                "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
                "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
            }
            result = engineer_features(data)
            assert result['EDUCATION_grouped'].iloc[0] == 4

    def test_engineer_features_marriage_grouped_married(self):
        """Test marriage grouping para casado (1)."""
        data = {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,  # Casado
            "AGE": 30,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 1000, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
        }
        result = engineer_features(data)

        assert result['MARRIAGE_grouped'].iloc[0] == 1

    def test_engineer_features_marriage_grouped_others(self):
        """Test marriage grouping para 'otros' (0)."""
        data = {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 0,  # Otros
            "AGE": 30,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 1000, "BILL_AMT2": 0, "BILL_AMT3": 0,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
        }
        result = engineer_features(data)

        assert result['MARRIAGE_grouped'].iloc[0] == 3

    def test_engineer_features_all_input_features_present(self, sample_credit_application):
        """Test que todas las features de entrada están presentes."""
        result = engineer_features(sample_credit_application)

        required_features = [
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]

        for feature in required_features:
            assert feature in result.columns


class TestRiskBanding:
    """Tests para risk banding."""

    def test_get_risk_band_aprobado(self):
        """Test risk band APROBADO (< 0.20)."""
        assert get_risk_band(0.05) == "APROBADO"
        assert get_risk_band(0.10) == "APROBADO"
        assert get_risk_band(0.19) == "APROBADO"

    def test_get_risk_band_revision(self):
        """Test risk band REVISION (0.20 - 0.50)."""
        assert get_risk_band(0.20) == "REVISION"
        assert get_risk_band(0.35) == "REVISION"
        assert get_risk_band(0.49) == "REVISION"

    def test_get_risk_band_rechazo(self):
        """Test risk band RECHAZO (>= 0.50)."""
        assert get_risk_band(0.50) == "RECHAZO"
        assert get_risk_band(0.75) == "RECHAZO"
        assert get_risk_band(0.99) == "RECHAZO"

    def test_get_risk_band_edge_cases(self):
        """Test edge cases para risk banding."""
        assert get_risk_band(0.0) == "APROBADO"
        assert get_risk_band(1.0) == "RECHAZO"
        assert get_risk_band(0.1999) == "APROBADO"
        assert get_risk_band(0.2000) == "REVISION"
        assert get_risk_band(0.4999) == "REVISION"
        assert get_risk_band(0.5000) == "RECHAZO"
