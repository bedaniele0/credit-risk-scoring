"""
Pytest Configuration and Fixtures
Centralized test fixtures for credit-risk-scoring project

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Metodología: DVP-PRO
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from fastapi.testclient import TestClient
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.main import app


@pytest.fixture
def test_client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
def sample_credit_application():
    """Sample valid credit application (APROBADO band)."""
    return {
        "LIMIT_BAL": 300000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 45,
        "PAY_0": -1,
        "PAY_2": -1,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -1,
        "PAY_6": -1,
        "BILL_AMT1": 12000,
        "BILL_AMT2": 11000,
        "BILL_AMT3": 9000,
        "BILL_AMT4": 8000,
        "BILL_AMT5": 6000,
        "BILL_AMT6": 5000,
        "PAY_AMT1": 12000,
        "PAY_AMT2": 11000,
        "PAY_AMT3": 9000,
        "PAY_AMT4": 8000,
        "PAY_AMT5": 6000,
        "PAY_AMT6": 5000
    }


@pytest.fixture
def sample_high_risk_application():
    """Sample high-risk credit application (RECHAZO band)."""
    return {
        "LIMIT_BAL": 10000,
        "SEX": 1,
        "EDUCATION": 3,
        "MARRIAGE": 2,
        "AGE": 23,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": 2,
        "PAY_4": 3,
        "PAY_5": 3,
        "PAY_6": 3,
        "BILL_AMT1": 8000,
        "BILL_AMT2": 7500,
        "BILL_AMT3": 7000,
        "BILL_AMT4": 6500,
        "BILL_AMT5": 6000,
        "BILL_AMT6": 5500,
        "PAY_AMT1": 0,
        "PAY_AMT2": 0,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    }


@pytest.fixture
def sample_medium_risk_application():
    """Sample medium-risk credit application (REVISION band)."""
    return {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 25,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'LIMIT_BAL': np.random.randint(10000, 500000, n_samples),
        'SEX': np.random.choice([1, 2], n_samples),
        'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples),
        'MARRIAGE': np.random.choice([1, 2, 3], n_samples),
        'AGE': np.random.randint(21, 70, n_samples),
        'PAY_0': np.random.randint(-1, 3, n_samples),
        'PAY_2': np.random.randint(-1, 3, n_samples),
        'PAY_3': np.random.randint(-1, 3, n_samples),
        'PAY_4': np.random.randint(-1, 3, n_samples),
        'PAY_5': np.random.randint(-1, 3, n_samples),
        'PAY_6': np.random.randint(-1, 3, n_samples),
        'BILL_AMT1': np.random.randint(0, 100000, n_samples),
        'BILL_AMT2': np.random.randint(0, 100000, n_samples),
        'BILL_AMT3': np.random.randint(0, 100000, n_samples),
        'BILL_AMT4': np.random.randint(0, 100000, n_samples),
        'BILL_AMT5': np.random.randint(0, 100000, n_samples),
        'BILL_AMT6': np.random.randint(0, 100000, n_samples),
        'PAY_AMT1': np.random.randint(0, 50000, n_samples),
        'PAY_AMT2': np.random.randint(0, 50000, n_samples),
        'PAY_AMT3': np.random.randint(0, 50000, n_samples),
        'PAY_AMT4': np.random.randint(0, 50000, n_samples),
        'PAY_AMT5': np.random.randint(0, 50000, n_samples),
        'PAY_AMT6': np.random.randint(0, 50000, n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.78, 0.22])
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_drift_reference_data():
    """Reference data for drift detection testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.poisson(5, 1000)
    })


@pytest.fixture
def sample_drift_current_data_no_drift():
    """Current data without drift."""
    np.random.seed(43)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.poisson(5, 1000)
    })


@pytest.fixture
def sample_drift_current_data_with_drift():
    """Current data with drift in feature2."""
    np.random.seed(44)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(2, 1, 1000),  # Drift!
        'feature3': np.random.poisson(5, 1000)
    })


@pytest.fixture
def mock_model_metadata():
    """Mock model metadata."""
    return {
        "version": "1.0.0",
        "training_date": "2025-12-09",
        "threshold": 0.12,
        "metrics": {
            "auc_roc": 0.7813,
            "ks": 0.4251,
            "recall": 0.8704,
            "precision": 0.3107
        }
    }


@pytest.fixture
def invalid_credit_application():
    """Invalid credit application for error testing."""
    return {
        "LIMIT_BAL": -1000,  # Invalid: negative
        "SEX": 3,  # Invalid: out of range
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 15,  # Invalid: too young
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 0,
        "BILL_AMT2": 0,
        "BILL_AMT3": 0,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 0,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    }
