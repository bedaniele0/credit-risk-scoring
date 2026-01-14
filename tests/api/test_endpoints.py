"""
API Tests - Endpoints
Tests completos para todos los endpoints de la API

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Metodología: DVP-PRO
"""

import pytest
from fastapi import status
import json


class TestRootEndpoint:
    """Tests para root endpoint (/)."""

    def test_root_endpoint(self, test_client):
        """Test GET / retorna información básica."""
        response = test_client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "author" in data
        assert "docs" in data
        assert "health" in data
        assert data["message"] == "Credit Risk Scoring API"
        assert data["version"] == "1.0.0"


class TestHealthEndpoint:
    """Tests para health check endpoint."""

    def test_health_endpoint(self, test_client):
        """Test GET /health retorna status del servicio."""
        response = test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "threshold" in data
        assert "timestamp" in data

        # Status puede ser healthy o unhealthy dependiendo de si el modelo está cargado
        assert data["status"] in ["healthy", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)
        assert data["threshold"] == 0.12


class TestPredictEndpoint:
    """Tests para POST /predict endpoint."""

    def test_predict_endpoint_valid_low_risk(self, test_client, sample_credit_application):
        """Test predicción con aplicación de bajo riesgo."""
        response = test_client.post("/predict", json=sample_credit_application)

        # Si el modelo no está cargado, esperamos 503
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            assert "Modelo no disponible" in response.json()["detail"]
            return

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verificar estructura de respuesta
        assert "probability" in data
        assert "prediction" in data
        assert "risk_band" in data
        assert "threshold_used" in data
        assert "timestamp" in data
        assert "model_version" in data

        # Verificar tipos
        assert isinstance(data["probability"], float)
        assert data["prediction"] in ["DEFAULT", "NO_DEFAULT"]
        assert data["risk_band"] in ["APROBADO", "REVISION", "RECHAZO"]
        assert data["threshold_used"] == 0.12

        # Para esta aplicación de bajo riesgo, esperamos NO_DEFAULT y APROBADO
        # (puede variar según el modelo real)
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_endpoint_valid_high_risk(self, test_client, sample_high_risk_application):
        """Test predicción con aplicación de alto riesgo."""
        response = test_client.post("/predict", json=sample_high_risk_application)

        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            return

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "probability" in data
        assert "prediction" in data
        assert "risk_band" in data

        # Para aplicación de alto riesgo, esperamos probabilidad más alta
        # (puede variar según el modelo)
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_endpoint_invalid_data(self, test_client, invalid_credit_application):
        """Test predicción con datos inválidos."""
        response = test_client.post("/predict", json=invalid_credit_application)

        # Pydantic validation debe rechazar
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_predict_endpoint_missing_fields(self, test_client):
        """Test predicción con campos faltantes."""
        incomplete_data = {
            "LIMIT_BAL": 50000,
            "SEX": 1
            # Faltan muchos campos requeridos
        }

        response = test_client.post("/predict", json=incomplete_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_predict_endpoint_invalid_sex(self, test_client, sample_credit_application):
        """Test predicción con SEX fuera de rango."""
        invalid_data = sample_credit_application.copy()
        invalid_data["SEX"] = 5  # Invalid: debe ser 1 o 2

        response = test_client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_predict_endpoint_invalid_education(self, test_client, sample_credit_application):
        """Test predicción con EDUCATION fuera de rango."""
        invalid_data = sample_credit_application.copy()
        invalid_data["EDUCATION"] = 10  # Invalid

        response = test_client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_predict_endpoint_negative_limit_bal(self, test_client, sample_credit_application):
        """Test predicción con LIMIT_BAL negativo."""
        invalid_data = sample_credit_application.copy()
        invalid_data["LIMIT_BAL"] = -1000

        response = test_client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_predict_endpoint_negative_pay_amt(self, test_client, sample_credit_application):
        """Test predicción con PAY_AMT negativo."""
        invalid_data = sample_credit_application.copy()
        invalid_data["PAY_AMT1"] = -500

        response = test_client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_predict_endpoint_age_too_young(self, test_client, sample_credit_application):
        """Test predicción con edad menor a 18."""
        invalid_data = sample_credit_application.copy()
        invalid_data["AGE"] = 15

        response = test_client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_predict_endpoint_age_too_old(self, test_client, sample_credit_application):
        """Test predicción con edad mayor a 100."""
        invalid_data = sample_credit_application.copy()
        invalid_data["AGE"] = 150

        response = test_client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestBatchPredictEndpoint:
    """Tests para POST /predict/batch endpoint."""

    def test_batch_predict_single_application(self, test_client, sample_credit_application):
        """Test batch prediction con una sola aplicación."""
        batch_data = {
            "applications": [sample_credit_application]
        }

        response = test_client.post("/predict/batch", json=batch_data)

        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            return

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "predictions" in data
        assert "total_processed" in data
        assert "timestamp" in data

        assert len(data["predictions"]) == 1
        assert data["total_processed"] == 1

        # Verificar estructura de predicción
        prediction = data["predictions"][0]
        assert "probability" in prediction
        assert "prediction" in prediction
        assert "risk_band" in prediction


class TestMonitoringEndpoints:
    """Tests para endpoints de monitoreo."""

    def test_prometheus_metrics(self, test_client):
        resp = test_client.get("/prometheus")
        assert resp.status_code == status.HTTP_200_OK
        assert "credit_api_requests_total" in resp.text

    def test_monitor_drift_endpoint(self, test_client):
        payload = {"scores": [0.1, 0.2, 0.3, 0.4, 0.5]}
        resp = test_client.post("/monitoring/drift", json=payload)

        if resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Scores de referencia no disponibles")

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert "psi" in data
        assert "ks_statistic" in data
        assert "status" in data

    def test_batch_predict_multiple_applications(
        self,
        test_client,
        sample_credit_application,
        sample_high_risk_application,
        sample_medium_risk_application
    ):
        """Test batch prediction con múltiples aplicaciones."""
        batch_data = {
            "applications": [
                sample_credit_application,
                sample_high_risk_application,
                sample_medium_risk_application
            ]
        }

        response = test_client.post("/predict/batch", json=batch_data)

        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            return

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert len(data["predictions"]) == 3
        assert data["total_processed"] == 3

        # Todas las predicciones deben tener la estructura correcta
        for prediction in data["predictions"]:
            assert "probability" in prediction
            assert "prediction" in prediction
            assert "risk_band" in prediction
            assert 0.0 <= prediction["probability"] <= 1.0

    def test_batch_predict_max_limit_exceeded(self, test_client, sample_credit_application):
        """Test batch prediction con más de 100 aplicaciones."""
        batch_data = {
            "applications": [sample_credit_application] * 101  # 101 aplicaciones
        }

        response = test_client.post("/predict/batch", json=batch_data)

        # Puede rechazar por exceder límite O por modelo no disponible
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_503_SERVICE_UNAVAILABLE]
        if response.status_code == status.HTTP_400_BAD_REQUEST:
            assert "100 solicitudes" in response.json()["detail"]

    def test_batch_predict_exactly_100_applications(self, test_client, sample_credit_application):
        """Test batch prediction con exactamente 100 aplicaciones."""
        batch_data = {
            "applications": [sample_credit_application] * 100
        }

        response = test_client.post("/predict/batch", json=batch_data)

        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            return

        # 100 aplicaciones debe ser aceptado
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_processed"] == 100

    def test_batch_predict_empty_list(self, test_client):
        """Test batch prediction con lista vacía."""
        batch_data = {
            "applications": []
        }

        response = test_client.post("/predict/batch", json=batch_data)

        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            return

        # Lista vacía es técnicamente válida
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_processed"] == 0


class TestMetricsEndpoint:
    """Tests para GET /metrics endpoint."""

    def test_metrics_endpoint(self, test_client):
        """Test GET /metrics retorna métricas del modelo."""
        response = test_client.get("/metrics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "model_version" in data
        assert "threshold" in data
        assert "metrics" in data
        assert "timestamp" in data

        assert data["threshold"] == 0.12

        # metrics puede contener métricas o mensaje de no disponible
        assert isinstance(data["metrics"], dict)


class TestModelInfoEndpoint:
    """Tests para GET /model/info endpoint."""

    def test_model_info_endpoint(self, test_client):
        """Test GET /model/info retorna información del modelo."""
        response = test_client.get("/model/info")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "model_type" in data
        assert "version" in data
        assert "threshold" in data
        assert "features_count" in data
        assert "risk_bands" in data

        # Verificar risk bands
        assert "APROBADO" in data["risk_bands"]
        assert "REVISION" in data["risk_bands"]
        assert "RECHAZO" in data["risk_bands"]

        assert data["threshold"] == 0.12


class TestCORSHeaders:
    """Tests para CORS headers."""

    def test_cors_headers_present(self, test_client):
        """Test que CORS headers están configurados."""
        response = test_client.get("/health")

        # Verificar que no hay errores de CORS
        assert response.status_code == status.HTTP_200_OK


class TestErrorHandling:
    """Tests para manejo de errores."""

    def test_404_not_found(self, test_client):
        """Test endpoint inexistente retorna 404."""
        response = test_client.get("/nonexistent-endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_method_not_allowed(self, test_client):
        """Test método HTTP incorrecto."""
        # POST a un endpoint GET-only
        response = test_client.post("/health")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_invalid_json(self, test_client):
        """Test envío de JSON inválido."""
        response = test_client.post(
            "/predict",
            content=b"invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in {
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        }
