"""
Integration Tests - API
Tests de integración end-to-end para flujos completos

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Metodología: DVP-PRO
"""

import pytest
from fastapi import status
import time


class TestAPIIntegration:
    """Tests de integración para API."""

    def test_complete_prediction_flow(self, test_client, sample_credit_application):
        """Test flujo completo: health check -> prediction."""
        # 1. Verificar que el servicio está saludable
        health_response = test_client.get("/health")
        assert health_response.status_code == status.HTTP_200_OK

        # 2. Obtener información del modelo
        info_response = test_client.get("/model/info")
        assert info_response.status_code == status.HTTP_200_OK
        info_data = info_response.json()
        threshold = info_data["threshold"]

        # 3. Hacer predicción
        predict_response = test_client.post("/predict", json=sample_credit_application)

        if predict_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not loaded")

        assert predict_response.status_code == status.HTTP_200_OK
        predict_data = predict_response.json()

        # 4. Verificar que threshold coincide
        assert predict_data["threshold_used"] == threshold
        assert predict_data["threshold_used"] == 0.12

    def test_batch_prediction_flow(
        self,
        test_client,
        sample_credit_application,
        sample_high_risk_application,
        sample_medium_risk_application
    ):
        """Test flujo completo de batch prediction."""
        # 1. Health check
        health_response = test_client.get("/health")
        assert health_response.status_code == status.HTTP_200_OK

        # 2. Batch prediction
        batch_data = {
            "applications": [
                sample_credit_application,
                sample_high_risk_application,
                sample_medium_risk_application
            ]
        }

        batch_response = test_client.post("/predict/batch", json=batch_data)

        if batch_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not loaded")

        assert batch_response.status_code == status.HTTP_200_OK
        batch_result = batch_response.json()

        # 3. Verificar que todas las predicciones fueron procesadas
        assert batch_result["total_processed"] == 3
        assert len(batch_result["predictions"]) == 3

        # 4. Verificar consistencia de risk bands
        for prediction in batch_result["predictions"]:
            prob = prediction["probability"]
            risk_band = prediction["risk_band"]

            if prob < 0.20:
                assert risk_band == "APROBADO"
            elif prob < 0.50:
                assert risk_band == "REVISION"
            else:
                assert risk_band == "RECHAZO"

    def test_api_performance_multiple_requests(self, test_client, sample_credit_application):
        """Test performance con múltiples requests."""
        # Health check
        health_response = test_client.get("/health")
        if health_response.json().get("model_loaded") == False:
            pytest.skip("Model not loaded")

        # Hacer 10 predicciones consecutivas
        responses = []
        start_time = time.time()

        for _ in range(10):
            response = test_client.post("/predict", json=sample_credit_application)
            responses.append(response)

        elapsed_time = time.time() - start_time

        # Verificar que todas tuvieron éxito
        for response in responses:
            if response.status_code != status.HTTP_503_SERVICE_UNAVAILABLE:
                assert response.status_code == status.HTTP_200_OK

        # Performance: 10 predicciones en menos de 5 segundos
        if all(r.status_code == 200 for r in responses):
            assert elapsed_time < 5.0

    def test_api_consistency_same_input(self, test_client, sample_credit_application):
        """Test que la misma entrada produce la misma predicción."""
        # Health check
        health_response = test_client.get("/health")
        if health_response.json().get("model_loaded") == False:
            pytest.skip("Model not loaded")

        # Hacer dos predicciones con la misma entrada
        response1 = test_client.post("/predict", json=sample_credit_application)
        response2 = test_client.post("/predict", json=sample_credit_application)

        if response1.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Model not loaded")

        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK

        # Las probabilidades deben ser idénticas (modelo determinístico)
        prob1 = response1.json()["probability"]
        prob2 = response2.json()["probability"]

        assert prob1 == prob2

    def test_risk_band_boundaries(self, test_client):
        """Test comportamiento en los límites de risk bands."""
        # Este test verifica el comportamiento en los bordes exactos
        # de las bandas de riesgo
        test_cases = [
            {"expected_band": "APROBADO", "description": "Low risk"},
            {"expected_band": "REVISION", "description": "Medium risk"},
            {"expected_band": "RECHAZO", "description": "High risk"}
        ]

        for test_case in test_cases:
            # Crear aplicación que debería caer en esta banda
            # (esto requeriría conocer exactamente qué inputs producen qué probabilidades)
            pass  # Placeholder - requiere calibración específica del modelo


class TestMetricsIntegration:
    """Tests de integración para métricas."""

    def test_metrics_consistency(self, test_client):
        """Test que las métricas son consistentes."""
        # Obtener métricas
        metrics_response = test_client.get("/metrics")
        assert metrics_response.status_code == status.HTTP_200_OK
        metrics_data = metrics_response.json()

        # Obtener model info
        info_response = test_client.get("/model/info")
        assert info_response.status_code == status.HTTP_200_OK
        info_data = info_response.json()

        # Threshold debe ser consistente
        assert metrics_data["threshold"] == info_data["threshold"]

    def test_health_and_model_info_consistency(self, test_client):
        """Test consistencia entre health y model info."""
        health_response = test_client.get("/health")
        info_response = test_client.get("/model/info")

        assert health_response.status_code == status.HTTP_200_OK
        assert info_response.status_code == status.HTTP_200_OK

        health_data = health_response.json()
        info_data = info_response.json()

        # Model version debe ser consistente
        assert health_data["model_version"] == info_data["version"]


class TestErrorRecovery:
    """Tests de recuperación de errores."""

    def test_invalid_input_recovery(self, test_client, invalid_credit_application, sample_credit_application):
        """Test que después de un error, la API sigue funcionando."""
        # 1. Enviar request inválido
        invalid_response = test_client.post("/predict", json=invalid_credit_application)
        assert invalid_response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # 2. Enviar request válido inmediatamente después
        valid_response = test_client.post("/predict", json=sample_credit_application)

        if valid_response.status_code != status.HTTP_503_SERVICE_UNAVAILABLE:
            assert valid_response.status_code == status.HTTP_200_OK

    def test_batch_partial_invalid(self, test_client, sample_credit_application, invalid_credit_application):
        """Test batch con algunos datos inválidos."""
        batch_data = {
            "applications": [
                sample_credit_application,
                invalid_credit_application  # Uno inválido
            ]
        }

        response = test_client.post("/predict/batch", json=batch_data)

        # Debe rechazar todo el batch si hay uno inválido
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestDocumentation:
    """Tests para documentación automática."""

    def test_openapi_docs_available(self, test_client):
        """Test que la documentación OpenAPI está disponible."""
        response = test_client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK

        openapi_schema = response.json()
        assert "openapi" in openapi_schema
        assert "info" in openapi_schema
        assert "paths" in openapi_schema

        # Verificar endpoints documentados
        assert "/predict" in openapi_schema["paths"]
        assert "/predict/batch" in openapi_schema["paths"]
        assert "/health" in openapi_schema["paths"]
        assert "/metrics" in openapi_schema["paths"]
