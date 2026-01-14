"""
Unit Tests - Monitoring
Tests para drift detection y alert system

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Metodología: DVP-PRO
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from monitoring.drift_monitor import DriftMonitor
from monitoring.alerts import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertChannel,
    SlackAlerter,
    TeamsAlerter,
    EmailAlerter
)


class TestDriftMonitor:
    """Tests para DriftMonitor."""

    def test_drift_monitor_initialization(self):
        """Test inicialización del monitor."""
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        reference_scores = np.random.uniform(0, 1, 100)

        monitor = DriftMonitor(reference_data, reference_scores)

        assert monitor.reference_data is not None
        assert monitor.reference_scores is not None
        assert monitor.baseline_ks >= 0

    def test_calculate_psi_no_drift(self):
        """Test PSI cuando no hay drift."""
        np.random.seed(42)
        reference_data = pd.DataFrame({'feat': [1, 2, 3]})
        reference_scores = np.array([0.1, 0.2, 0.3])
        monitor = DriftMonitor(reference_data, reference_scores)

        # Misma distribución
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)

        psi = monitor.calculate_psi(expected, actual)

        # PSI debería ser bajo (< 0.1)
        assert psi < 0.15  # Tolerancia para variabilidad aleatoria

    def test_calculate_psi_with_drift(self):
        """Test PSI cuando hay drift significativo."""
        np.random.seed(42)
        reference_data = pd.DataFrame({'feat': [1, 2, 3]})
        reference_scores = np.array([0.1, 0.2, 0.3])
        monitor = DriftMonitor(reference_data, reference_scores)

        # Distribuciones diferentes
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(3, 1, 1000)  # Mean shifted

        psi = monitor.calculate_psi(expected, actual)

        # PSI debería ser alto (>= 0.25)
        assert psi >= 0.25

    def test_calculate_ks_statistic(self):
        """Test cálculo de KS statistic."""
        np.random.seed(42)
        reference_data = pd.DataFrame({'feat': [1, 2, 3]})
        reference_scores = np.array([0.1, 0.2, 0.3])
        monitor = DriftMonitor(reference_data, reference_scores)

        # Crear scores y labels simulados
        scores = np.concatenate([
            np.random.uniform(0, 0.5, 100),  # Good clients (low scores)
            np.random.uniform(0.5, 1.0, 100)  # Bad clients (high scores)
        ])
        labels = np.array([0]*100 + [1]*100)

        ks = monitor.calculate_ks_statistic(scores, labels)

        # KS debe estar entre 0 y 1
        assert 0 <= ks <= 1

    def test_monitor_scores(self):
        """Test monitoreo de scores."""
        np.random.seed(42)
        reference_data = pd.DataFrame({'feat': np.random.normal(0, 1, 100)})
        reference_scores = np.random.uniform(0, 1, 100)
        monitor = DriftMonitor(reference_data, reference_scores)

        production_scores = np.random.uniform(0, 1, 50)

        result = monitor.monitor_scores(production_scores)

        assert "timestamp" in result
        assert "n_samples" in result
        assert "metrics" in result
        assert "alerts" in result
        assert result["n_samples"] == 50

        # Verificar métricas
        assert "psi" in result["metrics"]
        assert "score_stats" in result["metrics"]

    def test_monitor_scores_with_labels(self):
        """Test monitoreo de scores con labels."""
        np.random.seed(42)
        reference_data = pd.DataFrame({'feat': np.random.normal(0, 1, 100)})
        reference_scores = np.random.uniform(0, 1, 100)
        monitor = DriftMonitor(reference_data, reference_scores)

        production_scores = np.random.uniform(0, 1, 50)
        production_labels = np.random.choice([0, 1], 50)

        result = monitor.monitor_scores(production_scores, production_labels)

        # Debe incluir métricas de KS
        assert "ks" in result["metrics"]
        assert "baseline" in result["metrics"]["ks"]
        assert "current" in result["metrics"]["ks"]
        assert "decay" in result["metrics"]["ks"]

    def test_monitor_features(self):
        """Test monitoreo de features."""
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        reference_scores = np.random.uniform(0, 1, 100)
        monitor = DriftMonitor(reference_data, reference_scores)

        production_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50)
        })

        result = monitor.monitor_features(production_data)

        assert "timestamp" in result
        assert "n_samples" in result
        assert "features" in result
        assert "drifted_features" in result
        assert result["n_samples"] == 50

    def test_psi_status_classification(self):
        """Test clasificación de status PSI."""
        reference_data = pd.DataFrame({'feat': [1, 2, 3]})
        reference_scores = np.array([0.1, 0.2, 0.3])
        monitor = DriftMonitor(reference_data, reference_scores)

        assert monitor._get_psi_status(0.05) == "OK"
        assert monitor._get_psi_status(0.15) == "WARNING"
        assert monitor._get_psi_status(0.30) == "CRITICAL"


class TestAlerts:
    """Tests para el sistema de alertas."""

    def test_alert_creation(self):
        """Test creación de alerta."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.INFO,
            metadata={'key': 'value'}
        )

        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert alert.severity == AlertSeverity.INFO
        assert alert.metadata == {'key': 'value'}
        assert alert.timestamp is not None

    def test_email_alerter_missing_config(self):
        """Test EmailAlerter sin configuración retorna False."""
        alerter = EmailAlerter()
        alert = Alert(
            title="Email Test",
            message="Test message",
            severity=AlertSeverity.CRITICAL
        )

        # Sin configuración SMTP completa, debe retornar False
        result = alerter.send(alert)
        assert result == False

    def test_slack_alerter_missing_config(self):
        """Test SlackAlerter sin configuración retorna False."""
        alerter = SlackAlerter()
        alert = Alert(
            title="Slack Test",
            message="Test message",
            severity=AlertSeverity.INFO
        )

        # Sin webhook URL y enabled=False, debe retornar False
        result = alerter.send(alert)
        assert result == False

    def test_teams_alerter_missing_config(self):
        """Test TeamsAlerter sin configuración retorna False."""
        alerter = TeamsAlerter()
        alert = Alert(
            title="Teams Test",
            message="Test message",
            severity=AlertSeverity.WARNING
        )

        # Sin webhook URL y enabled=False, debe retornar False
        result = alerter.send(alert)
        assert result == False

    def test_alert_manager_initialization(self):
        """Test inicialización de AlertManager."""
        manager = AlertManager()
        assert len(manager.alerters) == 4  # Slack, Teams, Email, Webhook

    def test_alert_manager_send_alert(self):
        """Test envío de alerta por AlertManager."""
        manager = AlertManager()

        # Sin canales habilitados, no envía nada
        result = manager.send_alert(
            title="Manager Test",
            message="Test message",
            severity=AlertSeverity.INFO
        )

        assert isinstance(result, dict)

    def test_alert_manager_send_performance_alert(self):
        """Test send_performance_alert."""
        manager = AlertManager()

        # Performance degradation
        result = manager.send_performance_alert(
            metric_name="AUC-ROC",
            current_value=0.75,
            expected_value=0.80
        )
        assert isinstance(result, dict)

    def test_alert_manager_send_drift_alert(self):
        """Test send_drift_alert."""
        manager = AlertManager()

        result = manager.send_drift_alert(
            feature_name="PAY_0",
            drift_score=0.25,
            threshold=0.20
        )
        assert isinstance(result, dict)

    def test_alert_manager_send_training_alert(self):
        """Test send_training_alert."""
        manager = AlertManager()

        # Success
        result_success = manager.send_training_alert(
            status="success",
            metrics={"auc_roc": 0.78}
        )
        assert isinstance(result_success, dict)

        # Failure
        result_failure = manager.send_training_alert(
            status="failure"
        )
        assert isinstance(result_failure, dict)

    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_alert_channel_enum(self):
        """Test AlertChannel enum values."""
        assert AlertChannel.SLACK.value == "slack"
        assert AlertChannel.TEAMS.value == "teams"
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.WEBHOOK.value == "webhook"
