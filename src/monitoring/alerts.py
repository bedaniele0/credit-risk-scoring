"""
============================================================================
alerts.py - Multi-Channel Alerting System para Credit Risk Scoring
============================================================================
Sistema de alertas con soporte para:
- Slack
- Microsoft Teams
- Email (SMTP)
- Webhook genérico

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Tel: +52 55 4189 3428
Metodología: DVP-PRO
============================================================================
"""

import json
import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class AlertSeverity(Enum):
    """Severidad de la alerta."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Canales de notificación."""

    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class Alert:
    """Representa una alerta del sistema."""

    title: str
    message: str
    severity: AlertSeverity
    timestamp: Optional[str] = None
    metrics: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# BASE ALERTER
# ============================================================================


class BaseAlerter:
    """Clase base para alerters."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el alerter.

        Args:
            config: Configuración del alerter
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)

    def send(self, alert: Alert) -> bool:
        """
        Envía una alerta (debe ser implementado por subclases).

        Args:
            alert: Alerta a enviar

        Returns:
            True si se envió exitosamente, False caso contrario
        """
        raise NotImplementedError("Subclasses must implement send()")

    def format_message(self, alert: Alert) -> str:
        """
        Formatea el mensaje de la alerta.

        Args:
            alert: Alerta a formatear

        Returns:
            Mensaje formateado
        """
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }

        msg = f"{severity_emoji.get(alert.severity, '📢')} **{alert.title}**\n\n"
        msg += f"{alert.message}\n\n"
        msg += f"**Timestamp:** {alert.timestamp}\n"
        msg += f"**Severity:** {alert.severity.value.upper()}\n"

        if alert.metrics:
            msg += "\n**Metrics:**\n"
            for key, value in alert.metrics.items():
                msg += f"  - {key}: {value}\n"

        if alert.metadata:
            msg += "\n**Metadata:**\n"
            for key, value in alert.metadata.items():
                msg += f"  - {key}: {value}\n"

        return msg


# ============================================================================
# SLACK ALERTER
# ============================================================================


class SlackAlerter(BaseAlerter):
    """Alerter para Slack usando Incoming Webhooks."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", self.config.get("webhook_url"))
        self.channel = os.getenv("SLACK_CHANNEL", self.config.get("channel", "#alerts"))
        self.enabled = os.getenv("SLACK_ENABLED", "false").lower() == "true"

    def send(self, alert: Alert) -> bool:
        """Envía alerta a Slack."""
        if not self.enabled:
            logger.debug("Slack alerts disabled")
            return False

        if not self.webhook_url:
            logger.error("Slack webhook URL not configured")
            return False

        try:
            # Color basado en severidad
            color_map = {
                AlertSeverity.INFO: "#36a64f",  # Green
                AlertSeverity.WARNING: "#ff9800",  # Orange
                AlertSeverity.ERROR: "#f44336",  # Red
                AlertSeverity.CRITICAL: "#9c27b0",  # Purple
            }

            # Construir payload
            payload = {
                "channel": self.channel,
                "username": "Credit Risk Bot",
                "icon_emoji": ":credit_card:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": alert.title,
                        "text": alert.message,
                        "footer": f"Credit Risk Scoring | {alert.severity.value.upper()}",
                        "ts": int(datetime.now().timestamp()),
                        "fields": self._build_fields(alert),
                    }
                ],
            }

            # Enviar
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("Slack alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _build_fields(self, alert: Alert) -> List[Dict]:
        """Construye campos para Slack attachment."""
        fields = []

        if alert.metrics:
            for key, value in alert.metrics.items():
                fields.append({"title": key, "value": str(value), "short": True})

        if alert.metadata:
            for key, value in alert.metadata.items():
                fields.append({"title": key, "value": str(value), "short": True})

        return fields


# ============================================================================
# TEAMS ALERTER
# ============================================================================


class TeamsAlerter(BaseAlerter):
    """Alerter para Microsoft Teams usando Incoming Webhooks."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.webhook_url = os.getenv("TEAMS_WEBHOOK_URL", self.config.get("webhook_url"))
        self.enabled = os.getenv("TEAMS_ENABLED", "false").lower() == "true"

    def send(self, alert: Alert) -> bool:
        """Envía alerta a Microsoft Teams."""
        if not self.enabled:
            logger.debug("Teams alerts disabled")
            return False

        if not self.webhook_url:
            logger.error("Teams webhook URL not configured")
            return False

        try:
            # Color basado en severidad
            color_map = {
                AlertSeverity.INFO: "0078D4",  # Blue
                AlertSeverity.WARNING: "FF8C00",  # Orange
                AlertSeverity.ERROR: "D13438",  # Red
                AlertSeverity.CRITICAL: "5C2D91",  # Purple
            }

            # Construir payload (Adaptive Cards format)
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": alert.title,
                "themeColor": color_map.get(alert.severity, "808080"),
                "title": f"Credit Risk Alert: {alert.title}",
                "sections": [
                    {
                        "activityTitle": alert.message,
                        "activitySubtitle": f"Severity: {alert.severity.value.upper()}",
                        "facts": self._build_facts(alert),
                    }
                ],
            }

            # Enviar
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("Teams alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
            return False

    def _build_facts(self, alert: Alert) -> List[Dict]:
        """Construye facts para Teams card."""
        facts = [{"name": "Timestamp", "value": alert.timestamp}]

        if alert.metrics:
            for key, value in alert.metrics.items():
                facts.append({"name": key, "value": str(value)})

        if alert.metadata:
            for key, value in alert.metadata.items():
                facts.append({"name": key, "value": str(value)})

        return facts


# ============================================================================
# EMAIL ALERTER
# ============================================================================


class EmailAlerter(BaseAlerter):
    """Alerter para Email usando SMTP."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.smtp_server = os.getenv("SMTP_SERVER", self.config.get("smtp_server", "smtp.gmail.com"))
        self.smtp_port = int(os.getenv("SMTP_PORT", self.config.get("smtp_port", 587)))
        self.smtp_username = os.getenv("SMTP_USERNAME", self.config.get("smtp_username"))
        self.smtp_password = os.getenv("SMTP_PASSWORD", self.config.get("smtp_password"))
        self.from_email = os.getenv("ALERT_EMAIL_FROM", self.config.get("from_email"))
        self.to_email = os.getenv("ALERT_EMAIL_TO", self.config.get("to_email"))
        self.enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

    def send(self, alert: Alert) -> bool:
        """Envía alerta por email."""
        if not self.enabled:
            logger.debug("Email alerts disabled")
            return False

        if not all([self.smtp_username, self.smtp_password, self.from_email, self.to_email]):
            logger.error("Email configuration incomplete")
            return False

        try:
            # Crear mensaje
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Credit Risk Alert] {alert.title}"
            msg["From"] = self.from_email
            msg["To"] = self.to_email

            # HTML body
            html_body = self._build_html_body(alert)
            msg.attach(MIMEText(html_body, "html"))

            # Enviar
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email alert sent to {self.to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _build_html_body(self, alert: Alert) -> str:
        """Construye HTML body para email."""
        severity_colors = {
            AlertSeverity.INFO: "#2196F3",
            AlertSeverity.WARNING: "#FF9800",
            AlertSeverity.ERROR: "#F44336",
            AlertSeverity.CRITICAL: "#9C27B0",
        }

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: {severity_colors.get(alert.severity, '#808080')};
                          color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .metrics {{ background-color: #f5f5f5; padding: 15px; margin: 10px 0; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{alert.title}</h2>
                <p>Severity: {alert.severity.value.upper()}</p>
            </div>
            <div class="content">
                <p>{alert.message}</p>
                <p><strong>Timestamp:</strong> {alert.timestamp}</p>
        """

        if alert.metrics:
            html += '<div class="metrics"><h3>Metrics:</h3><ul>'
            for key, value in alert.metrics.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul></div>"

        if alert.metadata:
            html += '<div class="metrics"><h3>Metadata:</h3><ul>'
            for key, value in alert.metadata.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul></div>"

        html += """
            </div>
            <div class="footer">
                <p>Credit Risk Scoring System | DVP-PRO Methodology</p>
                <p>Ing. Daniel Varela Perez | bedaniele0@gmail.com</p>
            </div>
        </body>
        </html>
        """

        return html


# ============================================================================
# WEBHOOK ALERTER
# ============================================================================


class WebhookAlerter(BaseAlerter):
    """Alerter genérico para webhooks HTTP."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.webhook_url = os.getenv("GENERIC_WEBHOOK_URL", self.config.get("webhook_url"))
        self.enabled = os.getenv("WEBHOOK_ENABLED", "false").lower() == "true"

    def send(self, alert: Alert) -> bool:
        """Envía alerta a webhook genérico."""
        if not self.enabled:
            logger.debug("Webhook alerts disabled")
            return False

        if not self.webhook_url:
            logger.error("Webhook URL not configured")
            return False

        try:
            # Payload
            payload = {
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp,
                "metrics": alert.metrics or {},
                "metadata": alert.metadata or {},
            }

            # Enviar
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("Webhook alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


# ============================================================================
# ALERT MANAGER
# ============================================================================


class AlertManager:
    """Gestor central de alertas multi-canal."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el Alert Manager.

        Args:
            config_path: Path a archivo de configuración YAML (opcional)
        """
        self.config = self._load_config(config_path)
        self.alerters = self._initialize_alerters()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Carga configuración desde YAML."""
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def _initialize_alerters(self) -> Dict[AlertChannel, BaseAlerter]:
        """Inicializa todos los alerters."""
        return {
            AlertChannel.SLACK: SlackAlerter(self.config.get("slack", {})),
            AlertChannel.TEAMS: TeamsAlerter(self.config.get("teams", {})),
            AlertChannel.EMAIL: EmailAlerter(self.config.get("email", {})),
            AlertChannel.WEBHOOK: WebhookAlerter(self.config.get("webhook", {})),
        }

    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        channels: Optional[List[AlertChannel]] = None,
        metrics: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[AlertChannel, bool]:
        """
        Envía alerta a uno o más canales.

        Args:
            title: Título de la alerta
            message: Mensaje de la alerta
            severity: Severidad (INFO, WARNING, ERROR, CRITICAL)
            channels: Canales a usar (None = todos los habilitados)
            metrics: Métricas a incluir
            metadata: Metadata adicional

        Returns:
            Diccionario con resultado por canal
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            metrics=metrics,
            metadata=metadata,
        )

        # Si no se especifican canales, usar todos los habilitados
        if channels is None:
            channels = [ch for ch, alerter in self.alerters.items() if alerter.enabled]

        results = {}
        for channel in channels:
            alerter = self.alerters.get(channel)
            if alerter:
                results[channel] = alerter.send(alert)
            else:
                logger.warning(f"Unknown channel: {channel}")
                results[channel] = False

        return results

    def send_drift_alert(
        self,
        feature_name: str,
        drift_score: float,
        threshold: float = 0.2,
    ) -> Dict[AlertChannel, bool]:
        """Envía alerta de drift detectado."""
        severity = AlertSeverity.WARNING if drift_score < 0.3 else AlertSeverity.CRITICAL

        return self.send_alert(
            title=f"Data Drift Detected: {feature_name}",
            message=f"Drift score ({drift_score:.4f}) exceeds threshold ({threshold:.4f}). "
            f"Consider retraining the model.",
            severity=severity,
            metrics={"drift_score": drift_score, "threshold": threshold, "feature": feature_name},
            metadata={"alert_type": "data_drift", "model": "credit-risk-model"},
        )

    def send_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        expected_value: float,
    ) -> Dict[AlertChannel, bool]:
        """Envía alerta de degradación de performance."""
        degradation = abs(current_value - expected_value) / expected_value * 100

        severity = AlertSeverity.WARNING if degradation < 20 else AlertSeverity.ERROR

        return self.send_alert(
            title=f"Model Performance Degradation: {metric_name}",
            message=f"Current {metric_name} ({current_value:.4f}) is {degradation:.1f}% "
            f"below expected ({expected_value:.4f}). Investigate immediately.",
            severity=severity,
            metrics={
                "metric": metric_name,
                "current_value": current_value,
                "expected_value": expected_value,
                "degradation_pct": f"{degradation:.2f}%",
            },
            metadata={"alert_type": "performance_degradation", "model": "credit-risk-model"},
        )

    def send_training_alert(
        self,
        status: str,
        metrics: Optional[Dict] = None,
    ) -> Dict[AlertChannel, bool]:
        """Envía alerta de training completado o fallido."""
        if status == "success":
            severity = AlertSeverity.INFO
            title = "Model Training Completed Successfully"
            message = "New model trained and ready for deployment."
        else:
            severity = AlertSeverity.ERROR
            title = "Model Training Failed"
            message = "Training pipeline encountered an error. Check logs for details."

        return self.send_alert(
            title=title,
            message=message,
            severity=severity,
            metrics=metrics or {},
            metadata={"alert_type": "training", "status": status},
        )


# ============================================================================
# MAIN (TESTING)
# ============================================================================


def main():
    """Test de alertas."""
    manager = AlertManager()

    # Test INFO alert
    print("Sending INFO alert...")
    manager.send_alert(
        title="Model Deployed Successfully",
        message="Credit risk model v1.0.0 deployed to production",
        severity=AlertSeverity.INFO,
        metrics={"auc_roc": 0.7813, "ks_statistic": 0.4251},
    )

    # Test WARNING alert
    print("Sending WARNING alert...")
    manager.send_drift_alert(feature_name="PAY_0", drift_score=0.25, threshold=0.2)

    # Test ERROR alert
    print("Sending ERROR alert...")
    manager.send_performance_alert(
        metric_name="AUC-ROC", current_value=0.72, expected_value=0.78
    )

    print("Alerts sent!")


if __name__ == "__main__":
    main()
