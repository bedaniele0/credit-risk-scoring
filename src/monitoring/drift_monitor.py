"""
Credit Risk Model - Drift Monitoring

Proyecto: Credit Risk Scoring - UCI Taiwan Dataset
Fase DVP-PRO: F8 - Productización
Autor: Ing. Daniel Varela Pérez
Email: bedaniele0@gmail.com
Tel: +52 55 4189 3428
Fecha: 2025-11-18

Este script monitorea:
- PSI (Population Stability Index) para detectar drift en distribución
- KS (Kolmogorov-Smirnov) para detectar cambios en scores
- Métricas de rendimiento del modelo
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MONITORING_DIR = REPORTS_DIR / "monitoring"

# Crear directorio de monitoreo si no existe
MONITORING_DIR.mkdir(parents=True, exist_ok=True)


class DriftMonitor:
    """
    Clase para monitorear drift en el modelo de credit scoring.

    Métricas principales:
    - PSI: Population Stability Index
    - KS: Kolmogorov-Smirnov Statistic
    - CSI: Characteristic Stability Index (por feature)
    """

    # Thresholds para alertas
    PSI_THRESHOLD_WARNING = 0.10
    PSI_THRESHOLD_CRITICAL = 0.25
    KS_DECAY_THRESHOLD = 0.10  # 10% de decaimiento en KS

    def __init__(self, reference_data: pd.DataFrame, reference_scores: np.ndarray):
        """
        Inicializa el monitor con datos de referencia (training/validation).

        Args:
            reference_data: DataFrame con features de referencia
            reference_scores: Array con scores de referencia
        """
        self.reference_data = reference_data
        self.reference_scores = reference_scores
        self.baseline_ks = self._calculate_baseline_ks()

        logger.info(f"DriftMonitor inicializado con {len(reference_data)} muestras de referencia")
        logger.info(f"Baseline KS: {self.baseline_ks:.4f}")

    def _calculate_baseline_ks(self) -> float:
        """Calcula KS baseline de los scores de referencia."""
        # Asumiendo distribución 50-50 para simplificar
        # En producción, usar las etiquetas reales
        median_score = np.median(self.reference_scores)
        group1 = self.reference_scores[self.reference_scores <= median_score]
        group2 = self.reference_scores[self.reference_scores > median_score]

        ks_stat, _ = stats.ks_2samp(group1, group2)
        return ks_stat

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calcula el Population Stability Index (PSI).

        PSI < 0.10: No hay cambio significativo
        0.10 <= PSI < 0.25: Cambio moderado, investigar
        PSI >= 0.25: Cambio significativo, reentrenar

        Args:
            expected: Distribución esperada (referencia)
            actual: Distribución actual (producción)
            n_bins: Número de bins para discretizar

        Returns:
            Valor de PSI
        """
        # Convertir a float para evitar problemas con booleanos
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        # Crear bins basados en la distribución esperada
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calcular proporciones
        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]

        # Evitar división por cero
        expected_pct = (expected_counts + 0.0001) / len(expected)
        actual_pct = (actual_counts + 0.0001) / len(actual)

        # Calcular PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return psi

    def calculate_ks_statistic(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Calcula el estadístico KS (Kolmogorov-Smirnov).

        Args:
            scores: Scores predichos
            labels: Etiquetas reales (0/1)

        Returns:
            Estadístico KS
        """
        # Separar scores por clase
        scores_good = scores[labels == 0]
        scores_bad = scores[labels == 1]

        # Calcular KS
        ks_stat, _ = stats.ks_2samp(scores_good, scores_bad)

        return ks_stat

    def calculate_csi(
        self,
        feature_name: str,
        actual_data: pd.Series,
        n_bins: int = 10
    ) -> float:
        """
        Calcula el Characteristic Stability Index (CSI) para una feature.

        Similar a PSI pero para features individuales.

        Args:
            feature_name: Nombre de la feature
            actual_data: Datos actuales de la feature
            n_bins: Número de bins

        Returns:
            Valor de CSI
        """
        if feature_name not in self.reference_data.columns:
            logger.warning(f"Feature {feature_name} no encontrada en datos de referencia")
            return 0.0

        expected = self.reference_data[feature_name].values
        actual = actual_data.values

        return self.calculate_psi(expected, actual, n_bins)

    def monitor_scores(
        self,
        production_scores: np.ndarray,
        production_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Monitorea drift en los scores de producción.

        Args:
            production_scores: Scores de producción
            production_labels: Etiquetas reales (opcional, para KS)

        Returns:
            Diccionario con métricas de drift
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(production_scores),
            "metrics": {}
        }

        # 1. PSI de scores
        psi = self.calculate_psi(self.reference_scores, production_scores)
        results["metrics"]["psi"] = {
            "value": round(psi, 4),
            "threshold_warning": self.PSI_THRESHOLD_WARNING,
            "threshold_critical": self.PSI_THRESHOLD_CRITICAL,
            "status": self._get_psi_status(psi)
        }

        # 2. Estadísticas básicas de scores
        results["metrics"]["score_stats"] = {
            "mean_reference": round(float(np.mean(self.reference_scores)), 4),
            "mean_production": round(float(np.mean(production_scores)), 4),
            "std_reference": round(float(np.std(self.reference_scores)), 4),
            "std_production": round(float(np.std(production_scores)), 4),
            "median_reference": round(float(np.median(self.reference_scores)), 4),
            "median_production": round(float(np.median(production_scores)), 4)
        }

        # 3. KS decay (si hay etiquetas)
        if production_labels is not None:
            production_ks = self.calculate_ks_statistic(production_scores, production_labels)
            ks_decay = (self.baseline_ks - production_ks) / self.baseline_ks if self.baseline_ks > 0 else 0

            results["metrics"]["ks"] = {
                "baseline": round(self.baseline_ks, 4),
                "current": round(production_ks, 4),
                "decay": round(ks_decay, 4),
                "decay_threshold": self.KS_DECAY_THRESHOLD,
                "status": "ALERT" if ks_decay > self.KS_DECAY_THRESHOLD else "OK"
            }

        # 4. Alertas generales
        results["alerts"] = self._generate_alerts(results["metrics"])

        return results

    def monitor_features(
        self,
        production_data: pd.DataFrame,
        features_to_monitor: Optional[List[str]] = None
    ) -> Dict:
        """
        Monitorea drift en las features.

        Args:
            production_data: DataFrame con datos de producción
            features_to_monitor: Lista de features a monitorear (todas si None)

        Returns:
            Diccionario con CSI por feature
        """
        if features_to_monitor is None:
            features_to_monitor = list(self.reference_data.columns)

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(production_data),
            "features": {}
        }

        for feature in features_to_monitor:
            if feature in production_data.columns:
                csi = self.calculate_csi(feature, production_data[feature])
                results["features"][feature] = {
                    "csi": round(csi, 4),
                    "status": self._get_psi_status(csi)
                }

        # Features con drift significativo
        drifted_features = [
            f for f, v in results["features"].items()
            if v["status"] in ["WARNING", "CRITICAL"]
        ]
        results["drifted_features"] = drifted_features

        return results

    def _get_psi_status(self, psi: float) -> str:
        """Determina el status basado en PSI."""
        if psi < self.PSI_THRESHOLD_WARNING:
            return "OK"
        elif psi < self.PSI_THRESHOLD_CRITICAL:
            return "WARNING"
        else:
            return "CRITICAL"

    def _generate_alerts(self, metrics: Dict) -> List[str]:
        """Genera lista de alertas basadas en métricas."""
        alerts = []

        # PSI alert
        if "psi" in metrics:
            psi_status = metrics["psi"]["status"]
            if psi_status == "WARNING":
                alerts.append(f"PSI moderado ({metrics['psi']['value']:.3f}): Investigar distribución de scores")
            elif psi_status == "CRITICAL":
                alerts.append(f"PSI crítico ({metrics['psi']['value']:.3f}): Considerar reentrenamiento")

        # KS decay alert
        if "ks" in metrics:
            if metrics["ks"]["status"] == "ALERT":
                alerts.append(f"KS decay significativo ({metrics['ks']['decay']:.1%}): Rendimiento degradado")

        return alerts

    def generate_report(
        self,
        production_scores: np.ndarray,
        production_data: pd.DataFrame,
        production_labels: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Genera un reporte completo de monitoreo.

        Args:
            production_scores: Scores de producción
            production_data: Features de producción
            production_labels: Etiquetas reales (opcional)
            save_path: Ruta para guardar el reporte

        Returns:
            Reporte completo
        """
        report = {
            "report_type": "drift_monitoring",
            "generated_at": datetime.now().isoformat(),
            "reference_period": "training_data",
            "production_period": datetime.now().strftime("%Y-%m"),
        }

        # Monitoreo de scores
        scores_monitoring = self.monitor_scores(production_scores, production_labels)
        report["scores"] = scores_monitoring

        # Monitoreo de features
        features_monitoring = self.monitor_features(production_data)
        report["features"] = features_monitoring

        # Resumen ejecutivo
        report["summary"] = {
            "overall_status": self._get_overall_status(scores_monitoring, features_monitoring),
            "total_alerts": len(scores_monitoring.get("alerts", [])),
            "drifted_features_count": len(features_monitoring.get("drifted_features", [])),
            "recommendation": self._get_recommendation(scores_monitoring, features_monitoring)
        }

        # Guardar reporte
        if save_path is None:
            save_path = MONITORING_DIR / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Reporte de drift guardado en: {save_path}")

        return report

    def _get_overall_status(self, scores_result: Dict, features_result: Dict) -> str:
        """Determina el status general del modelo."""
        psi_status = scores_result.get("metrics", {}).get("psi", {}).get("status", "OK")
        ks_status = scores_result.get("metrics", {}).get("ks", {}).get("status", "OK")
        n_drifted = len(features_result.get("drifted_features", []))

        if psi_status == "CRITICAL" or ks_status == "ALERT":
            return "CRITICAL"
        elif psi_status == "WARNING" or n_drifted > 3:
            return "WARNING"
        else:
            return "HEALTHY"

    def _get_recommendation(self, scores_result: Dict, features_result: Dict) -> str:
        """Genera recomendación basada en resultados."""
        status = self._get_overall_status(scores_result, features_result)

        if status == "CRITICAL":
            return "Reentrenamiento urgente recomendado. Considerar rollback a versión anterior."
        elif status == "WARNING":
            return "Investigar causas del drift. Preparar datos para potencial reentrenamiento."
        else:
            return "Modelo estable. Continuar monitoreo regular."


# =====================================================
# FUNCIONES DE UTILIDAD
# =====================================================

def load_reference_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Carga datos de referencia (training) para monitoreo.

    Returns:
        Tuple con (features DataFrame, scores array)
    """
    try:
        X_train = pd.read_csv(DATA_DIR / "processed" / "X_train.csv")

        # Cargar modelo para generar scores de referencia
        import joblib
        model = joblib.load(BASE_DIR / "models" / "final_model.joblib")
        reference_scores = model.predict_proba(X_train)[:, 1]

        return X_train, reference_scores

    except Exception as e:
        logger.error(f"Error cargando datos de referencia: {e}")
        raise


def run_monitoring_check(
    production_data_path: Optional[str] = None,
    production_labels_path: Optional[str] = None
) -> Dict:
    """
    Ejecuta un check de monitoreo.

    Args:
        production_data_path: Ruta a datos de producción (CSV)
        production_labels_path: Ruta a etiquetas de producción (CSV)

    Returns:
        Reporte de monitoreo
    """
    # Cargar datos de referencia
    reference_data, reference_scores = load_reference_data()

    # Inicializar monitor
    monitor = DriftMonitor(reference_data, reference_scores)

    # Cargar datos de producción (o simular con test data)
    if production_data_path:
        production_data = pd.read_csv(production_data_path)
    else:
        # Usar test data como simulación
        production_data = pd.read_csv(DATA_DIR / "processed" / "X_test.csv")
        logger.info("Usando datos de test como simulación de producción")

    # Cargar etiquetas si disponibles
    production_labels = None
    if production_labels_path:
        production_labels = pd.read_csv(production_labels_path).values.ravel()

    # Generar scores de producción
    import joblib
    model = joblib.load(BASE_DIR / "models" / "final_model.joblib")
    production_scores = model.predict_proba(production_data)[:, 1]

    # Generar reporte
    report = monitor.generate_report(
        production_scores,
        production_data,
        production_labels
    )

    return report


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("="*60)
    print("  CREDIT RISK MODEL - DRIFT MONITORING")
    print("="*60)

    try:
        report = run_monitoring_check()

        print(f"\nStatus General: {report['summary']['overall_status']}")
        print(f"Alertas: {report['summary']['total_alerts']}")
        print(f"Features con drift: {report['summary']['drifted_features_count']}")
        print(f"\nRecomendación: {report['summary']['recommendation']}")

        if report['scores'].get('alerts'):
            print("\nAlertas:")
            for alert in report['scores']['alerts']:
                print(f"  - {alert}")

        print(f"\nReporte guardado en: {MONITORING_DIR}")

    except Exception as e:
        logger.error(f"Error en monitoreo: {e}")
        raise
