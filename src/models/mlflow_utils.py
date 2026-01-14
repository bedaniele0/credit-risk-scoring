"""
============================================================================
mlflow_utils.py - Utilidades para MLflow Tracking (Credit Risk)
============================================================================
Funciones wrapper para integración de MLflow en el pipeline de credit scoring

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Tel: +52 55 4189 3428
Metodología: DVP-PRO
============================================================================
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix
)

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Clase para gestionar tracking de experimentos con MLflow para Credit Risk.

    Facilita el logging de parámetros, métricas, modelos y artifacts
    siguiendo mejores prácticas de MLOps para credit scoring.
    """

    def __init__(self,
                 experiment_name: str,
                 tracking_uri: Optional[str] = None,
                 artifact_location: Optional[str] = None):
        """
        Inicializa el tracker de MLflow.

        Args:
            experiment_name: Nombre del experimento
            tracking_uri: URI del servidor MLflow (default: ./mlruns)
            artifact_location: Ubicación para guardar artifacts
        """
        self.experiment_name = experiment_name

        # Configurar tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default: local mlruns directory
            mlflow.set_tracking_uri("file:./mlruns")

        # Crear o obtener experimento
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                logger.info(f"✅ Experimento '{experiment_name}' creado con ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"📊 Usando experimento existente: {experiment_name}")

            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id

        except Exception as e:
            logger.error(f"❌ Error configurando experimento: {e}")
            raise

    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Inicia un nuevo run de MLflow.

        Args:
            run_name: Nombre del run
            tags: Tags adicionales para el run

        Returns:
            Active MLflow run
        """
        run_tags = {
            "methodology": "DVP-PRO",
            "author": "Ing. Daniel Varela Perez",
            "project": "credit-risk-scoring",
            "model_type": "classification",
            "problem_type": "binary_classification"
        }

        if tags:
            run_tags.update(tags)

        return mlflow.start_run(run_name=run_name, tags=run_tags)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Registra parámetros del modelo.

        Args:
            params: Diccionario de parámetros
        """
        try:
            # Flatten nested params
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
            logger.info(f"✅ Parámetros registrados: {len(flat_params)} items")
        except Exception as e:
            logger.warning(f"⚠️ Error logging params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Registra métricas del modelo.

        Args:
            metrics: Diccionario de métricas
            step: Step/epoch para métricas evolutivas
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"✅ Métricas registradas: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ Error logging metrics: {e}")

    def log_credit_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_proba: np.ndarray, metric_prefix: str = "") -> Dict[str, float]:
        """
        Calcula y registra métricas específicas de credit risk.

        Args:
            y_true: Valores reales (0=no default, 1=default)
            y_pred: Predicciones binarias
            y_proba: Probabilidades de default
            metric_prefix: Prefijo para nombres de métricas

        Returns:
            Diccionario con métricas calculadas
        """
        try:
            metrics = {
                # Métricas de clasificación
                f"{metric_prefix}auc_roc": float(roc_auc_score(y_true, y_proba)),
                f"{metric_prefix}recall": float(recall_score(y_true, y_pred)),
                f"{metric_prefix}precision": float(precision_score(y_true, y_pred)),
                f"{metric_prefix}f1": float(f1_score(y_true, y_pred)),

                # Métricas de calibración
                f"{metric_prefix}brier": float(brier_score_loss(y_true, y_proba)),

                # KS Statistic (Kolmogorov-Smirnov)
                f"{metric_prefix}ks": float(self._calculate_ks(y_true, y_proba)),
            }

            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.update({
                f"{metric_prefix}true_negatives": int(tn),
                f"{metric_prefix}false_positives": int(fp),
                f"{metric_prefix}false_negatives": int(fn),
                f"{metric_prefix}true_positives": int(tp),
            })

            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f"{metric_prefix}specificity"] = float(specificity)

            self.log_metrics(metrics)
            return metrics

        except Exception as e:
            logger.error(f"❌ Error calculando métricas de credit risk: {e}")
            raise

    def _calculate_ks(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Calcula el KS Statistic (Kolmogorov-Smirnov).

        Args:
            y_true: Valores reales
            y_proba: Probabilidades predichas

        Returns:
            KS statistic
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            ks = np.max(tpr - fpr)
            return ks
        except Exception as e:
            logger.warning(f"⚠️ Error calculando KS: {e}")
            return 0.0

    def log_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            cost_fp: float = 1000, cost_fn: float = 10000) -> Dict[str, float]:
        """
        Calcula y registra métricas de negocio (costos/ahorros).

        Args:
            y_true: Valores reales
            y_pred: Predicciones binarias
            cost_fp: Costo de falso positivo (rechazar buen cliente)
            cost_fn: Costo de falso negativo (aprobar mal cliente)

        Returns:
            Métricas de negocio
        """
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Calcular costos
            total_cost_fp = fp * cost_fp
            total_cost_fn = fn * cost_fn
            total_cost = total_cost_fp + total_cost_fn

            # Calcular savings vs baseline (predecir todo como no-default)
            baseline_cost = y_true.sum() * cost_fn
            savings = baseline_cost - total_cost

            business_metrics = {
                'total_cost': float(total_cost),
                'cost_false_positives': float(total_cost_fp),
                'cost_false_negatives': float(total_cost_fn),
                'expected_savings': float(savings),
                'savings_percentage': float((savings / baseline_cost * 100) if baseline_cost > 0 else 0)
            }

            self.log_metrics(business_metrics)
            return business_metrics

        except Exception as e:
            logger.error(f"❌ Error calculando métricas de negocio: {e}")
            raise

    def log_model(self, model: Any, artifact_path: str = "model",
                  signature: Optional[Any] = None,
                  input_example: Optional[Any] = None,
                  registered_model_name: Optional[str] = None) -> None:
        """
        Registra modelo en MLflow.

        Args:
            model: Modelo entrenado
            artifact_path: Path dentro del artifact store
            signature: MLflow model signature
            input_example: Ejemplo de input
            registered_model_name: Nombre para model registry
        """
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
            logger.info(f"✅ Modelo registrado en: {artifact_path}")

            if registered_model_name:
                logger.info(f"📦 Modelo registrado en Model Registry: {registered_model_name}")

        except Exception as e:
            logger.error(f"❌ Error logging model: {e}")
            raise

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Registra artifact (archivo o directorio).

        Args:
            local_path: Path local del artifact
            artifact_path: Path dentro del artifact store
        """
        try:
            if Path(local_path).is_dir():
                mlflow.log_artifacts(local_path, artifact_path)
            else:
                mlflow.log_artifact(local_path, artifact_path)

            logger.info(f"✅ Artifact registrado: {local_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error logging artifact: {e}")

    def log_dict(self, dictionary: Dict, filename: str) -> None:
        """
        Registra diccionario como archivo YAML.

        Args:
            dictionary: Diccionario a guardar
            filename: Nombre del archivo
        """
        try:
            mlflow.log_dict(dictionary, filename)
            logger.info(f"✅ Diccionario guardado: {filename}")
        except Exception as e:
            logger.warning(f"⚠️ Error logging dict: {e}")

    def log_feature_importance(self, model: Any, feature_names: List[str],
                              top_n: int = 20) -> None:
        """
        Registra feature importance como artifact.

        Args:
            model: Modelo con feature_importances_
            feature_names: Nombres de features
            top_n: Top N features a visualizar
        """
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            # Obtener importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                logger.warning("⚠️ Modelo no tiene feature importance")
                return

            # Crear DataFrame
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Guardar CSV
            fi_path = "feature_importance.csv"
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)

            # Crear plot
            plt.figure(figsize=(10, 8))
            top_features = fi_df.head(top_n)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()

            plot_path = "feature_importance.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            mlflow.log_artifact(plot_path)

            # Cleanup
            os.remove(fi_path)
            os.remove(plot_path)

            logger.info("✅ Feature importance registrado")

        except Exception as e:
            logger.warning(f"⚠️ Error logging feature importance: {e}")

    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Registra confusion matrix como artifact.

        Args:
            y_true: Valores reales
            y_pred: Predicciones binarias
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Calcular confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            plot_path = "confusion_matrix.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

            logger.info("✅ Confusion matrix registrada")

        except Exception as e:
            logger.warning(f"⚠️ Error logging confusion matrix: {e}")

    def log_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        """
        Registra ROC curve como artifact.

        Args:
            y_true: Valores reales
            y_proba: Probabilidades predichas
        """
        try:
            import matplotlib.pyplot as plt

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = "roc_curve.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

            logger.info("✅ ROC curve registrada")

        except Exception as e:
            logger.warning(f"⚠️ Error logging ROC curve: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """
        Finaliza el run actual.

        Args:
            status: Estado del run (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            logger.info(f"✅ Run finalizado con status: {status}")
        except Exception as e:
            logger.warning(f"⚠️ Error ending run: {e}")

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        Aplana diccionario anidado.

        Args:
            d: Diccionario a aplanar
            parent_key: Key del padre (para recursión)
            sep: Separador para keys anidadas

        Returns:
            Diccionario plano
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convertir a string si no es número
                if not isinstance(v, (int, float, bool)):
                    v = str(v)
                items.append((new_key, v))

        return dict(items)

    @staticmethod
    def load_model(model_uri: str) -> Any:
        """
        Carga modelo desde MLflow.

        Args:
            model_uri: URI del modelo (runs:/..., models:/...)

        Returns:
            Modelo cargado
        """
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Modelo cargado desde: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    @staticmethod
    def get_best_run(experiment_name: str, metric: str = "auc_roc",
                     ascending: bool = False) -> Optional[str]:
        """
        Obtiene el mejor run de un experimento.

        Args:
            experiment_name: Nombre del experimento
            metric: Métrica para comparar
            ascending: True si menor es mejor

        Returns:
            Run ID del mejor run
        """
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)

            if experiment is None:
                logger.warning(f"⚠️ Experimento no encontrado: {experiment_name}")
                return None

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )

            if runs:
                best_run = runs[0]
                logger.info(f"✅ Mejor run: {best_run.info.run_id} ({metric}={best_run.data.metrics.get(metric)})")
                return best_run.info.run_id

            return None

        except Exception as e:
            logger.error(f"❌ Error getting best run: {e}")
            return None


def setup_mlflow_tracking(config_path: str = "config/mlflow_config.yaml") -> MLflowTracker:
    """
    Configura MLflow tracking desde archivo de configuración.

    Args:
        config_path: Path al archivo de configuración

    Returns:
        MLflowTracker configurado
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        tracking_uri = config.get('tracking_uri', 'file:./mlruns')
        experiment_name = config.get('experiment_name', 'credit-risk-scoring')
        artifact_location = config.get('artifact_location')

        tracker = MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location
        )

        logger.info("✅ MLflow tracker configurado desde archivo")
        return tracker

    except FileNotFoundError:
        logger.warning(f"⚠️ Config no encontrado: {config_path}, usando defaults")
        return MLflowTracker(experiment_name="credit-risk-scoring")
    except Exception as e:
        logger.error(f"❌ Error configurando MLflow: {e}")
        raise


if __name__ == "__main__":
    # Demo de uso
    import logging
    logging.basicConfig(level=logging.INFO)

    # Crear tracker
    tracker = MLflowTracker(experiment_name="test-credit-risk")

    # Iniciar run
    with tracker.start_run(run_name="test-run"):
        # Log params
        tracker.log_params({
            "model_type": "lightgbm",
            "n_estimators": 500,
            "learning_rate": 0.05
        })

        # Log metrics
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.2, 0.7, 0.85, 0.15])

        tracker.log_credit_metrics(y_true, y_pred, y_proba, metric_prefix="test_")
        tracker.log_business_metrics(y_true, y_pred)

        print("✅ Demo MLflow completado")
