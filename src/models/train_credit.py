"""
============================================================================
train_credit.py - Training Pipeline para Credit Risk Scoring
============================================================================
Script principal de entrenamiento con MLflow tracking, calibración isotónica
y optimización de threshold basada en costos de negocio.

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Tel: +52 55 4189 3428
Metodología: DVP-PRO
============================================================================
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Imports internos
try:
    from models.mlflow_utils import MLflowTracker
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from models.mlflow_utils import MLflowTracker

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train_credit.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES DE THRESHOLD OPTIMIZATION
# ============================================================================


def calculate_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calcula el KS Statistic (Kolmogorov-Smirnov).

    El KS mide la separación máxima entre las distribuciones de buenos y malos.
    Valores > 0.40 son excelentes para credit scoring.

    Args:
        y_true: Labels verdaderos (0=no default, 1=default)
        y_proba: Probabilidades predichas

    Returns:
        KS statistic (0-1, mayor es mejor)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks = np.max(tpr - fpr)
    return ks


def optimize_threshold_business_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 1000,
    cost_fn: float = 10000,
) -> Tuple[float, Dict[str, float]]:
    """
    Optimiza el threshold basado en costos de negocio.

    Encuentra el threshold que minimiza:
        Total Cost = (FP * cost_fp) + (FN * cost_fn)

    Args:
        y_true: Labels verdaderos
        y_proba: Probabilidades predichas
        cost_fp: Costo de rechazar buen cliente (False Positive)
        cost_fn: Costo de aprobar mal cliente (False Negative)

    Returns:
        Tuple de (optimal_threshold, metrics_dict)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        costs.append(total_cost)

    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    # Calcular métricas con threshold óptimo
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()

    baseline_cost = y_true.sum() * cost_fn  # Asumir que todos son buenos
    optimal_cost = costs[optimal_idx]
    savings = baseline_cost - optimal_cost

    metrics = {
        "optimal_threshold": float(optimal_threshold),
        "optimal_cost": float(optimal_cost),
        "baseline_cost": float(baseline_cost),
        "expected_savings": float(savings),
        "savings_percentage": float((savings / baseline_cost) * 100),
        "fp_at_optimal": int(fp),
        "fn_at_optimal": int(fn),
        "tp_at_optimal": int(tp),
        "tn_at_optimal": int(tn),
    }

    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"Expected savings: ${savings:,.0f} MXN ({metrics['savings_percentage']:.2f}%)")

    return optimal_threshold, metrics


# ============================================================================
# FUNCIÓN PRINCIPAL DE TRAINING
# ============================================================================


def train_model(
    data_path: str,
    model_type: str = "lightgbm",
    calibration_method: str = "isotonic",
    cost_fp: float = 1000,
    cost_fn: float = 10000,
    config_path: Optional[str] = None,
) -> Tuple[object, Dict[str, float]]:
    """
    Entrena modelo de credit risk con calibración y tracking en MLflow.

    Args:
        data_path: Path al dataset procesado
        model_type: Tipo de modelo (lightgbm, random_forest, logistic_regression)
        calibration_method: Método de calibración (isotonic, sigmoid, none)
        cost_fp: Costo de False Positive (rechazar buen cliente)
        cost_fn: Costo de False Negative (aprobar mal cliente)
        config_path: Path a configuración de MLflow

    Returns:
        Tuple de (modelo_calibrado, metrics_dict)
    """
    logger.info("=" * 80)
    logger.info("CREDIT RISK SCORING - TRAINING PIPELINE")
    logger.info("Metodología: DVP-PRO")
    logger.info("=" * 80)

    # ==================== 1. LOAD CONFIG ====================
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # ==================== 2. LOAD DATA ====================
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Separar features y target
    target_col = "default.payment.next.month"
    # Alinear con diseño (F2) que usa `default_flag` post-ingesta.
    if target_col not in df.columns:
        if "default_flag" in df.columns:
            df[target_col] = df["default_flag"]
        else:
            raise ValueError(
                f"Target column '{target_col}' not found in dataset (se aceptan 'default.payment.next.month' o 'default_flag')"
            )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Default rate: {y.mean():.2%}")

    # Split train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    # ==================== 3. INITIALIZE MLFLOW ====================
    tracker = MLflowTracker(
        experiment_name=config.get("tracking", {}).get("experiment_name", "credit-risk-scoring"),
        tracking_uri=config.get("tracking", {}).get("uri", "file:./mlruns"),
    )

    run_name = f"{model_type}_calibrated_{calibration_method}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Log tags
        mlflow.set_tags(
            {
                "model_type": model_type,
                "calibration_method": calibration_method,
                "methodology": "DVP-PRO",
                "author": "Ing. Daniel Varela Perez",
            }
        )

        # ==================== 4. TRAIN BASE MODEL ====================
        logger.info(f"Training {model_type} model...")

        if model_type == "lightgbm":
            base_model = LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            params = base_model.get_params()

        elif model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            params = base_model.get_params()

        elif model_type == "logistic_regression":
            base_model = LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42, n_jobs=-1
            )
            params = base_model.get_params()

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("cost_false_positive", cost_fp)
        mlflow.log_param("cost_false_negative", cost_fn)

        # Train
        base_model.fit(X_train, y_train)
        logger.info("Base model trained successfully")

        # ==================== 5. CALIBRATION ====================
        if calibration_method != "none":
            logger.info(f"Applying {calibration_method} calibration...")
            calibrated_model = CalibratedClassifierCV(
                base_model, method=calibration_method, cv="prefit"
            )
            calibrated_model.fit(X_val, y_val)
            final_model = calibrated_model
            mlflow.log_param("calibration_applied", True)
        else:
            final_model = base_model
            mlflow.log_param("calibration_applied", False)

        # ==================== 6. PREDICTIONS ====================
        logger.info("Generating predictions...")

        # Train predictions
        y_train_proba = final_model.predict_proba(X_train)[:, 1]
        y_train_pred_default = (y_train_proba >= 0.5).astype(int)

        # Validation predictions
        y_val_proba = final_model.predict_proba(X_val)[:, 1]
        y_val_pred_default = (y_val_proba >= 0.5).astype(int)

        # Test predictions
        y_test_proba = final_model.predict_proba(X_test)[:, 1]
        y_test_pred_default = (y_test_proba >= 0.5).astype(int)

        # ==================== 7. THRESHOLD OPTIMIZATION ====================
        logger.info("Optimizing threshold based on business costs...")
        optimal_threshold, threshold_metrics = optimize_threshold_business_cost(
            y_val, y_val_proba, cost_fp, cost_fn
        )

        mlflow.log_metrics(threshold_metrics)

        # Predictions with optimal threshold
        y_val_pred_optimal = (y_val_proba >= optimal_threshold).astype(int)
        y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

        # ==================== 8. EVALUATE METRICS ====================
        logger.info("Calculating metrics...")

        # Métricas de clasificación (threshold default 0.5)
        train_metrics = tracker.log_credit_metrics(
            y_train, y_train_pred_default, y_train_proba, metric_prefix="train_"
        )
        val_metrics = tracker.log_credit_metrics(
            y_val, y_val_pred_default, y_val_proba, metric_prefix="val_"
        )
        test_metrics = tracker.log_credit_metrics(
            y_test, y_test_pred_default, y_test_proba, metric_prefix="test_"
        )

        # Métricas con threshold óptimo
        val_metrics_optimal = tracker.log_credit_metrics(
            y_val, y_val_pred_optimal, y_val_proba, metric_prefix="val_optimal_"
        )
        test_metrics_optimal = tracker.log_credit_metrics(
            y_test, y_test_pred_optimal, y_test_proba, metric_prefix="test_optimal_"
        )

        # Business metrics
        val_business = tracker.log_business_metrics(y_val, y_val_pred_optimal, cost_fp, cost_fn)
        test_business = tracker.log_business_metrics(y_test, y_test_pred_optimal, cost_fp, cost_fn)

        mlflow.log_metrics({**val_business, **test_business})

        # ==================== 9. VISUALIZATIONS ====================
        logger.info("Creating visualizations...")

        # ROC Curve
        tracker.log_roc_curve(y_test, y_test_proba)

        # Confusion Matrix (optimal threshold)
        tracker.log_confusion_matrix(y_test, y_test_pred_optimal)

        # Feature Importance
        if hasattr(base_model, "feature_importances_"):
            tracker.log_feature_importance(base_model, X.columns.tolist(), top_n=20)

        # ==================== 10. SAVE MODEL ====================
        logger.info("Saving model...")

        # Crear directorio de modelos
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save con joblib
        model_path = models_dir / "final_model.joblib"
        joblib.dump(final_model, model_path)
        logger.info(f"Model saved to: {model_path}")

        # Log model en MLflow
        mlflow.sklearn.log_model(
            final_model,
            "model",
            registered_model_name=config.get("registry", {}).get(
                "registered_model_name", "credit-risk-model"
            ),
        )

        # Log threshold óptimo como artifact
        threshold_artifact = {"optimal_threshold": optimal_threshold}
        mlflow.log_dict(threshold_artifact, "optimal_threshold.json")

        # ==================== 11. SUMMARY ====================
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Model Type: {model_type}")
        logger.info(f"Calibration: {calibration_method}")
        logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")
        logger.info(f"Test AUC-ROC: {test_metrics['test_auc_roc']:.4f}")
        logger.info(f"Test KS Statistic: {test_metrics['test_ks']:.4f}")
        logger.info(f"Test Recall: {test_metrics_optimal['test_optimal_recall']:.4f}")
        logger.info(
            f"Expected Savings: ${test_business['expected_savings']:,.0f} MXN "
            f"({test_business['savings_percentage']:.2f}%)"
        )
        logger.info("=" * 80)

        # Consolidar métricas
        all_metrics = {
            **train_metrics,
            **val_metrics,
            **test_metrics,
            **val_metrics_optimal,
            **test_metrics_optimal,
            **threshold_metrics,
            **val_business,
            **test_business,
        }

        return final_model, all_metrics


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Entry point para CLI."""
    parser = argparse.ArgumentParser(
        description="Credit Risk Scoring - Training Pipeline (DVP-PRO)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/UCI_Credit_Card_processed.csv",
        help="Path al dataset procesado",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "random_forest", "logistic_regression"],
        help="Tipo de modelo a entrenar",
    )

    parser.add_argument(
        "--calibration-method",
        type=str,
        default="isotonic",
        choices=["isotonic", "sigmoid", "none"],
        help="Método de calibración de probabilidades",
    )

    parser.add_argument(
        "--cost-fp",
        type=float,
        default=1000,
        help="Costo de False Positive (rechazar buen cliente) en MXN",
    )

    parser.add_argument(
        "--cost-fn",
        type=float,
        default=10000,
        help="Costo de False Negative (aprobar mal cliente) en MXN",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/mlflow_config.yaml",
        help="Path a configuración de MLflow",
    )

    args = parser.parse_args()

    # Crear directorio de logs
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Entrenar modelo
    try:
        model, metrics = train_model(
            data_path=args.data_path,
            model_type=args.model_type,
            calibration_method=args.calibration_method,
            cost_fp=args.cost_fp,
            cost_fn=args.cost_fn,
            config_path=args.config,
        )

        logger.info("Training pipeline completed successfully! ✓")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
