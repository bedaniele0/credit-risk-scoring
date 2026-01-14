"""
============================================================================
evaluate.py - Model Evaluation Pipeline para Credit Risk Scoring
============================================================================
Script completo de evaluación post-deployment con métricas de clasificación,
negocio, calibración, y generación de reportes detallados.

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Tel: +52 55 4189 3428
Metodología: DVP-PRO
============================================================================
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluate.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


# ============================================================================
# METRICS CALCULATION
# ============================================================================


def calculate_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calcula KS Statistic (Kolmogorov-Smirnov)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(tpr - fpr)
    return ks


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas completas de clasificación.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones binarias
        y_proba: Probabilidades predichas

    Returns:
        Diccionario con todas las métricas
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        # Métricas básicas
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        # Métricas de probabilidad
        "auc_roc": roc_auc_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "ks_statistic": calculate_ks_statistic(y_true, y_proba),
        # Confusion matrix
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        # Tasas
        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
    }

    return metrics


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 1000,
    cost_fn: float = 10000,
) -> Dict[str, float]:
    """
    Calcula métricas de negocio basadas en costos.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones binarias
        cost_fp: Costo de False Positive (rechazar buen cliente)
        cost_fn: Costo de False Negative (aprobar mal cliente)

    Returns:
        Diccionario con métricas de negocio
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Costos totales
    total_cost_fp = fp * cost_fp
    total_cost_fn = fn * cost_fn
    total_cost = total_cost_fp + total_cost_fn

    # Baseline: asumir que todos son buenos clientes (no rechazar a nadie)
    baseline_cost = y_true.sum() * cost_fn

    # Savings vs baseline
    savings = baseline_cost - total_cost
    savings_percentage = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

    metrics = {
        "total_cost": float(total_cost),
        "cost_false_positives": float(total_cost_fp),
        "cost_false_negatives": float(total_cost_fn),
        "baseline_cost": float(baseline_cost),
        "expected_savings": float(savings),
        "savings_percentage": float(savings_percentage),
        "roi": float(savings / baseline_cost) if baseline_cost > 0 else 0,
    }

    return metrics


def calculate_calibration_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Evalúa la calibración del modelo.

    Un modelo bien calibrado tiene probabilidades que coinciden con
    la frecuencia observada de eventos.

    Args:
        y_true: Labels verdaderos
        y_proba: Probabilidades predichas
        n_bins: Número de bins para calibration curve

    Returns:
        Métricas de calibración
    """
    # Expected Calibration Error (ECE)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    # ECE: promedio ponderado de diferencias absolutas
    bin_counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_proba)
    ece = np.sum(
        bin_weights[: len(fraction_of_positives)]
        * np.abs(fraction_of_positives - mean_predicted_value)
    )

    metrics = {
        "expected_calibration_error": float(ece),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }

    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, output_path: Optional[str] = None
) -> None:
    """Plot ROC curve con AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve - Credit Risk Model")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curve saved to: {output_path}")
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray, y_proba: np.ndarray, output_path: Optional[str] = None
) -> None:
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.axhline(y=y_true.mean(), color="navy", linestyle="--", label="Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Credit Risk Model")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"PR curve saved to: {output_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Optional[str] = None
) -> None:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
    )
    plt.title("Confusion Matrix - Credit Risk Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to: {output_path}")
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10, output_path: Optional[str] = None
) -> None:
    """Plot calibration curve."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        color="darkorange",
        label="Model",
        markersize=8,
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve - Credit Risk Model")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Calibration curve saved to: {output_path}")
    plt.close()


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 1000,
    cost_fn: float = 10000,
    output_path: Optional[str] = None,
) -> None:
    """Plot análisis de threshold vs métricas."""
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics_by_threshold = {
        "threshold": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "total_cost": [],
    }

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        total_cost = (fp * cost_fp) + (fn * cost_fn)

        metrics_by_threshold["threshold"].append(threshold)
        metrics_by_threshold["precision"].append(precision)
        metrics_by_threshold["recall"].append(recall)
        metrics_by_threshold["f1"].append(f1)
        metrics_by_threshold["total_cost"].append(total_cost)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Métricas de clasificación
    ax1.plot(
        metrics_by_threshold["threshold"], metrics_by_threshold["precision"], label="Precision"
    )
    ax1.plot(metrics_by_threshold["threshold"], metrics_by_threshold["recall"], label="Recall")
    ax1.plot(metrics_by_threshold["threshold"], metrics_by_threshold["f1"], label="F1 Score")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Classification Metrics vs Threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Costo total
    ax2.plot(metrics_by_threshold["threshold"], metrics_by_threshold["total_cost"], color="red")
    optimal_idx = np.argmin(metrics_by_threshold["total_cost"])
    optimal_threshold = metrics_by_threshold["threshold"][optimal_idx]
    ax2.axvline(optimal_threshold, color="green", linestyle="--", label=f"Optimal: {optimal_threshold:.3f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Total Cost (MXN)")
    ax2.set_title("Total Business Cost vs Threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Threshold analysis saved to: {output_path}")
    plt.close()


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================


def evaluate_model(
    predictions_path: str,
    actuals_path: Optional[str] = None,
    cost_fp: float = 1000,
    cost_fn: float = 10000,
    optimal_threshold: float = 0.12,
    output_dir: str = "reports/evaluation",
) -> Dict:
    """
    Pipeline completo de evaluación del modelo.

    Args:
        predictions_path: Path a archivo con predicciones
        actuals_path: Path a archivo con valores reales (opcional)
        cost_fp: Costo de False Positive
        cost_fn: Costo de False Negative
        optimal_threshold: Threshold óptimo para clasificación
        output_dir: Directorio de salida para reportes

    Returns:
        Diccionario con todas las métricas
    """
    logger.info("=" * 80)
    logger.info("CREDIT RISK SCORING - MODEL EVALUATION")
    logger.info("Metodología: DVP-PRO")
    logger.info("=" * 80)

    # Crear directorio de salida
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar predicciones
    logger.info(f"Loading predictions from: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)

    # Obtener probabilidades y predicciones
    if "default_probability" not in predictions_df.columns:
        raise ValueError("Predictions file must contain 'default_probability' column")

    y_proba = predictions_df["default_probability"].values

    # Obtener valores reales
    if actuals_path:
        logger.info(f"Loading actuals from: {actuals_path}")
        actuals_df = pd.read_csv(actuals_path)
        if "default.payment.next.month" in actuals_df.columns:
            y_true = actuals_df["default.payment.next.month"].values
        else:
            raise ValueError("Actuals file must contain 'default.payment.next.month' column")
    elif "default.payment.next.month" in predictions_df.columns:
        y_true = predictions_df["default.payment.next.month"].values
    else:
        raise ValueError("No actuals found. Provide actuals_path or include in predictions file")

    # Generar predicciones binarias con threshold óptimo
    y_pred = (y_proba >= optimal_threshold).astype(int)

    logger.info(f"Evaluating {len(y_true)} predictions...")
    logger.info(f"Using optimal threshold: {optimal_threshold}")

    # ==================== CALCULATE METRICS ====================

    # Métricas de clasificación
    classification_metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

    # Métricas de negocio
    business_metrics = calculate_business_metrics(y_true, y_pred, cost_fp, cost_fn)

    # Métricas de calibración
    calibration_metrics = calculate_calibration_metrics(y_true, y_proba)

    # ==================== GENERATE PLOTS ====================
    logger.info("Generating visualizations...")

    plot_roc_curve(y_true, y_proba, output_dir / "roc_curve.png")
    plot_precision_recall_curve(y_true, y_proba, output_dir / "precision_recall_curve.png")
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    plot_calibration_curve(y_true, y_proba, output_path=output_dir / "calibration_curve.png")
    plot_threshold_analysis(
        y_true, y_proba, cost_fp, cost_fn, output_path=output_dir / "threshold_analysis.png"
    )

    # ==================== CLASSIFICATION REPORT ====================
    class_report = classification_report(
        y_true, y_pred, target_names=["No Default", "Default"], output_dict=True
    )

    # ==================== CONSOLIDATE RESULTS ====================
    all_metrics = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optimal_threshold": optimal_threshold,
        "n_samples": len(y_true),
        "n_defaults": int(y_true.sum()),
        "default_rate": float(y_true.mean()),
        **classification_metrics,
        **business_metrics,
        **calibration_metrics,
        "classification_report": class_report,
    }

    # ==================== SAVE RESULTS ====================

    # Save metrics JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Save summary report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CREDIT RISK MODEL - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {all_metrics['evaluation_timestamp']}\n")
        f.write(f"Samples Evaluated: {all_metrics['n_samples']:,}\n")
        f.write(f"Default Rate: {all_metrics['default_rate']:.2%}\n")
        f.write(f"Optimal Threshold: {optimal_threshold}\n\n")

        f.write("CLASSIFICATION METRICS:\n")
        f.write(f"  - AUC-ROC:      {classification_metrics['auc_roc']:.4f}\n")
        f.write(f"  - KS Statistic: {classification_metrics['ks_statistic']:.4f}\n")
        f.write(f"  - Recall:       {classification_metrics['recall']:.4f}\n")
        f.write(f"  - Precision:    {classification_metrics['precision']:.4f}\n")
        f.write(f"  - F1 Score:     {classification_metrics['f1_score']:.4f}\n")
        f.write(f"  - Accuracy:     {classification_metrics['accuracy']:.4f}\n")
        f.write(f"  - Brier Score:  {classification_metrics['brier_score']:.4f}\n\n")

        f.write("BUSINESS METRICS:\n")
        f.write(f"  - Expected Savings:  ${business_metrics['expected_savings']:,.0f} MXN\n")
        f.write(f"  - Savings %:         {business_metrics['savings_percentage']:.2f}%\n")
        f.write(f"  - Total Cost:        ${business_metrics['total_cost']:,.0f} MXN\n")
        f.write(f"  - Baseline Cost:     ${business_metrics['baseline_cost']:,.0f} MXN\n")
        f.write(f"  - ROI:               {business_metrics['roi']:.2%}\n\n")

        f.write("CONFUSION MATRIX:\n")
        f.write(f"  - True Positives:  {classification_metrics['true_positives']:,}\n")
        f.write(f"  - True Negatives:  {classification_metrics['true_negatives']:,}\n")
        f.write(f"  - False Positives: {classification_metrics['false_positives']:,}\n")
        f.write(f"  - False Negatives: {classification_metrics['false_negatives']:,}\n\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Report saved to: {report_path}")

    # ==================== PRINT SUMMARY ====================
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"AUC-ROC: {classification_metrics['auc_roc']:.4f}")
    logger.info(f"KS Statistic: {classification_metrics['ks_statistic']:.4f}")
    logger.info(f"Recall: {classification_metrics['recall']:.4f}")
    logger.info(f"Precision: {classification_metrics['precision']:.4f}")
    logger.info(
        f"Expected Savings: ${business_metrics['expected_savings']:,.0f} MXN "
        f"({business_metrics['savings_percentage']:.2f}%)"
    )
    logger.info("=" * 80)

    return all_metrics


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Entry point para CLI."""
    parser = argparse.ArgumentParser(
        description="Credit Risk Scoring - Evaluation Pipeline (DVP-PRO)"
    )

    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="Path a archivo con predicciones",
    )

    parser.add_argument(
        "--actuals-path",
        type=str,
        default=None,
        help="Path a archivo con valores reales (opcional si incluido en predictions)",
    )

    parser.add_argument(
        "--cost-fp",
        type=float,
        default=1000,
        help="Costo de False Positive (MXN)",
    )

    parser.add_argument(
        "--cost-fn",
        type=float,
        default=10000,
        help="Costo de False Negative (MXN)",
    )

    parser.add_argument(
        "--optimal-threshold",
        type=float,
        default=0.12,
        help="Threshold óptimo para clasificación binaria",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation",
        help="Directorio de salida para reportes",
    )

    args = parser.parse_args()

    # Crear directorio de logs
    Path("logs").mkdir(parents=True, exist_ok=True)

    try:
        metrics = evaluate_model(
            predictions_path=args.predictions_path,
            actuals_path=args.actuals_path,
            cost_fp=args.cost_fp,
            cost_fn=args.cost_fn,
            optimal_threshold=args.optimal_threshold,
            output_dir=args.output_dir,
        )

        logger.info("Evaluation pipeline completed successfully! ✓")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
