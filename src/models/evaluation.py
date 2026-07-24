"""
Machine learning model evaluation library.

Provides helper functions to calculate accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
balanced accuracy, log loss, confusion matrix metrics, per-symbol performance, volatility
regime breakdowns, and metric export utilities.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict[str, Any]:
    """Calculates comprehensive classification metrics for ML model evaluation."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "specificity": specificity,
        "npv": npv,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        if len(np.unique(y_true_arr)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
            metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_prob_arr))
            y_prob_clipped = np.clip(y_prob_arr, 1e-15, 1 - 1e-15)
            metrics["log_loss"] = float(log_loss(y_true_arr, y_prob_clipped))
        else:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = 0.5
            metrics["log_loss"] = 0.0

    return metrics


def log_metrics(
    metrics: dict[str, Any],
    model_name: str = "Model",
    logger_instance: logging.Logger | None = None,
) -> None:
    """Logs evaluation metrics in a clean human-readable format."""
    log = logger_instance or logger
    log.info(f"=== Evaluation Metrics for {model_name} ===")
    log.info(
        f"Acc: {metrics.get('accuracy', 0.0):.4f} | "
        f"Prec: {metrics.get('precision', 0.0):.4f} | "
        f"Rec: {metrics.get('recall', 0.0):.4f} | "
        f"F1: {metrics.get('f1', 0.0):.4f} | "
        f"Bal Acc: {metrics.get('balanced_accuracy', 0.0):.4f}"
    )
    if "roc_auc" in metrics:
        log.info(
            f"ROC-AUC: {metrics.get('roc_auc', 0.0):.4f} | "
            f"PR-AUC: {metrics.get('pr_auc', 0.0):.4f} | "
            f"Log Loss: {metrics.get('log_loss', 0.0):.4f}"
        )
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        log.info(
            f"Confusion Matrix: TN={cm.get('tn')}, FP={cm.get('fp')}, FN={cm.get('fn')}, TP={cm.get('tp')}"
        )


def evaluate_per_symbol(
    df: pl.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    symbol_col: str = "symbol",
) -> dict[str, dict[str, Any]]:
    """Evaluates classification performance per individual asset symbol."""
    symbols = df[symbol_col].to_numpy()
    unique_symbols = np.unique(symbols)
    per_symbol_metrics: dict[str, dict[str, Any]] = {}

    for sym in unique_symbols:
        mask = symbols == sym
        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_pr = y_prob[mask] if y_prob is not None else None

        if len(y_t) > 0:
            sym_metrics = calculate_metrics(y_t, y_p, y_pr)
            sym_metrics["count"] = int(len(y_t))
            per_symbol_metrics[str(sym)] = sym_metrics

    return per_symbol_metrics


def evaluate_by_regime(
    df: pl.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    regime_col: str = "volatility_30",
    num_bins: int = 3,
) -> dict[str, dict[str, Any]]:
    """Evaluates classification performance split by volatility regimes (Low, Medium, High)."""
    if regime_col not in df.columns:
        logger.warning(f"Regime column '{regime_col}' not found in DataFrame.")
        return {}

    vol_vals = df[regime_col].to_numpy()
    quantiles = np.quantile(vol_vals, np.linspace(0, 1, num_bins + 1))
    regimes = (
        ["Low", "Medium", "High"]
        if num_bins == 3
        else [f"Bin_{i+1}" for i in range(num_bins)]
    )

    regime_metrics: dict[str, dict[str, Any]] = {}

    for i in range(num_bins):
        q_low = quantiles[i]
        q_high = quantiles[i + 1]

        if i == num_bins - 1:
            mask = (vol_vals >= q_low) & (vol_vals <= q_high)
        else:
            mask = (vol_vals >= q_low) & (vol_vals < q_high)

        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_pr = y_prob[mask] if y_prob is not None else None

        if len(y_t) > 0:
            bin_m = calculate_metrics(y_t, y_p, y_pr)
            bin_m["count"] = int(len(y_t))
            bin_m["min_val"] = float(q_low)
            bin_m["max_val"] = float(q_high)
            regime_name = regimes[i]
            regime_metrics[regime_name] = bin_m

    return regime_metrics


def _convert_numpy_types(obj: Any) -> Any:
    """Helper function to convert NumPy scalars/arrays to standard Python types for JSON serialization."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return [_convert_numpy_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    return obj


def save_metrics_json(metrics: dict[str, Any], filepath: Path | str) -> None:
    """Serializes metric dictionaries to JSON file cleanly."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_metrics = _convert_numpy_types(metrics)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics JSON to {path}")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"
) -> plt.Figure:
    """Generates a confusion matrix plot. Returns the matplotlib Figure object."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["DOWN/STABLE", "UP"],
        yticklabels=["DOWN/STABLE", "UP"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_curves(model_probs: dict, y_true: np.ndarray) -> plt.Figure:
    """Plots comparative ROC curves for multiple models.

    'model_probs' is a dictionary mapping model names to their prediction probabilities on the test set.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC = 0.50)")

    for model_name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_score = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
