"""
Machine learning model evaluation library.

Provides helper functions to calculate accuracy, precision, recall, F1, ROC-AUC,
and generate plots for confusion matrices and ROC curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """Calculates standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


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
