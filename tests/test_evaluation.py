"""Unit tests for ML evaluation and metrics logging library (src/models/evaluation.py)."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from src.models.evaluation import (
    calculate_metrics,
    evaluate_by_regime,
    evaluate_per_symbol,
    log_metrics,
    save_metrics_json,
)


def test_calculate_metrics() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=int)
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3, 0.85, 0.6], dtype=float)

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "balanced_accuracy" in metrics
    assert "specificity" in metrics
    assert "npv" in metrics
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "log_loss" in metrics

    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 3
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 3

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_log_metrics() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.2, 0.8, 0.1, 0.4])
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    # Calling log_metrics should not raise any exceptions
    log_metrics(metrics, model_name="TestModel")


def test_evaluate_per_symbol() -> None:
    df = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
            "volatility_30": [0.01, 0.02, 0.05, 0.06],
        }
    )
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.7])

    per_symbol = evaluate_per_symbol(df, y_true, y_pred, y_prob, symbol_col="symbol")

    assert "BTCUSDT" in per_symbol
    assert "ETHUSDT" in per_symbol
    assert per_symbol["BTCUSDT"]["count"] == 2
    assert per_symbol["ETHUSDT"]["count"] == 2
    assert per_symbol["BTCUSDT"]["accuracy"] == 1.0


def test_evaluate_by_regime() -> None:
    df = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 6,
            "volatility_30": [0.01, 0.02, 0.05, 0.06, 0.10, 0.12],
        }
    )
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3])

    regimes = evaluate_by_regime(df, y_true, y_pred, y_prob, regime_col="volatility_30", num_bins=3)

    assert "Low" in regimes
    assert "Medium" in regimes
    assert "High" in regimes
    assert regimes["Low"]["count"] == 2
    assert regimes["Medium"]["count"] == 2
    assert regimes["High"]["count"] == 2


def test_save_metrics_json(tmp_path: Path) -> None:
    metrics = {
        "model_name": "TestModel",
        "int_val": np.int64(42),
        "float_val": np.float64(0.95),
        "nested": {"arr": np.array([0.1, 0.2])},
    }
    json_file = tmp_path / "metrics.json"
    save_metrics_json(metrics, json_file)

    assert json_file.exists()
    with open(json_file, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["int_val"] == 42
    assert loaded["float_val"] == 0.95
    assert loaded["nested"]["arr"] == [0.1, 0.2]
