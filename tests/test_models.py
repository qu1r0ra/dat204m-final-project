from datetime import datetime

import numpy as np

from src.models.evaluation import calculate_metrics
from src.models.train import split_data_chronologically


def test_calculate_metrics():
    """Verify metric values on known y_true/y_pred arrays, verify optional y_prob behavior."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])

    metrics = calculate_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" not in metrics

    assert metrics["accuracy"] == 0.8  # 4 correct out of 5

    metrics_with_prob = calculate_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" in metrics_with_prob


def test_split_data_chronologically(sample_ohlcv_df):
    """Verify chronological split logic and empty partition handling."""
    # Data is from 2024-01-01 00:00:00 to 2024-01-01 01:39:00 (100 minutes)
    train_end = "2024-01-01 00:30:00"
    val_end = "2024-01-01 01:00:00"

    train_df, val_df, test_df = split_data_chronologically(sample_ohlcv_df, train_end, val_end)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    train_end_dt = datetime.strptime(train_end, "%Y-%m-%d %H:%M:%S")
    val_end_dt = datetime.strptime(val_end, "%Y-%m-%d %H:%M:%S")

    assert (train_df["open_time"] < train_end_dt).all()
    assert ((val_df["open_time"] >= train_end_dt) & (val_df["open_time"] < val_end_dt)).all()
    assert (test_df["open_time"] >= val_end_dt).all()

    # Verify empty partition handling
    _, _, test_empty = split_data_chronologically(sample_ohlcv_df, train_end, "2025-01-01 00:00:00")
    assert len(test_empty) == 0
