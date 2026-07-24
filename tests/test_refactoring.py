"""
Unit tests verifying codebase refactoring additions:
- Seed reproducibility across PyTorch, NumPy, and scikit-learn models
- Polars LazyFrame calculation parity with DataFrame in indicators module
- DataValidationError raising on insufficient sequence lengths
- CLI argument parsing and execution
"""

import sys

import numpy as np
import polars as pl
import pytest

from src.exceptions import DataValidationError
from src.features.indicators import compute_indicators
from src.models.lstm import SequenceDataset
from src.utils.seed import get_provenance_metadata, set_seed


def test_seed_reproducibility():
    """Verifies that set_seed produces identical random arrays and model outputs."""
    set_seed(42)
    rand1 = np.random.randn(100)

    set_seed(42)
    rand2 = np.random.randn(100)

    np.testing.assert_array_equal(rand1, rand2)

    import torch

    set_seed(42)
    t1 = torch.randn(50, 50)

    set_seed(42)
    t2 = torch.randn(50, 50)

    assert torch.equal(t1, t2)


def test_provenance_metadata():
    """Verifies provenance metadata content and formatting."""
    meta = get_provenance_metadata(seed=123)
    assert meta["random_seed"] == 123
    assert "python_version" in meta
    assert "timestamp_utc" in meta
    assert "polars" in meta["packages"]


def test_lazyframe_indicators_parity():
    """Verifies that compute_indicators produces identical results for DataFrame and LazyFrame."""
    data = {
        "symbol": ["BTCUSDT"] * 60,
        "open_time": list(range(60)),
        "open": [100.0 + i for i in range(60)],
        "high": [105.0 + i for i in range(60)],
        "low": [95.0 + i for i in range(60)],
        "close": [102.0 + i for i in range(60)],
        "volume": [1000.0] * 60,
    }
    df = pl.DataFrame(data)
    lf = df.lazy()

    df_res = compute_indicators(df)
    lf_res = compute_indicators(lf).collect()

    assert df_res.shape == lf_res.shape
    assert df_res.columns == lf_res.columns
    np.testing.assert_allclose(
        df_res["sma_15"].drop_nulls().to_numpy(),
        lf_res["sma_15"].drop_nulls().to_numpy(),
    )


def test_sequence_dataset_validation():
    """Verifies that SequenceDataset raises DataValidationError for short inputs."""
    data = {
        "symbol": ["BTCUSDT"] * 10,
        "open_time": list(range(10)),
        "open": [100.0] * 10,
        "high": [105.0] * 10,
        "low": [95.0] * 10,
        "close": [102.0] * 10,
        "volume": [1000.0] * 10,
        "close_to_sma_15": [0.01] * 10,
        "close_to_sma_50": [0.02] * 10,
        "close_to_ema_15": [0.01] * 10,
        "close_to_ema_50": [0.02] * 10,
        "bb_position": [0.5] * 10,
        "macd_line_norm": [0.001] * 10,
        "macd_signal_norm": [0.001] * 10,
        "macd_hist_norm": [0.0] * 10,
        "volatility_30": [0.01] * 10,
        "rsi_14": [50.0] * 10,
        "log_return": [0.001] * 10,
        "taker_buy_ratio": [0.5] * 10,
        "volume_z30": [0.0] * 10,
        "trades_z30": [0.0] * 10,
        "hour_sin": [0.0] * 10,
        "hour_cos": [1.0] * 10,
        "target": [1] * 10,
    }
    df = pl.DataFrame(data)

    with pytest.raises(DataValidationError, match="less than sequence length 60"):
        SequenceDataset(df, seq_len=60)


def test_generate_evaluation_report():
    """Verifies multi-dimensional evaluation report generation."""
    from src.models.evaluation import generate_evaluation_report

    data = {
        "symbol": ["BTCUSDT"] * 50 + ["ETHUSDT"] * 50,
        "volatility_30": [0.01 + i * 0.001 for i in range(100)],
    }
    df = pl.DataFrame(data)
    y_true = np.array([0, 1] * 50)
    y_pred = np.array([0, 1] * 50)
    y_prob = np.array([0.2, 0.8] * 50)

    report = generate_evaluation_report(df, y_true, y_pred, y_prob, model_name="Test Model")

    assert report["model_name"] == "Test Model"
    assert "overall" in report
    assert report["overall"]["accuracy"] == 1.0
    assert "per_symbol" in report
    assert "BTCUSDT" in report["per_symbol"]
    assert "ETHUSDT" in report["per_symbol"]
    assert "volatility_regimes" in report
    assert "Low" in report["volatility_regimes"]


def test_cli_subcommands(monkeypatch):
    """Verifies that all CLI subcommands parse arguments cleanly."""
    from src.cli import main

    for cmd in ["train-sklearn", "train-lstm", "train-spark", "evaluate"]:
        monkeypatch.setattr(sys, "argv", ["src.cli", cmd, "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
