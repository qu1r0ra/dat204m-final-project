"""Unit tests for PyTorch LSTM sequence classifier module."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from src.models.lstm import (
    LSTMClassifier,
    SequenceDataset,
    load_lstm_artifacts,
    predict_lstm,
    save_lstm_artifacts,
    train_lstm,
)


@pytest.fixture
def multi_symbol_synthetic_df():
    """Provides a synthetic multi-symbol Polars DataFrame with features and targets."""
    np.random.seed(42)
    base_time = datetime(2024, 1, 1)

    dfs = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        times = [base_time + timedelta(minutes=i) for i in range(100)]
        f1 = np.random.normal(0, 1, 100)
        f2 = np.random.normal(0, 1, 100)
        f3 = np.random.normal(0, 1, 100)
        f4 = np.random.normal(0, 1, 100)
        target = np.random.choice([0.0, 1.0], size=100)

        df_sym = pl.DataFrame(
            {
                "symbol": [symbol] * 100,
                "open_time": times,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "f4": f4,
                "target": target,
            }
        )
        dfs.append(df_sym)

    return pl.concat(dfs)


def test_sequence_dataset_no_cross_symbol(multi_symbol_synthetic_df):
    """Verifies that SequenceDataset windows never cross symbol boundaries."""
    feature_cols = ["f1", "f2", "f3", "f4"]
    seq_len = 10

    dataset = SequenceDataset(
        multi_symbol_synthetic_df,
        feature_cols=feature_cols,
        target_col="target",
        seq_len=seq_len,
    )

    # For 2 symbols with 100 rows each and seq_len=10,
    # each symbol yields (100 - 10 + 1) = 91 valid windows -> total 182 windows.
    assert len(dataset) == 182

    # Spot check window end indices
    symbols = multi_symbol_synthetic_df.sort(["symbol", "open_time"])["symbol"].to_numpy()
    for idx in range(len(dataset)):
        end_idx = dataset.valid_end_indices[idx]
        start_idx = end_idx - seq_len + 1
        # The symbol at start_idx must equal the symbol at end_idx
        assert symbols[start_idx] == symbols[end_idx]


def test_lstm_forward_pass_dimensions():
    """Verifies LSTMClassifier forward pass output shapes and tensor properties."""
    batch_size = 8
    seq_len = 10
    input_size = 4
    hidden_size = 16

    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    x_dummy = torch.randn(batch_size, seq_len, input_size)

    logits = model(x_dummy)
    assert logits.shape == (batch_size,)
    assert torch.isfinite(logits).all()


def test_lstm_training_loop_convergence(multi_symbol_synthetic_df, tmp_path):
    """Verifies train_lstm execution, threshold tuning, and artifact serialization."""
    feature_cols = ["f1", "f2", "f3", "f4"]
    seq_len = 10

    # Split synthetic data into train and val
    train_df = multi_symbol_synthetic_df.filter(pl.col("open_time") < datetime(2024, 1, 1, 1, 10))
    val_df = multi_symbol_synthetic_df.filter(pl.col("open_time") >= datetime(2024, 1, 1, 1, 10))

    model, scaler, threshold, history = train_lstm(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col="target",
        seq_len=seq_len,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        batch_size=32,
        max_epochs=3,
        patience=2,
        device="cpu",
    )

    assert isinstance(model, LSTMClassifier)
    assert 0.40 <= threshold <= 0.60
    assert len(history["epoch"]) > 0

    # Test prediction utility
    val_ds = SequenceDataset(
        val_df, feature_cols=feature_cols, target_col="target", seq_len=seq_len, scaler=scaler
    )
    probs, preds = predict_lstm(model, val_ds, device="cpu")
    assert len(probs) == len(val_ds)
    assert len(preds) == len(val_ds)

    # Test artifact save and load
    checkpoint_path = tmp_path / "lstm_test.pt"
    hparams = {"hidden_size": 16, "num_layers": 1, "dropout": 0.0}
    save_lstm_artifacts(model, scaler, threshold, feature_cols, seq_len, hparams, checkpoint_path)

    loaded = load_lstm_artifacts(checkpoint_path, device="cpu")
    assert loaded["threshold"] == threshold
    assert loaded["seq_len"] == seq_len

    # Verify loaded model predictions match
    loaded_probs, _ = predict_lstm(loaded["model"], val_ds, device="cpu")
    np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)
