"""
Models package containing baseline classification, PySpark distributed ML, and PyTorch sequence models.
"""

from src.models.baselines import (
    BaselineArtifacts,
    load_baseline_artifacts,
    predict_majority_class,
    predict_ols_direction,
    predict_ols_return,
    save_baseline_artifacts,
    train_baselines,
)
from src.models.lstm import (
    LSTMClassifier,
    SequenceDataset,
    load_lstm_artifacts,
    predict_lstm,
    save_lstm_artifacts,
    train_lstm,
)

__all__ = [
    "BaselineArtifacts",
    "train_baselines",
    "predict_majority_class",
    "predict_ols_return",
    "predict_ols_direction",
    "save_baseline_artifacts",
    "load_baseline_artifacts",
    "LSTMClassifier",
    "SequenceDataset",
    "train_lstm",
    "predict_lstm",
    "save_lstm_artifacts",
    "load_lstm_artifacts",
]
