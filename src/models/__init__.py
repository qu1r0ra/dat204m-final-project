"""Models package containing baseline classification, PySpark distributed ML,
and PyTorch sequence models.
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
from src.models.train import (
    ModelArtifacts,
    load_model_artifacts,
    save_model_artifacts,
    train_pipeline,
)

__all__ = [
    "BaselineArtifacts",
    "LSTMClassifier",
    "ModelArtifacts",
    "SequenceDataset",
    "load_baseline_artifacts",
    "load_lstm_artifacts",
    "load_model_artifacts",
    "predict_lstm",
    "predict_majority_class",
    "predict_ols_direction",
    "predict_ols_return",
    "save_baseline_artifacts",
    "save_lstm_artifacts",
    "save_model_artifacts",
    "train_baselines",
    "train_lstm",
    "train_pipeline",
]
