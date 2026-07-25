"""
Baseline models (Majority Class and OLS Return Regression) for benchmark comparison.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class BaselineArtifacts:
    majority_label: int
    ols_model: LinearRegression
    feature_cols: list[str]
    decision_threshold: float = 0.0


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    r_train: np.ndarray,
    feature_cols: list[str],
    threshold: float = 0.0,
) -> BaselineArtifacts:
    """Fits majority class label and OLS linear regression model on continuous returns.

    Args:
        X_train: Feature matrix.
        y_train: Binary target array.
        r_train: Continuous target array (future returns).
        feature_cols: List of feature names.
        threshold: Return threshold for positive classification.

    Returns:
        BaselineArtifacts dataclass instance.
    """
    counts = np.bincount(y_train.astype(int))
    majority_label = int(np.argmax(counts))

    ols_model = LinearRegression()
    ols_model.fit(X_train, r_train)

    return BaselineArtifacts(
        majority_label=majority_label,
        ols_model=ols_model,
        feature_cols=feature_cols,
        decision_threshold=threshold,
    )


def predict_majority_class(majority_label: int, n_samples: int) -> np.ndarray:
    """Predicts majority class for n_samples."""
    return np.full(n_samples, majority_label, dtype=int)


def predict_ols_return(ols_model: LinearRegression, X: np.ndarray) -> np.ndarray:
    """Predicts continuous returns using fitted OLS model."""
    return ols_model.predict(X)


def predict_ols_direction(
    ols_model: LinearRegression, X: np.ndarray, threshold: float = 0.0
) -> np.ndarray:
    """Predicts binary direction (1 if return > threshold else 0) using fitted OLS model."""
    r_pred = predict_ols_return(ols_model, X)
    return (r_pred > threshold).astype(int)


def save_baseline_artifacts(artifacts: BaselineArtifacts, dest_dir: Path) -> None:
    """Saves baseline artifacts to destination directory as pickle file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / "baseline_artifacts.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(artifacts, f)


def load_baseline_artifacts(filepath: Path) -> BaselineArtifacts:
    """Loads baseline artifacts from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
