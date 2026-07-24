"""
Machine learning model training and dataset splitting pipeline.

Implements chronological validation splitting, feature extraction, scaling,
and fitting of classifiers (Logistic Regression and Random Forest) with seed reproducibility.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.exceptions import DataValidationError
from src.models.evaluation import calculate_metrics, log_metrics, save_metrics_json
from src.utils.seed import get_provenance_metadata, set_seed

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    scaler: StandardScaler
    logistic_regression: LogisticRegression
    random_forest: RandomForestClassifier
    feature_names: list[str]
    metrics: dict[str, dict[str, Any]] | None = None
    provenance: dict[str, Any] | None = None

    def to_trainer_result(self) -> Any:
        from src.models.base import TrainerResult

        return TrainerResult(
            model_name="Sklearn Ensemble",
            model={
                "logistic_regression": self.logistic_regression,
                "random_forest": self.random_forest,
            },
            scaler=self.scaler,
            metrics=self.metrics or {},
            provenance=self.provenance or {},
            feature_names=self.feature_names,
        )


def split_data_chronologically(
    df: pl.DataFrame, train_end: str, val_end: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Splits the dataframe chronologically to prevent look-ahead bias.

    - Train: open_time < train_end
    - Val: train_end <= open_time < val_end
    - Test: open_time >= val_end
    """
    logger.info(
        f"Chronologically splitting data: train_end={train_end}, val_end={val_end}"
    )

    # Parse inputs to datetime objects
    train_end_dt = pl.lit(train_end).str.to_datetime()
    val_end_dt = pl.lit(val_end).str.to_datetime()

    train_df = df.filter(pl.col("open_time") < train_end_dt)
    val_df = df.filter(
        (pl.col("open_time") >= train_end_dt) & (pl.col("open_time") < val_end_dt)
    )
    test_df = df.filter(pl.col("open_time") >= val_end_dt)

    logger.info(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def prepare_features_and_targets(
    df: pl.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts features and target labels as NumPy arrays, dropping any remaining nulls."""
    missing_cols = set(feature_cols + [target_col]) - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"DataFrame missing required feature/target columns: {missing_cols}"
        )

    clean_df = df.select(feature_cols + [target_col]).drop_nulls()

    if len(clean_df) == 0:
        raise DataValidationError(
            f"No data remaining after dropping null values from feature columns: {feature_cols}"
        )

    # Extract features and targets
    X = clean_df.select(feature_cols).to_numpy()
    y = clean_df.select(target_col).to_numpy().ravel()

    return X, y


def train_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    seed: int = 42,
) -> ModelArtifacts:
    """Trains a Logistic Regression and a Random Forest Classifier on scaled features.

    Returns ModelArtifacts containing trained models, scaler, metrics, and provenance.
    """
    set_seed(seed)

    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=seed, C=0.1)
    lr.fit(X_train_scaled, y_train)
    lr_val_preds = lr.predict(X_val_scaled)
    lr_val_probs = lr.predict_proba(X_val_scaled)[:, 1]
    lr_metrics = calculate_metrics(y_val, lr_val_preds, lr_val_probs)
    log_metrics(lr_metrics, model_name="Logistic Regression")

    # 2. Random Forest Classifier
    logger.info("Training Random Forest Classifier (this may take a few moments)...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_val_preds = rf.predict(X_val_scaled)
    rf_val_probs = rf.predict_proba(X_val_scaled)[:, 1]
    rf_metrics = calculate_metrics(y_val, rf_val_preds, rf_val_probs)
    log_metrics(rf_metrics, model_name="Random Forest")

    metrics_dict = {
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
    }

    provenance = get_provenance_metadata(seed)

    return ModelArtifacts(
        scaler=scaler,
        logistic_regression=lr,
        random_forest=rf,
        feature_names=feature_cols,
        metrics=metrics_dict,
        provenance=provenance,
    )


def save_model_artifacts(artifacts: ModelArtifacts, dest_dir: Path) -> None:
    """Saves model and scaler binaries as pickle files."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Save the consolidated artifacts dictionary
    artifact_path = dest_dir / "ml_artifacts.pkl"
    logger.info(f"Saving ML artifacts to {artifact_path}...")
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)

    if artifacts.metrics:
        json_path = dest_dir / "ml_metrics.json"
        save_metrics_json(artifacts.metrics, json_path)

    logger.info("Model artifacts saved successfully.")
