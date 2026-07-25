"""
Machine learning model training and dataset splitting pipeline.

Implements chronological validation splitting, feature extraction, scaling,
and fitting of classifiers (Logistic Regression and Random Forest) with seed reproducibility.
"""

import logging
import pickle
import time
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
    df: pl.DataFrame,
    train_end: str,
    val_end: str,
    purge_minutes: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Splits the dataframe chronologically to prevent look-ahead bias.

    - Train: open_time < train_end
    - Val: train_end + purge_minutes <= open_time < val_end
    - Test: open_time >= val_end + purge_minutes
    """
    logger.info(
        f"Chronologically splitting data: train_end={train_end}, "
        f"val_end={val_end}, purge_minutes={purge_minutes}"
    )

    # Parse inputs to datetime objects
    train_end_dt = pl.lit(train_end).str.to_datetime()
    val_end_dt = pl.lit(val_end).str.to_datetime()

    purge_delta = (
        pl.duration(minutes=purge_minutes) if purge_minutes > 0 else pl.duration(seconds=0)
    )
    val_start_dt = train_end_dt + purge_delta
    test_start_dt = val_end_dt + purge_delta

    train_df = df.filter(pl.col("open_time") < train_end_dt)
    val_df = df.filter((pl.col("open_time") >= val_start_dt) & (pl.col("open_time") < val_end_dt))
    test_df = df.filter(pl.col("open_time") >= test_start_dt)

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def prepare_features_and_targets(
    df: pl.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts features and target labels as NumPy arrays, dropping any remaining nulls."""
    missing_cols = set([*feature_cols, target_col]) - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"DataFrame missing required feature/target columns: {missing_cols}"
        )

    clean_df = df.select([*feature_cols, target_col]).drop_nulls()
    rows_dropped = len(df) - len(clean_df)
    if rows_dropped > 0:
        logger.info(
            f"prepare_features_and_targets: dropped {rows_dropped:,} null rows "
            f"({len(clean_df):,} remaining from {len(df):,})"
        )

    if len(clean_df) == 0:
        raise DataValidationError(
            f"No data remaining after dropping null values from feature columns: {feature_cols}"
        )

    # Extract features and targets
    X = clean_df.select(feature_cols).to_numpy()
    y = clean_df.select(target_col).to_numpy().ravel()

    pos_rate = float(y.mean()) if len(y) > 0 else 0.0
    logger.info(
        f"Feature matrix shape: {X.shape}, target '{target_col}' positive rate: {pos_rate:.4f}"
    )

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
    t0 = time.perf_counter()
    lr.fit(X_train_scaled, y_train)
    lr_elapsed = time.perf_counter() - t0
    lr_val_preds = lr.predict(X_val_scaled)
    lr_val_probs = lr.predict_proba(X_val_scaled)[:, 1]
    lr_metrics = calculate_metrics(y_val, lr_val_preds, lr_val_probs)
    log_metrics(lr_metrics, model_name="Logistic Regression")
    logger.info(f"Logistic Regression fitted in {lr_elapsed:.2f}s")

    # Log top LR coefficients
    coef_indices = np.argsort(np.abs(lr.coef_[0]))[::-1][:5]
    top_lr = [(feature_cols[i], float(lr.coef_[0][i])) for i in coef_indices]
    logger.info(f"Top 5 LR coefficients (abs): {top_lr}")

    # 2. Random Forest Classifier
    logger.info("Training Random Forest Classifier (this may take a few moments)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    t0 = time.perf_counter()
    rf.fit(X_train_scaled, y_train)
    rf_elapsed = time.perf_counter() - t0
    rf_val_preds = rf.predict(X_val_scaled)
    rf_val_probs = rf.predict_proba(X_val_scaled)[:, 1]
    rf_metrics = calculate_metrics(y_val, rf_val_preds, rf_val_probs)
    log_metrics(rf_metrics, model_name="Random Forest")
    logger.info(f"Random Forest fitted in {rf_elapsed:.2f}s")

    # Log top RF feature importances
    imp_indices = np.argsort(rf.feature_importances_)[::-1][:5]
    top_rf = [(feature_cols[i], float(rf.feature_importances_[i])) for i in imp_indices]
    logger.info(f"Top 5 RF feature importances: {top_rf}")

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
