"""
Machine learning model training and dataset splitting pipeline.

Implements chronological validation splitting, feature extraction, scaling,
and fitting of classifiers (Logistic Regression and Random Forest).
"""

import pickle
from pathlib import Path
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Setup logging
import logging

logger = logging.getLogger(__name__)


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
) -> tuple[list, list]:
    """Extracts features and target labels as NumPy arrays, dropping any remaining nulls."""
    clean_df = df.select(feature_cols + [target_col]).drop_nulls()

    if len(clean_df) == 0:
        raise ValueError(
            f"No data remaining after dropping null values from feature columns: {feature_cols}"
        )

    # Extract features and targets
    X = clean_df.select(feature_cols).to_numpy()
    y = clean_df.select(target_col).to_numpy().ravel()

    return X, y


def train_pipeline(X_train, y_train, X_val, y_val, feature_cols: list[str]) -> dict:
    """Trains a Logistic Regression and a Random Forest Classifier on scaled features.

    Returns a dictionary containing the trained models, scaler, and logs.
    """
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    lr.fit(X_train_scaled, y_train)
    lr_val_acc = lr.score(X_val_scaled, y_val)
    logger.info(f"Logistic Regression Validation Accuracy: {lr_val_acc:.4f}")

    # 2. Random Forest Classifier
    logger.info("Training Random Forest Classifier (this may take a few moments)...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_val_acc = rf.score(X_val_scaled, y_val)
    logger.info(f"Random Forest Validation Accuracy: {rf_val_acc:.4f}")

    return {
        "scaler": scaler,
        "logistic_regression": lr,
        "random_forest": rf,
        "feature_names": feature_cols,
    }


def save_model_artifacts(artifacts: dict, dest_dir: Path) -> None:
    """Saves model and scaler binaries as pickle files."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Save the consolidated artifacts dictionary
    artifact_path = dest_dir / "ml_artifacts.pkl"
    logger.info(f"Saving ML artifacts to {artifact_path}...")
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)
    logger.info("Model artifacts saved successfully.")
