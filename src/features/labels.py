"""
Target label creation for machine learning classification and regression tasks.
"""

from typing import TypeVar

import polars as pl

from src.exceptions import DataValidationError

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)


def add_direction_labels(df: FrameType, horizon: int = 15, threshold: float = 0.0) -> FrameType:
    """Computes future returns and binary direction labels over specified horizon per symbol.

    Args:
        df: Polars DataFrame or LazyFrame containing symbol, open_time, close columns.
        horizon: Lookahead horizon (N periods, e.g., 15 for 15m ahead).
        threshold: Return threshold for positive (UP=1) direction label.

    Returns:
        DataFrame/LazyFrame with added columns: future_close, future_return, target.
    """
    required_cols = {"symbol", "open_time", "close"}
    cols = set(df.collect_schema().names()) if isinstance(df, pl.LazyFrame) else set(df.columns)
    missing = required_cols - cols
    if missing:
        raise DataValidationError(
            f"Input DataFrame/LazyFrame is missing required columns for target labeling: {missing}"
        )

    # Ensure dataset is chronologically sorted by symbol and timestamp
    df = df.sort(["symbol", "open_time"])

    future_close = pl.col("close").shift(-horizon).over("symbol")
    future_return = (future_close / pl.col("close")) - 1.0
    target = (future_return > threshold).cast(pl.Int32)

    return df.with_columns(
        [
            future_close.alias("future_close"),
            future_return.alias("future_return"),
            target.alias("target"),
        ]
    )
