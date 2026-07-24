"""
Technical indicator computation library.

Implements highly optimized vectorized financial indicator calculations in Polars
grouped by asset symbol to prevent cross-contamination. Supports both pl.DataFrame
and pl.LazyFrame for memory-efficient query execution.
"""

from typing import Any, TypeVar

import polars as pl

from src.exceptions import DataValidationError

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)


def compute_indicators(df: FrameType) -> FrameType:
    """Computes technical indicators for OHLCV data.

    Assumes input contains columns: symbol, open_time, open, high, low, close, volume.
    Supports both pl.DataFrame and pl.LazyFrame inputs.
    Returns the DataFrame or LazyFrame with indicator columns added.
    """
    # Validate required columns
    required_cols = {"symbol", "open_time", "open", "high", "low", "close", "volume"}
    cols = set(df.collect_schema().names()) if isinstance(df, pl.LazyFrame) else set(df.columns)
    missing = required_cols - cols
    if missing:
        raise DataValidationError(
            f"Input Polars DataFrame/LazyFrame is missing required columns: {missing}"
        )

    # Ensure dataset is chronologically sorted by symbol and timestamp
    df = df.sort(["symbol", "open_time"])

    # Step 1: Base Indicators (Moving Averages, Log Returns, MACD Line)
    df_step1 = df.with_columns(
        [
            pl.col("close").rolling_mean(window_size=15).over("symbol").alias("sma_15"),
            pl.col("close").rolling_mean(window_size=50).over("symbol").alias("sma_50"),
            pl.col("close").ewm_mean(span=15, adjust=False).over("symbol").alias("ema_15"),
            pl.col("close").ewm_mean(span=50, adjust=False).over("symbol").alias("ema_50"),
            # Log return calculation
            (pl.col("close") / pl.col("close").shift(1).over("symbol")).log().alias("log_return"),
            # MACD Base components
            (
                pl.col("close").ewm_mean(span=12, adjust=False).over("symbol")
                - pl.col("close").ewm_mean(span=26, adjust=False).over("symbol")
            ).alias("macd_line"),
            # Bollinger middle band & std deviation
            pl.col("close").rolling_mean(window_size=20).over("symbol").alias("bb_middle"),
            pl.col("close").rolling_std(window_size=20).over("symbol").alias("bb_std"),
        ]
    )

    # Step 2: Derived Indicators (RSI, Volatility, Bollinger Bands, MACD Signal)
    change = pl.col("close").diff().over("symbol")
    gain = pl.when(change > 0).then(change).otherwise(0.0)
    loss = pl.when(change < 0).then(-change).otherwise(0.0)

    avg_gain = gain.rolling_mean(window_size=14).over("symbol")
    avg_loss = loss.rolling_mean(window_size=14).over("symbol")
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    df_step2 = df_step1.with_columns(
        [
            # MACD Signal and Histogram
            pl.col("macd_line").ewm_mean(span=9, adjust=False).over("symbol").alias("macd_signal"),
            # Bollinger Bands
            (pl.col("bb_middle") + 2 * pl.col("bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - 2 * pl.col("bb_std")).alias("bb_lower"),
            # Volatility (Rolling std of log return over 30 periods)
            pl.col("log_return").rolling_std(window_size=30).over("symbol").alias("volatility_30"),
            # RSI
            rsi.alias("rsi_14"),
        ]
    ).with_columns(
        # MACD Histogram derived from line and signal
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
    )

    # Cleanup intermediate calculation columns
    return df_step2.drop(["bb_std"])


def compute_stationary_features(df: FrameType) -> FrameType:
    """Transforms raw indicator prices into stationary relative metrics.

    This ensures features are scale-invariant across different timeframes and symbols.
    Supports both pl.DataFrame and pl.LazyFrame inputs.
    """
    required_cols = {
        "close",
        "sma_15",
        "sma_50",
        "ema_15",
        "ema_50",
        "bb_lower",
        "bb_upper",
        "macd_line",
        "macd_signal",
        "macd_hist",
    }
    cols = set(df.collect_schema().names()) if isinstance(df, pl.LazyFrame) else set(df.columns)
    missing = required_cols - cols
    if missing:
        raise DataValidationError(
            f"Input Polars DataFrame/LazyFrame is missing required indicator columns: {missing}"
        )

    return df.with_columns(
        [
            ((pl.col("close") - pl.col("sma_15")) / pl.col("close")).alias("close_to_sma_15"),
            ((pl.col("close") - pl.col("sma_50")) / pl.col("close")).alias("close_to_sma_50"),
            ((pl.col("close") - pl.col("ema_15")) / pl.col("close")).alias("close_to_ema_15"),
            ((pl.col("close") - pl.col("ema_50")) / pl.col("close")).alias("close_to_ema_50"),
            (
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-10)
            ).alias("bb_position"),
            (pl.col("macd_line") / pl.col("close")).alias("macd_line_norm"),
            (pl.col("macd_signal") / pl.col("close")).alias("macd_signal_norm"),
            (pl.col("macd_hist") / pl.col("close")).alias("macd_hist_norm"),
        ]
    )


def validate_feature_parity(
    polars_df: pl.DataFrame,
    spark_df_pandas: Any,
    feature_cols: list[str] | None = None,
    tol: float = 1e-5,
) -> bool:
    """Asserts mathematical equivalence between Polars and Spark feature outputs within tolerance.

    Args:
        polars_df: Output DataFrame from Polars feature engineering.
        spark_df_pandas: Output DataFrame from PySpark Pandas Grouped Map UDF converted to Pandas.
        feature_cols: Feature column names to check for parity.
        tol: Absolute tolerance limit.

    Returns:
        True if all feature columns match within tolerance.
    """
    import numpy as np

    from src.config import FEATURE_COLS

    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    check_cols = [c for c in cols if c in polars_df.columns and c in spark_df_pandas.columns]

    for col in check_cols:
        p_vals = polars_df.select(col).drop_nulls().to_numpy().ravel()
        s_vals = spark_df_pandas[col].dropna().to_numpy().ravel()
        min_len = min(len(p_vals), len(s_vals))
        if min_len > 0:
            max_diff = float(np.max(np.abs(p_vals[:min_len] - s_vals[:min_len])))
            if max_diff > tol:
                raise DataValidationError(
                    f"Feature parity validation failed for column '{col}': "
                    f"max difference {max_diff:.6f} > tolerance {tol}"
                )
    return True
