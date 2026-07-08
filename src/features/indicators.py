"""
Technical indicator computation library.

Implements highly optimized vectorized financial indicator calculations in Polars
grouped by asset symbol to prevent cross-contamination.
"""

import polars as pl


def compute_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Computes technical indicators for OHLCV data.

    Assumes input contains columns: symbol, open_time, open, high, low, close, volume.
    Returns the DataFrame with indicator columns added.
    """
    # Validate required columns
    required_cols = {"symbol", "open_time", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input Polars DataFrame is missing required columns: {missing}"
        )

    # Ensure dataset is chronologically sorted by symbol and timestamp
    df = df.sort(["symbol", "open_time"])

    # Step 1: Base Indicators (Moving Averages, Log Returns, MACD Line)
    df_step1 = df.with_columns(
        [
            pl.col("close").rolling_mean(window_size=15).over("symbol").alias("sma_15"),
            pl.col("close").rolling_mean(window_size=50).over("symbol").alias("sma_50"),
            pl.col("close")
            .ewm_mean(span=15, adjust=False)
            .over("symbol")
            .alias("ema_15"),
            pl.col("close")
            .ewm_mean(span=50, adjust=False)
            .over("symbol")
            .alias("ema_50"),
            # Log return calculation
            (pl.col("close") / pl.col("close").shift(1).over("symbol"))
            .log()
            .alias("log_return"),
            # MACD Base components
            (
                pl.col("close").ewm_mean(span=12, adjust=False).over("symbol")
                - pl.col("close").ewm_mean(span=26, adjust=False).over("symbol")
            ).alias("macd_line"),
            # Bollinger middle band & std deviation
            pl.col("close")
            .rolling_mean(window_size=20)
            .over("symbol")
            .alias("bb_middle"),
            pl.col("close").rolling_std(window_size=20).over("symbol").alias("bb_std"),
        ]
    )

    # Step 2: Derived Indicators (RSI, Volatility, Bollinger Bands, MACD Signal)
    # To compute RSI:
    # 1. Get price change
    # 2. Split change into positive (gain) and negative (loss)
    # 3. Calculate rolling averages
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
            pl.col("macd_line")
            .ewm_mean(span=9, adjust=False)
            .over("symbol")
            .alias("macd_signal"),
            # Bollinger Bands
            (pl.col("bb_middle") + 2 * pl.col("bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - 2 * pl.col("bb_std")).alias("bb_lower"),
            # Volatility (Rolling std of log return over 30 periods)
            pl.col("log_return")
            .rolling_std(window_size=30)
            .over("symbol")
            .alias("volatility_30"),
            # RSI
            rsi.alias("rsi_14"),
        ]
    ).with_columns(
        # MACD Histogram derived from line and signal
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
    )

    # Cleanup intermediate calculation columns
    return df_step2.drop(["bb_std"])


def compute_stationary_features(df: pl.DataFrame) -> pl.DataFrame:
    """Transforms raw indicator prices into stationary relative metrics.

    This ensures features are scale-invariant across different timeframes and symbols.
    """
    return df.with_columns(
        [
            ((pl.col("close") - pl.col("sma_15")) / pl.col("close")).alias(
                "close_to_sma_15"
            ),
            ((pl.col("close") - pl.col("sma_50")) / pl.col("close")).alias(
                "close_to_sma_50"
            ),
            ((pl.col("close") - pl.col("ema_15")) / pl.col("close")).alias(
                "close_to_ema_15"
            ),
            ((pl.col("close") - pl.col("ema_50")) / pl.col("close")).alias(
                "close_to_ema_50"
            ),
            (
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-10)
            ).alias("bb_position"),
            (pl.col("macd_line") / pl.col("close")).alias("macd_line_norm"),
            (pl.col("macd_signal") / pl.col("close")).alias("macd_signal_norm"),
            (pl.col("macd_hist") / pl.col("close")).alias("macd_hist_norm"),
        ]
    )
