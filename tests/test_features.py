import polars as pl

from src.features.indicators import compute_indicators, compute_stationary_features


def test_compute_indicators_basic(sample_ohlcv_df):
    """Verify compute_indicators adds expected columns and RSI is bounded."""
    df = compute_indicators(sample_ohlcv_df)

    # Check if all columns are present
    expected_cols = {
        "sma_15",
        "sma_50",
        "ema_15",
        "ema_50",
        "log_return",
        "macd_line",
        "bb_middle",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "volatility_30",
        "rsi_14",
        "macd_hist",
    }
    assert expected_cols.issubset(set(df.columns))

    # Verify RSI is bounded [0, 100]
    rsi_vals = df["rsi_14"].drop_nulls()
    assert (rsi_vals >= 0).all()
    assert (rsi_vals <= 100).all()


def test_compute_stationary_features_edge_case(sample_ohlcv_df):
    """Verify stationary features handles division by zero."""
    df_ind = compute_indicators(sample_ohlcv_df)

    # Force bb_upper == bb_lower to test division by zero protection
    df_ind = df_ind.with_columns(bb_upper=pl.lit(100.0), bb_lower=pl.lit(100.0))

    df_stat = compute_stationary_features(df_ind)

    # Check if bb_position has no inf or nan
    bb_pos = df_stat["bb_position"].drop_nulls()
    assert not bb_pos.is_infinite().any()
    assert not bb_pos.is_nan().any()

    # Verify expected column names
    expected_cols = {
        "close_to_sma_15",
        "close_to_sma_50",
        "close_to_ema_15",
        "close_to_ema_50",
        "bb_position",
        "macd_line_norm",
        "macd_signal_norm",
        "macd_hist_norm",
    }
    assert expected_cols.issubset(set(df_stat.columns))
