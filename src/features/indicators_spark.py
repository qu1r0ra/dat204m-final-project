"""
Technical indicator computation for Spark DataFrames.

Applies highly optimized Polars technical indicator calculations in parallel
across symbols using Spark Pandas Grouped Map UDFs (applyInPandas).
"""

import polars as pl
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    LongType,
    TimestampType,
    StringType,
    IntegerType,
)
from src.features.indicators import compute_indicators, compute_stationary_features

# Complete schema for the Spark DataFrame after running compute_indicators
# and compute_stationary_features.
INDICATOR_SCHEMA = StructType(
    [
        StructField("open_time", TimestampType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
        StructField("close_time", TimestampType(), True),
        StructField("quote_asset_volume", DoubleType(), True),
        StructField("number_of_trades", IntegerType(), True),
        StructField("taker_buy_base_asset_volume", DoubleType(), True),
        StructField("taker_buy_quote_asset_volume", DoubleType(), True),
        StructField("symbol", StringType(), True),
        # Computed indicators
        StructField("sma_15", DoubleType(), True),
        StructField("sma_50", DoubleType(), True),
        StructField("ema_15", DoubleType(), True),
        StructField("ema_50", DoubleType(), True),
        StructField("log_return", DoubleType(), True),
        StructField("macd_line", DoubleType(), True),
        StructField("bb_middle", DoubleType(), True),
        StructField("macd_signal", DoubleType(), True),
        StructField("bb_upper", DoubleType(), True),
        StructField("bb_lower", DoubleType(), True),
        StructField("volatility_30", DoubleType(), True),
        StructField("rsi_14", DoubleType(), True),
        StructField("macd_hist", DoubleType(), True),
        # Stationary features
        StructField("close_to_sma_15", DoubleType(), True),
        StructField("close_to_sma_50", DoubleType(), True),
        StructField("close_to_ema_15", DoubleType(), True),
        StructField("close_to_ema_50", DoubleType(), True),
        StructField("bb_position", DoubleType(), True),
        StructField("macd_line_norm", DoubleType(), True),
        StructField("macd_signal_norm", DoubleType(), True),
        StructField("macd_hist_norm", DoubleType(), True),
    ]
)


def compute_indicators_spark(spark_df) -> pl.DataFrame:
    """Computes technical indicators and stationary features on a PySpark DataFrame.

    Uses applyInPandas to run the optimized Polars feature engineering in parallel
    per symbol across Spark partitions.
    """

    def pandas_udf_wrapper(pdf):
        # Convert incoming pandas DataFrame to Polars
        pl_df = pl.from_pandas(pdf)

        # Run original, validated feature calculations
        pl_df_indicators = compute_indicators(pl_df)
        pl_df_stationary = compute_stationary_features(pl_df_indicators)

        # Return as pandas DataFrame matching INDICATOR_SCHEMA
        # Ensure column ordering matches schema
        cols_order = INDICATOR_SCHEMA.fieldNames()
        pdf_out = pl_df_stationary.to_pandas()
        return pdf_out[cols_order]

    # Group by symbol and execute the Pandas Grouped Map UDF in parallel
    return spark_df.groupBy("symbol").applyInPandas(
        pandas_udf_wrapper, schema=INDICATOR_SCHEMA
    )
