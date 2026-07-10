"""
Shared schemas for pipeline datasets.
"""

from pyspark.sql import Column
from pyspark.sql.types import DoubleType, IntegerType, LongType, StructField, StructType

RAW_KLINE_CSV_SCHEMA = StructType(
    [
        StructField("_c0", LongType(), True),  # open_time (epoch ms)
        StructField("_c1", DoubleType(), True),  # open
        StructField("_c2", DoubleType(), True),  # high
        StructField("_c3", DoubleType(), True),  # low
        StructField("_c4", DoubleType(), True),  # close
        StructField("_c5", DoubleType(), True),  # volume
        StructField("_c6", LongType(), True),  # close_time (epoch ms)
        StructField("_c7", DoubleType(), True),  # quote_asset_volume
        StructField("_c8", IntegerType(), True),  # number_of_trades
        StructField("_c9", DoubleType(), True),  # taker_buy_base_asset_volume
        StructField("_c10", DoubleType(), True),  # taker_buy_quote_asset_volume
        StructField("_c11", DoubleType(), True),  # ignore/unused
    ]
)


# SQL fragments for DuckDB timestamp normalization
def get_duckdb_timestamp_sql(col_name: str) -> str:
    """Returns SQL fragment for normalizing timestamps to milliseconds."""
    return (
        f"CASE WHEN {col_name}::BIGINT >= 1000000000000000 THEN "
        f"({col_name}::BIGINT / 1000)::BIGINT ELSE {col_name}::BIGINT END"
    )


def get_duckdb_epoch_ms_sql(col_name: str) -> str:
    """Returns SQL fragment for normalizing timestamps to DuckDB epoch_ms."""
    return (
        f"CASE WHEN {col_name}::BIGINT >= 1000000000000000 THEN "
        f"epoch_ms(({col_name}::BIGINT / 1000)::BIGINT) ELSE "
        f"epoch_ms({col_name}::BIGINT) END"
    )


# PySpark column expressions for timestamp normalization
def get_spark_timestamp_ms_col(col_name: str) -> "Column":
    """Returns PySpark Column expression for normalizing timestamps to milliseconds."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import LongType

    return F.when(
        F.col(col_name) >= 1000000000000000, (F.col(col_name) / 1000).cast(LongType())
    ).otherwise(F.col(col_name).cast(LongType()))


def get_spark_timestamp_sec_col(col_name: str) -> "Column":
    """Returns PySpark Column expression for normalizing timestamps to seconds."""
    from pyspark.sql import functions as F

    return F.when(F.col(col_name) >= 1000000000000000, F.col(col_name) / 1000000.0).otherwise(
        F.col(col_name) / 1000.0
    )
