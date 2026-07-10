"""
Shared schemas for pipeline datasets.
"""

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
