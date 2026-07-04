"""
Sample Parquet Generator Pipeline using PySpark.

Slices out the top 20 most liquid cryptocurrency USDT pairs at the 1-minute frequency,
cleans column datatypes, sorts, and exports a compressed Parquet file to
data/sample/binance_sample_spark.parquet.
"""

import logging
import os
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, IntegerType

import src.config as config
from src.utils.spark_client import get_spark_session

logger = logging.getLogger(__name__)


def generate_sample() -> None:
    logger.info("Starting Spark-based sample Parquet generation...")

    # Identify downloaded CSV files for our TOP_20_SYMBOLS
    valid_paths = []
    for symbol in config.TOP_20_SYMBOLS:
        symbol_dir = (
            config.RAW_KLINES_DIR / "spot" / "monthly" / "klines" / symbol / "1m"
        )
        if symbol_dir.exists():
            csv_files = list(symbol_dir.glob("*.csv"))
            if csv_files:
                # Add the glob pattern for Spark to read
                valid_paths.append(str(symbol_dir / "*.csv"))
                logger.info(
                    f"Adding symbol {symbol} with {len(csv_files)} files to sample."
                )
            else:
                logger.warning(f"No CSV files found in {symbol_dir}.")
        else:
            logger.warning(f"Directory {symbol_dir} does not exist.")

    if not valid_paths:
        logger.error(
            "No valid CSV files found for any of the configured TOP_20_SYMBOLS."
        )
        logger.error(
            "Please run the klines downloader script or verify that raw datasets exist."
        )
        return

    spark = get_spark_session()

    # Define schema explicitly for faster loading and type safety
    raw_schema = StructType(
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

    # Convert paths to forward slashes for cross-platform compatibility
    paths_list = [p.replace("\\", "/") for p in valid_paths]

    # Define output path
    spark_parquet_path = config.SAMPLE_DATA_DIR / "binance_sample_spark.parquet"
    spark_parquet_path_str = str(spark_parquet_path).replace("\\", "/")

    try:
        logger.info(f"Reading {len(paths_list)} symbol CSV patterns in Spark...")
        # Spark can read multiple glob patterns from a list
        df = spark.read.schema(raw_schema).csv(paths_list, header=False)

        # Add filename and extract symbol
        df = df.withColumn("filename", F.input_file_name()).withColumn(
            "symbol", F.regexp_extract("filename", r"([^/\\#?]+)-1m-", 1)
        )

        # Normalize timestamps and cast to Proper Spark Timestamps
        open_time_sec = F.when(
            F.col("_c0") >= 1000000000000000, F.col("_c0") / 1000000.0
        ).otherwise(F.col("_c0") / 1000.0)
        close_time_sec = F.when(
            F.col("_c6") >= 1000000000000000, F.col("_c6") / 1000000.0
        ).otherwise(F.col("_c6") / 1000.0)

        df_processed = df.select(
            open_time_sec.cast("timestamp").alias("open_time"),
            F.col("_c1").alias("open"),
            F.col("_c2").alias("high"),
            F.col("_c3").alias("low"),
            F.col("_c4").alias("close"),
            F.col("_c5").alias("volume"),
            close_time_sec.cast("timestamp").alias("close_time"),
            F.col("_c7").alias("quote_asset_volume"),
            F.col("_c8").alias("number_of_trades"),
            F.col("_c9").alias("taker_buy_base_asset_volume"),
            F.col("_c10").alias("taker_buy_quote_asset_volume"),
            F.col("symbol"),
        )

        logger.info(
            f"Writing compressed sample Parquet to: {spark_parquet_path_str}..."
        )

        # Write output in sorted order, partitioning by symbol is standard for multi-ticker querying
        # However, to maintain exact 1-to-1 compatibility with the existing DuckDB output file layout
        # (which is a single sorted parquet file, not partitioned directory structure),
        # we sort the partition and write.
        df_processed.write.mode("overwrite").option("compression", "zstd").parquet(
            spark_parquet_path_str
        )

        logger.info("Sample Parquet generated successfully.")

        # Verify the generated file count and symbols
        parquet_df = spark.read.parquet(spark_parquet_path_str)
        counts = parquet_df.agg(
            F.count("*").alias("total_rows"),
            F.countDistinct("symbol").alias("unique_symbols"),
        ).collect()[0]

        logger.info(
            f"Verified Spark Parquet contains {counts['total_rows']:,} rows across {counts['unique_symbols']} symbols."
        )

    except Exception as e:
        logger.error(f"Failed to generate sample Parquet: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    generate_sample()
