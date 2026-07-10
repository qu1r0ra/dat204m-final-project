"""
Data preprocessing and profiling pipeline using PySpark.

Scans the raw CSV dataset, performs distributed profiling (row counts,
date bounds, duplicate counts, and data gaps), and writes the output report
to docs/data_profile_spark.md.
"""

import logging

from pyspark.sql import functions as F

import src.config as config
from src.pipeline.schemas import (
    RAW_KLINE_CSV_SCHEMA,
    get_spark_timestamp_ms_col,
)
from src.utils.helpers import discover_all_csvs, generate_profile_markdown
from src.utils.spark_client import get_spark_session

logger = logging.getLogger(__name__)


def run_profiling() -> None:
    logger.info("Starting Spark-based dataset profiling...")

    # Check if any files exist before launching Spark
    base_dir = config.RAW_KLINES_DIR
    if not (base_dir / "spot" / "monthly" / "klines").exists():
        logger.error(f"Raw data directory does not exist: {base_dir}")
        logger.error(
            "Please run the downloader script first or place datasets under data/raw/binance_data/."
        )
        return

    csv_files = discover_all_csvs(base_dir)
    if not csv_files:
        logger.error(f"No CSV files found in: {base_dir}/*/1m/")
        return

    logger.info(f"Found {len(csv_files)} raw CSV files to profile.")

    spark = get_spark_session()

    # Define schema explicitly for faster loading and type safety
    raw_schema = RAW_KLINE_CSV_SCHEMA

    paths_list = [str(p).replace("\\", "/") for p in csv_files]
    logger.info(f"Reading {len(paths_list)} resolved raw CSVs in Spark...")

    try:
        # Load CSVs in parallel using PySpark
        df = spark.read.schema(raw_schema).csv(paths_list, header=False)

        # Add filename and extract symbol
        df = df.withColumn("filename", F.input_file_name()).withColumn(
            "symbol", F.regexp_extract("filename", r"([^/\\#?]+)-1m-", 1)
        )

        # Normalize open_time (handle 13-digit and 16-digit timestamps)
        df_normalized = df.withColumn(
            "clean_open_time",
            get_spark_timestamp_ms_col("_c0"),
        )

        # Check for null rows in core pricing columns
        null_check = F.when(
            F.col("_c1").isNull()
            | F.col("_c2").isNull()
            | F.col("_c3").isNull()
            | F.col("_c4").isNull()
            | F.col("_c5").isNull(),
            1,
        ).otherwise(0)
        df_normalized = df_normalized.withColumn("is_null", null_check)

        logger.info("Executing Spark profiling aggregations...")

        # Aggregate stats grouped by symbol (without countDistinct)
        base_agg_df = df_normalized.groupBy("symbol").agg(
            F.count("*").alias("row_count"),
            F.min("clean_open_time").alias("min_time_ms"),
            F.max("clean_open_time").alias("max_time_ms"),
            F.sum("is_null").alias("null_values_count"),
        )

        # Calculate duplicate timestamps per symbol efficiently
        # by grouping by (symbol, clean_open_time)
        dup_df = (
            df_normalized.groupBy("symbol", "clean_open_time")
            .count()
            .filter(F.col("count") > 1)
            .groupBy("symbol")
            .agg(F.sum(F.col("count") - 1).alias("duplicate_timestamps"))
        )

        # Join the base stats and duplicates
        agg_df = base_agg_df.join(dup_df, on="symbol", how="left").fillna(
            0, subset=["duplicate_timestamps"]
        )
        agg_df = agg_df.orderBy(F.desc("row_count"))

        # Collect results to driver (compact summary dataset, safe to convert to Pandas)
        df_profile = agg_df.toPandas()

    except Exception as e:
        logger.error(f"Failed to profile dataset via PySpark: {e}")
        raise

    report_content = generate_profile_markdown(
        df_profile,
        title="Dataset Profile Report (Spark Edition)",
        description="This report profiles the raw Binance Spot 1-Minute K-Lines using PySpark.",
    )

    # Save the output Spark profile report
    output_profile_path = config.DOCS_DIR / "data_profile_spark.md"
    output_profile_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_profile_path, "w") as f:
        f.write(report_content)

    logger.info(
        f"Dataset profiling completed successfully! Spark report written to {output_profile_path}"
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        run_profiling()
    except Exception:
        sys.exit(1)
