"""
Data preprocessing and profiling pipeline using PySpark.

Scans the raw CSV dataset, performs distributed profiling (row counts,
date bounds, duplicate counts, and data gaps), and writes the output report
to docs/data_profile_spark.md.
"""

import datetime
import logging
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType

import src.config as config
from src.utils.spark_client import get_spark_session

logger = logging.getLogger(__name__)


def ms_to_str(ms: float) -> str:
    """Converts epoch milliseconds to formatted UTC timestamp string."""
    try:
        return datetime.datetime.fromtimestamp(
            ms / 1000.0, datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ms)


def run_profiling() -> None:
    logger.info("Starting Spark-based dataset profiling...")

    # Pattern to match all Spot Monthly 1m Kline CSV files
    csv_pattern = str(
        config.RAW_KLINES_DIR / "spot" / "monthly" / "klines" / "*" / "1m" / "*.csv"
    )

    # Check if any files exist before launching Spark
    base_dir = config.RAW_KLINES_DIR / "spot" / "monthly" / "klines"
    if not base_dir.exists():
        logger.error(f"Raw data directory does not exist: {base_dir}")
        logger.error(
            "Please run the downloader script first or place datasets under data/raw/binance_data/."
        )
        return

    csv_files = list(base_dir.glob("*/1m/*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in: {base_dir}/*/1m/")
        return

    logger.info(f"Found {len(csv_files)} raw CSV files to profile.")

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
            StructField("_c8", LongType(), True),  # number_of_trades
            StructField("_c9", DoubleType(), True),  # taker_buy_base_asset_volume
            StructField("_c10", DoubleType(), True),  # taker_buy_quote_asset_volume
            StructField("_c11", DoubleType(), True),  # ignore/unused
        ]
    )

    csv_pattern_str = csv_pattern.replace("\\", "/")
    logger.info(f"Reading raw CSVs from: {csv_pattern_str}")

    try:
        # Load CSVs in parallel using PySpark
        df = spark.read.schema(raw_schema).csv(csv_pattern_str, header=False)

        # Add filename and extract symbol
        df = df.withColumn("filename", F.input_file_name()).withColumn(
            "symbol", F.regexp_extract("filename", r"([^/\\#?]+)-1m-", 1)
        )

        # Normalize open_time (handle 13-digit and 16-digit timestamps)
        df_normalized = df.withColumn(
            "clean_open_time",
            F.when(
                F.col("_c0") >= 1000000000000000, (F.col("_c0") / 1000).cast(LongType())
            ).otherwise(F.col("_c0").cast(LongType())),
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

        # Calculate duplicate timestamps per symbol efficiently by grouping by (symbol, clean_open_time)
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
        return

    df_profile["start_date"] = df_profile["min_time_ms"].apply(ms_to_str)
    df_profile["end_date"] = df_profile["max_time_ms"].apply(ms_to_str)

    # Calculate missing intervals (1 minute = 60,000 milliseconds)
    df_profile["expected_rows"] = (
        (df_profile["max_time_ms"] - df_profile["min_time_ms"]) / 60000 + 1
    ).astype(int)
    df_profile["missing_rows"] = df_profile["expected_rows"] - df_profile["row_count"]
    df_profile["gap_percentage"] = (
        df_profile["missing_rows"] / df_profile["expected_rows"] * 100
    ).round(4)

    total_symbols = len(df_profile)
    total_rows = df_profile["row_count"].sum()
    total_duplicates = df_profile["duplicate_timestamps"].sum()
    total_nulls = df_profile["null_values_count"].sum()

    report_lines = [
        "# Dataset Profile Report (Spark Edition)",
        "",
        "This report profiles the raw Binance Spot 1-Minute K-Lines using PySpark.",
        "",
        "## Summary Metrics",
        "",
        "- **Total Unique Symbols**: " + str(total_symbols),
        f"- **Total Rows (Observations)**: {total_rows:,}",
        f"- **Total Duplicate Timestamps**: {total_duplicates}",
        f"- **Total Rows with Null Values**: {total_nulls}",
        "",
        "## Detailed Symbol Analysis",
        "",
        "| Symbol | Total Rows | Expected Rows | Gaps (Rows) | Gap % | Start Date (UTC) | End Date (UTC) | Duplicates | Nulls |",
        "| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: |",
    ]

    for _, row in df_profile.iterrows():
        report_lines.append(
            f"| {row['symbol']} | {row['row_count']:,} | {row['expected_rows']:,} | {row['missing_rows']:,} | {row['gap_percentage']}% | {row['start_date']} | {row['end_date']} | {row['duplicate_timestamps']} | {row['null_values_count']} |"
        )

    # Save the output Spark profile report
    output_profile_path = config.DOCS_DIR / "data_profile_spark.md"
    output_profile_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_profile_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info(
        f"Dataset profiling completed successfully! Spark report written to {output_profile_path}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_profiling()
