"""
Data preprocessing and profiling pipeline.

Scans the raw downloaded CSV dataset, performs basic profiling (row counts,
date bounds, duplicate counts, and data gaps), and writes the output report
to docs/data_profile.md.
"""

import logging
import duckdb

import src.config as config
from src.utils.helpers import generate_profile_markdown

# Configure logging
logger = logging.getLogger(__name__)


def run_profiling() -> None:
    logger.info("Starting dataset profiling...")

    # Pattern to match all Spot Monthly 1m Kline CSV files
    csv_pattern = str(
        config.RAW_KLINES_DIR / "spot" / "monthly" / "klines" / "*" / "1m" / "*.csv"
    )

    # Check if any files exist before launching DuckDB
    base_dir = config.RAW_KLINES_DIR / "spot" / "monthly" / "klines"
    if not base_dir.exists():
        logger.error(f"Raw data directory does not exist: {base_dir}")
        logger.error(
            "Please run the downloader script first or place datasets under data/raw/binance_data/."
        )
        return

    # Use glob to verify there's at least one CSV file
    csv_files = list(base_dir.glob("*/1m/*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in: {base_dir}/*/1m/")
        return

    logger.info(f"Found {len(csv_files)} raw CSV files to profile.")

    # Query description:
    # column0 = open_time (epoch milliseconds)
    # column1-4 = OHLC
    # column5 = volume
    csv_pattern_str = csv_pattern.replace("\\", "/")
    query = f"""
        SELECT
            regexp_extract(filename, '([^/\\\\]+)-1m-', 1) AS symbol,
            COUNT(*) AS row_count,
            MIN(CASE WHEN column00::BIGINT >= 1000000000000000 THEN (column00::BIGINT / 1000)::BIGINT ELSE column00::BIGINT END) AS min_time_ms,
            MAX(CASE WHEN column00::BIGINT >= 1000000000000000 THEN (column00::BIGINT / 1000)::BIGINT ELSE column00::BIGINT END) AS max_time_ms,
            COUNT(column00) - COUNT(DISTINCT column00) AS duplicate_timestamps,
            COUNT(CASE WHEN column01 IS NULL OR column02 IS NULL OR column03 IS NULL OR column04 IS NULL OR column05 IS NULL THEN 1 END) AS null_values_count
        FROM read_csv('{csv_pattern_str}', header=False, filename=True)
        GROUP BY 1
        ORDER BY row_count DESC
    """

    try:
        with duckdb.connect(database=":memory:") as con:
            logger.info("Running aggregation queries across raw CSV files...")
            df_profile = con.execute(query).df()
    except Exception as e:
        logger.error(f"Failed to profile dataset via DuckDB: {e}")
        return

    report_content = generate_profile_markdown(
        df_profile,
        title="Dataset Profile Report",
        description="This report profiles the raw Binance Spot 1-Minute K-Lines downloaded to the repository.",
    )

    # Ensure output docs folder exists
    config.DATA_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(config.DATA_PROFILE_PATH, "w") as f:
        f.write(report_content)

    logger.info(
        f"Dataset profiling completed successfully! Report written to {config.DATA_PROFILE_PATH}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_profiling()
