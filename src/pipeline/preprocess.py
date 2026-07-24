"""
Data preprocessing and profiling pipeline.

Scans the raw downloaded CSV dataset, performs basic profiling (row counts,
date bounds, duplicate counts, and data gaps), and writes the output report
to docs/data_profile.md.
"""

import logging

import duckdb
import polars as pl

import src.config as config
from src.exceptions import DataValidationError
from src.pipeline.schemas import get_duckdb_timestamp_sql
from src.utils.helpers import (
    discover_all_csvs,
    generate_profile_markdown,
    normalize_path_str,
)

# Configure logging
logger = logging.getLogger(__name__)


def compute_split_boundaries(
    df: pl.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[str, str]:
    """Computes boundary timestamps for chronological splits based on timestamp quantiles.

    Args:
        df: Polars DataFrame containing 'open_time' column.
        train_frac: Fraction of data allocated to training (default 0.70).
        val_frac: Fraction of data allocated to validation (default 0.15).

    Returns:
        Tuple of (train_end_str, val_end_str) in 'YYYY-MM-DD HH:MM:SS' format.
    """
    times = df.select("open_time").sort("open_time")
    n = len(times)
    if n == 0:
        raise DataValidationError("Cannot compute split boundaries on an empty DataFrame.")

    train_idx = int(n * train_frac)
    val_idx = int(n * (train_frac + val_frac))

    train_end_val = times["open_time"][min(train_idx, n - 1)]
    val_end_val = times["open_time"][min(val_idx, n - 1)]

    if isinstance(train_end_val, str):
        return train_end_val, str(val_end_val)

    train_end_str = train_end_val.strftime("%Y-%m-%d %H:%M:%S")
    val_end_str = val_end_val.strftime("%Y-%m-%d %H:%M:%S")

    return train_end_str, val_end_str


def run_profiling() -> None:
    logger.info("Starting dataset profiling...")

    # Use helper to discover CSV files
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

    # Query description:
    # column0 = open_time (epoch milliseconds)
    # column1-4 = OHLC
    # column5 = volume
    csv_pattern_str = normalize_path_str(
        config.RAW_KLINES_DIR / "spot" / "monthly" / "klines" / "*" / "1m" / "*.csv"
    )

    col0_ms = get_duckdb_timestamp_sql("column00")

    query = f"""
        SELECT
            regexp_extract(filename, '([^/\\\\]+)-1m-', 1) AS symbol,
            COUNT(*) AS row_count,
            MIN({col0_ms}) AS min_time_ms,
            MAX({col0_ms}) AS max_time_ms,
            COUNT(column00) - COUNT(DISTINCT column00) AS duplicate_timestamps,
            COUNT(CASE
                WHEN column01 IS NULL OR column02 IS NULL OR column03 IS NULL
                     OR column04 IS NULL OR column05 IS NULL THEN 1
            END) AS null_values_count
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
        raise

    report_content = generate_profile_markdown(
        df_profile,
        title="Dataset Profile Report",
        description=(
            "This report profiles the raw Binance Spot 1-Minute K-Lines "
            "downloaded to the repository."
        ),
    )

    # Ensure output docs folder exists
    config.DATA_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(config.DATA_PROFILE_PATH, "w") as f:
        f.write(report_content)

    logger.info(
        f"Dataset profiling completed successfully! Report written to {config.DATA_PROFILE_PATH}"
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        run_profiling()
    except Exception:
        sys.exit(1)
