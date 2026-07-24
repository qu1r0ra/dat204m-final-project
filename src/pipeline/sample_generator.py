"""
Sample Parquet Generator Pipeline.

Slices out the top 20 most liquid cryptocurrency USDT pairs at the 1-minute frequency,
cleans column datatypes, sorts, and exports a compressed Parquet file to
data/sample/binance_sample.parquet.
"""

import logging
import os

import duckdb

import src.config as config
from src.exceptions import ConfigurationError
from src.pipeline.schemas import get_duckdb_epoch_ms_sql
from src.utils.helpers import discover_symbol_csvs, normalize_path_str

# Configure logging
logger = logging.getLogger(__name__)


def generate_sample() -> None:
    logger.info("Starting sample Parquet generation...")

    base_dir = config.RAW_KLINES_DIR
    valid_paths = discover_symbol_csvs(config.TOP_20_SYMBOLS, base_dir)

    if valid_paths:
        logger.info(f"Adding {len(valid_paths)} files to sample.")

    if not valid_paths:
        logger.error("No valid CSV files found for any of the configured TOP_20_SYMBOLS.")
        logger.error("Please run the klines downloader script or verify that raw datasets exist.")
        raise ConfigurationError("No valid raw CSV files found for top 20 symbols.")

    # Output file setup
    config.SAMPLE_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Format list of valid paths into SQL string array format: ['path1', 'path2']
    paths_str = ", ".join("'" + normalize_path_str(p) + "'" for p in valid_paths)
    read_csv_src = f"[{paths_str}]"
    parquet_path_str = normalize_path_str(config.SAMPLE_PARQUET_PATH)

    col0_epoch = get_duckdb_epoch_ms_sql("column00")
    col6_epoch = get_duckdb_epoch_ms_sql("column06")

    # Prepare export query. We use standard zstd compression for high efficiency.
    export_query = f"""
        COPY (
            SELECT
                {col0_epoch} AS open_time,
                column01::DOUBLE AS open,
                column02::DOUBLE AS high,
                column03::DOUBLE AS low,
                column04::DOUBLE AS close,
                column05::DOUBLE AS volume,
                {col6_epoch} AS close_time,
                column07::DOUBLE AS quote_asset_volume,
                column08::INTEGER AS number_of_trades,
                column09::DOUBLE AS taker_buy_base_asset_volume,
                column10::DOUBLE AS taker_buy_quote_asset_volume,
                regexp_extract(filename, '([^/\\\\]+)-1m-', 1) AS symbol
            FROM read_csv({read_csv_src}, header=False, filename=True)
            ORDER BY symbol, open_time
        ) TO '{parquet_path_str}' (FORMAT 'parquet', COMPRESSION 'zstd')
    """

    try:
        with duckdb.connect(database=":memory:") as con:
            logger.info(f"Aggregating and exporting {len(valid_paths)} symbol paths to Parquet...")
            con.execute(export_query)
            logger.info(
                f"Sample Parquet file generated successfully at: {config.SAMPLE_PARQUET_PATH}"
            )

            # Verify the generated file size and count
            count_res = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT symbol) FROM read_parquet('{parquet_path_str}')"
            ).fetchone()
            if count_res:
                logger.info(
                    f"Verified Parquet file contains {count_res[0]:,} rows "
                    f"across {count_res[1]} unique symbols."
                )
            else:
                logger.warning("Could not verify Parquet file contents (empty result).")

            # Report size
            file_size_mb = os.path.getsize(config.SAMPLE_PARQUET_PATH) / (1024 * 1024)
            logger.info(f"Parquet file size: {file_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Failed to generate sample Parquet: {e}")
        raise


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        generate_sample()
    except Exception:
        sys.exit(1)
