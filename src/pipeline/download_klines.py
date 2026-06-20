"""
Bulk-download Binance spot 1-minute klines for ~3 years across all USDT pairs.

Source: official Binance historical data archive (data.binance.vision),
via the `binance_historical_data` package. No API key required.
"""

import datetime
import logging
import os
from dateutil.relativedelta import relativedelta
from binance_historical_data import BinanceDataDumper

# Import parameters and exclusion config from central src/config.py
from src.config import (
    RAW_KLINES_DIR,
    YEARS_OF_HISTORY,
    DATA_FREQUENCY,
    STABLECOIN_BASES,
    EXCLUDED_SUFFIXES,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = str(RAW_KLINES_DIR)


def is_excluded(ticker: str) -> bool:
    if not ticker.endswith("USDT"):
        return True
    base = ticker[: -len("USDT")]
    if any(base.endswith(sfx) for sfx in EXCLUDED_SUFFIXES):
        return True
    if base in STABLECOIN_BASES:
        return True
    return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=OUTPUT_DIR,
        asset_class="spot",
        data_type="klines",
        data_frequency=DATA_FREQUENCY,
    )

    # Build the exclusion list ourselves so we can apply the UP/DOWN +
    # stablecoin filter; the library only auto-filters to "*USDT".
    all_tickers = data_dumper.get_list_all_trading_pairs()
    usdt_tickers = [t for t in all_tickers if t.endswith("USDT")]
    tickers_to_exclude = [t for t in usdt_tickers if is_excluded(t)]

    logger.info(
        "Found %d USDT pairs total, excluding %d (leveraged tokens / stablecoin pairs)",
        len(usdt_tickers),
        len(tickers_to_exclude),
    )

    date_end = datetime.date.today() - relativedelta(
        days=1
    )  # yesterday (last complete day)
    date_start = date_end - relativedelta(years=YEARS_OF_HISTORY)

    data_dumper.dump_data(
        tickers=None,  # None -> library defaults to all *USDT pairs
        date_start=date_start,
        date_end=date_end,
        is_to_update_existing=False,
        tickers_to_exclude=tickers_to_exclude,
    )

    # Daily files that are already fully covered by a monthly file are
    # redundant - clean them up.
    data_dumper.delete_outdated_daily_results()

    report_size(OUTPUT_DIR)


def report_size(path: str) -> None:
    total_bytes = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total_bytes += os.path.getsize(os.path.join(dirpath, f))
    logger.info("Total downloaded size: %.2f GB", total_bytes / (1024**3))


if __name__ == "__main__":
    main()
