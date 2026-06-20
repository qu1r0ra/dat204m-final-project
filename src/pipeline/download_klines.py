"""
Bulk-download Binance spot 1-minute klines for ~3 years across all USDT pairs.

Source: official Binance historical data archive (data.binance.vision),
via the `binance_historical_data` package. No API key required.

Sizing target: ~400 symbols x 3 years x 1m candles -> ~85-90 GB of extracted
CSV, comfortably clearing a 50 GB raw-data requirement.

Usage:
    pip install -r requirements.txt
    python download_klines.py

Safe to interrupt (Ctrl+C) and re-run: already-downloaded dates are skipped.
"""

import datetime
import logging
import os

from dateutil.relativedelta import relativedelta
from binance_historical_data import BinanceDataDumper

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = "./data/raw/binance_data"
YEARS_OF_HISTORY = 3
DATA_FREQUENCY = "1m"

# Pairs to drop from the default "all *USDT" universe:
#   - leveraged tokens (e.g. BTCUPUSDT / BTCDOWNUSDT) - synthetic, not spot price action
#   - stablecoin-vs-stablecoin pairs (e.g. USDCUSDT) - near-flat, not useful for ML features
STABLECOIN_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "PAX", "UST", "USTC"}


def is_excluded(ticker: str) -> bool:
    base = ticker[: -len("USDT")]
    if base.endswith("UP") or base.endswith("DOWN"):
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

    logging.info(
        "Found %d USDT pairs total, excluding %d (leveraged tokens / stablecoin pairs)",
        len(usdt_tickers),
        len(tickers_to_exclude),
    )

    date_end = datetime.date.today() - relativedelta(days=1)  # yesterday (last complete day)
    date_start = date_end - relativedelta(years=YEARS_OF_HISTORY)

    data_dumper.dump_data(
        tickers=None,  # None -> library defaults to all *USDT pairs
        date_start=date_start,
        date_end=date_end,
        is_to_update_existing=False,
        tickers_to_exclude=tickers_to_exclude,
    )

    # Daily files that are already fully covered by a monthly file are
    # redundant - clean them up (mainly relevant if you re-run this later
    # to pick up new data).
    data_dumper.delete_outdated_daily_results()

    report_size(OUTPUT_DIR)


def report_size(path: str) -> None:
    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total_bytes += os.path.getsize(os.path.join(dirpath, f))
    logging.info("Total downloaded size: %.2f GB", total_bytes / (1024 ** 3))


if __name__ == "__main__":
    main()
