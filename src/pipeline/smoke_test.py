import datetime
from binance_historical_data import BinanceDataDumper

if __name__ == "__main__":
    dumper = BinanceDataDumper(
        path_dir_where_to_dump="./data/raw/smoke_test",
        asset_class="spot",
        data_type="klines",
        data_frequency="1m",
    )
    dumper.dump_data(
        tickers=["BTCUSDT"],
        date_start=datetime.date(2024, 1, 1),
        date_end=datetime.date(2024, 1, 3),
    )
