# Data Registry

Registry tracking datasets, files, formats, schemas, and S3 keys.

---

## 1. Raw Dataset (Full historical Binance Spot 1m K-Lines)

- **Source**: Ingested via `download_klines.py` from Binance Historical Data archive.
- **Estimated Size**: ~75+ GB uncompressed (CSVs).
- **Local Directory**: `data/raw/binance_data/spot/monthly/klines/` (git-ignored).
- **S3 Prefix**: `s3://dat204m-binance-bigdata-hub-sg/raw/`
- **Normalization**: Older monthly files using 13-digit millisecond timestamps are normalized alongside newer 16-digit microsecond files.

---

## 2. Sample Dataset (Downsampled Parquet)

- **Source**: Extracted using DuckDB from raw CSV files.
- **Filtering Criteria**: Top 20 most liquid cryptocurrency USDT pairs at 1-minute frequency.
- **Format**: Parquet, compressed with `zstd`.
- **Local Location**: `data/sample/binance_sample.parquet` (git-ignored).
- **S3 Prefix**: `s3://dat204m-binance-bigdata-hub-sg/sample/binance_sample.parquet`

---

## 3. Schema Specifications

Both datasets share the following column schema:

| Column                         | Type      | Description                               |
| ------------------------------ | --------- | ----------------------------------------- |
| `open_time`                    | TIMESTAMP | Period start time (UTC)                   |
| `open`                         | DOUBLE    | Period open price                         |
| `high`                         | DOUBLE    | Period high price                         |
| `low`                          | DOUBLE    | Period low price                          |
| `close`                        | DOUBLE    | Period close price                        |
| `volume`                       | DOUBLE    | Base asset volume traded                  |
| `close_time`                   | TIMESTAMP | Period close time (UTC)                   |
| `quote_asset_volume`           | DOUBLE    | Quote asset (USDT) volume traded          |
| `number_of_trades`             | INTEGER   | Total count of trades in period           |
| `taker_buy_base_asset_volume`  | DOUBLE    | Base asset volume bought by takers        |
| `taker_buy_quote_asset_volume` | DOUBLE    | Quote asset volume bought by takers       |
| `symbol`                       | VARCHAR   | Cryptocurrency pair symbol (e.g. BTCUSDT) |
