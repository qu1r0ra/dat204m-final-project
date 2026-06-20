import os
import shutil
import tempfile
import unittest
from pathlib import Path
import duckdb
import pandas as pd

# Import the modules we want to test
import src.config as config
from src.pipeline.preprocess import run_profiling
from src.pipeline.sample_generator import generate_sample


class TestPipelines(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for our mock data
        self.test_dir = Path(tempfile.mkdtemp())
        self.mock_raw_dir = self.test_dir / "binance_data"
        self.mock_profile_path = self.test_dir / "data_profile.md"
        self.mock_parquet_path = self.test_dir / "binance_sample.parquet"

        # Mock the config paths
        self.original_raw_klines_dir = config.RAW_KLINES_DIR
        self.original_data_profile_path = config.DATA_PROFILE_PATH
        self.original_sample_parquet_path = config.SAMPLE_PARQUET_PATH
        self.original_top_20 = config.TOP_20_SYMBOLS

        config.RAW_KLINES_DIR = self.mock_raw_dir
        config.DATA_PROFILE_PATH = self.mock_profile_path
        config.SAMPLE_PARQUET_PATH = self.mock_parquet_path
        config.TOP_20_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

        # Create mock monthly kline directories
        self.btc_dir = (
            self.mock_raw_dir / "spot" / "monthly" / "klines" / "BTCUSDT" / "1m"
        )
        self.eth_dir = (
            self.mock_raw_dir / "spot" / "monthly" / "klines" / "ETHUSDT" / "1m"
        )

        self.btc_dir.mkdir(parents=True, exist_ok=True)
        self.eth_dir.mkdir(parents=True, exist_ok=True)

        # Write dummy CSV data for BTCUSDT
        # columns: open_time, open, high, low, close, volume, close_time, quote_vol, trades, buy_base, buy_quote, ignore
        self.btc_csv = self.btc_dir / "BTCUSDT-1m-2024-01.csv"
        with open(self.btc_csv, "w") as f:
            f.write(
                "1704067200000,42000.0,42100.0,41900.0,42050.0,10.0,1704067259999,420000.0,100,5.0,210000.0,0\n"
            )
            f.write(
                "1704067260000,42050.0,42200.0,42000.0,42150.0,12.0,1704067319999,505000.0,120,6.0,252500.0,0\n"
            )

        # Write dummy CSV data for ETHUSDT
        self.eth_csv = self.eth_dir / "ETHUSDT-1m-2024-01.csv"
        with open(self.eth_csv, "w") as f:
            f.write(
                "1704067200000,2200.0,2210.0,2190.0,2205.0,50.0,1704067259999,110000.0,80,25.0,55000.0,0\n"
            )
            f.write(
                "1704067260000,2205.0,2220.0,2200.0,2215.0,60.0,1704067319999,132600.0,90,30.0,66300.0,0\n"
            )

    def tearDown(self):
        # Clean up the mock directories
        shutil.rmtree(self.test_dir)

        # Restore the original config paths
        config.RAW_KLINES_DIR = self.original_raw_klines_dir
        config.DATA_PROFILE_PATH = self.original_data_profile_path
        config.SAMPLE_PARQUET_PATH = self.original_sample_parquet_path
        config.TOP_20_SYMBOLS = self.original_top_20

    def test_config_loader(self):
        # Verify that path variables exist and are path objects
        self.assertIsInstance(config.PROJECT_ROOT, Path)
        self.assertTrue(config.DATA_DIR.name == "data")

    def test_run_profiling(self):
        # Run profiling on the mock directory
        run_profiling()

        # Check if report was written
        self.assertTrue(self.mock_profile_path.exists())

        # Verify report content
        with open(self.mock_profile_path, "r") as f:
            content = f.read()
            self.assertIn("Dataset Profile Report", content)
            self.assertIn("BTCUSDT", content)
            self.assertIn("ETHUSDT", content)
            self.assertIn("Total Unique Symbols**: 2", content)
            self.assertIn("Total Rows (Observations)**: 4", content)

    def test_generate_sample(self):
        # Run sample generation
        generate_sample()

        # Check if Parquet was created
        self.assertTrue(self.mock_parquet_path.exists())

        # Load Parquet back using DuckDB to inspect the contents
        con = duckdb.connect()
        parquet_path_str = str(self.mock_parquet_path).replace("\\", "/")
        df = con.execute(f"SELECT * FROM read_parquet('{parquet_path_str}')").df()

        # Check schema and content
        self.assertEqual(len(df), 4)
        self.assertListEqual(
            list(df.columns),
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "symbol",
            ],
        )

        # Assert data values are correct and symbols match
        btc_rows = df[df["symbol"] == "BTCUSDT"]
        eth_rows = df[df["symbol"] == "ETHUSDT"]

        self.assertEqual(len(btc_rows), 2)
        self.assertEqual(len(eth_rows), 2)

        self.assertEqual(btc_rows["open"].iloc[0], 42000.0)
        self.assertEqual(eth_rows["close"].iloc[1], 2215.0)


if __name__ == "__main__":
    unittest.main()
