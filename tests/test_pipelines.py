import pytest
from pathlib import Path
import duckdb
import src.config as config
from src.pipeline.preprocess import run_profiling
from src.pipeline.sample_generator import generate_sample


@pytest.fixture
def mock_pipeline_env(tmp_path, monkeypatch):
    mock_raw_dir = tmp_path / "binance_data"
    mock_profile_path = tmp_path / "data_profile.md"
    mock_parquet_path = tmp_path / "binance_sample.parquet"

    # Mock the config paths using monkeypatch
    monkeypatch.setattr(config, "RAW_KLINES_DIR", mock_raw_dir)
    monkeypatch.setattr(config, "DATA_PROFILE_PATH", mock_profile_path)
    monkeypatch.setattr(config, "SAMPLE_PARQUET_PATH", mock_parquet_path)
    monkeypatch.setattr(config, "TOP_20_SYMBOLS", ["BTCUSDT", "ETHUSDT"])

    # Create mock monthly kline directories
    btc_dir = mock_raw_dir / "spot" / "monthly" / "klines" / "BTCUSDT" / "1m"
    eth_dir = mock_raw_dir / "spot" / "monthly" / "klines" / "ETHUSDT" / "1m"

    btc_dir.mkdir(parents=True, exist_ok=True)
    eth_dir.mkdir(parents=True, exist_ok=True)

    # Write dummy CSV data for BTCUSDT
    btc_csv = btc_dir / "BTCUSDT-1m-2024-01.csv"
    with open(btc_csv, "w") as f:
        f.write(
            "1704067200000,42000.0,42100.0,41900.0,42050.0,10.0,1704067259999,420000.0,100,5.0,210000.0,0\n"
        )
        f.write(
            "1704067260000,42050.0,42200.0,42000.0,42150.0,12.0,1704067319999,505000.0,120,6.0,252500.0,0\n"
        )

    # Write dummy CSV data for ETHUSDT
    eth_csv = eth_dir / "ETHUSDT-1m-2024-01.csv"
    with open(eth_csv, "w") as f:
        f.write(
            "1704067200000,2200.0,2210.0,2190.0,2205.0,50.0,1704067259999,110000.0,80,25.0,55000.0,0\n"
        )
        f.write(
            "1704067260000,2205.0,2220.0,2200.0,2215.0,60.0,1704067319999,132600.0,90,30.0,66300.0,0\n"
        )

    return {
        "raw_dir": mock_raw_dir,
        "profile_path": mock_profile_path,
        "parquet_path": mock_parquet_path,
    }


def test_config_loader():
    # Verify that path variables exist and are path objects
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.DATA_DIR.name == "data"


def test_run_profiling(mock_pipeline_env):
    # Run profiling on the mock directory
    run_profiling()

    profile_path = mock_pipeline_env["profile_path"]
    # Check if report was written
    assert profile_path.exists()

    # Verify report content
    with open(profile_path, "r") as f:
        content = f.read()
        assert "Dataset Profile Report" in content
        assert "BTCUSDT" in content
        assert "ETHUSDT" in content
        assert "Total Unique Symbols**: 2" in content
        assert "Total Rows (Observations)**: 4" in content


def test_generate_sample(mock_pipeline_env):
    # Run sample generation
    generate_sample()

    parquet_path = mock_pipeline_env["parquet_path"]
    # Check if Parquet was created
    assert parquet_path.exists()

    # Load Parquet back using DuckDB to inspect the contents
    parquet_path_str = str(parquet_path).replace("\\", "/")
    with duckdb.connect() as con:
        df = con.execute(f"SELECT * FROM read_parquet('{parquet_path_str}')").df()

    # Check schema and content
    assert len(df) == 4
    assert list(df.columns) == [
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
    ]

    # Assert data values are correct and symbols match
    btc_rows = df[df["symbol"] == "BTCUSDT"]
    eth_rows = df[df["symbol"] == "ETHUSDT"]

    assert len(btc_rows) == 2
    assert len(eth_rows) == 2

    assert btc_rows["open"].iloc[0] == 42000.0
    assert eth_rows["close"].iloc[1] == 2215.0
