import os
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

import src.config as config
from src.features.indicators_spark import compute_indicators_spark
from src.models.train_spark import (
    compute_targets_spark,
    split_data_chronologically_spark,
    train_pipeline_spark,
)
from src.pipeline.preprocess_spark import run_profiling
from src.pipeline.sample_generator_spark import generate_sample

# JVM options required on Java 11+ to allow Spark to access jdk.internal packages
from src.utils.spark_client import JAVA_OPTIONS

pytestmark = [pytest.mark.slow, pytest.mark.spark]


@pytest.fixture(scope="module")
def spark_session():
    """Module-level SparkSession for fast testing."""
    from src.utils.spark_client import setup_spark_env

    setup_spark_env()

    if os.name == "nt":
        from src.utils.spark_client import ensure_hadoop_home

        ensure_hadoop_home()

    # Mock PySpark's Parquet read/write on Windows to bypass Hadoop/winutils DLL issues
    import urllib.request
    from urllib.parse import urlparse

    import pandas as pd
    from pyspark.sql import DataFrameReader, DataFrameWriter

    def to_local_path(path_str):
        p_str = str(path_str)
        if p_str.startswith("file:"):
            # Handle file URIs correctly
            parsed = urlparse(p_str)
            return urllib.request.url2pathname(parsed.path)
        return p_str

    def mock_parquet_write(self, path, *args, **kwargs):
        p_str = to_local_path(path)
        df_pandas = self._df.toPandas()
        Path(p_str).mkdir(parents=True, exist_ok=True)
        df_pandas.to_parquet(Path(p_str) / "data.parquet", index=False)

    def mock_parquet_read(self, path, *args, **kwargs):
        p_str = to_local_path(path)
        p = Path(p_str)
        if p.is_dir():
            files = list(p.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files in {p_str}")
            df_pandas = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        else:
            df_pandas = pd.read_parquet(p_str)
        from pyspark.sql import SparkSession

        return SparkSession.builder.getOrCreate().createDataFrame(df_pandas)

    DataFrameWriter.parquet = mock_parquet_write
    DataFrameReader.parquet = mock_parquet_read

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("TestSparkPipelines")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", JAVA_OPTIONS)
        .config("spark.executor.extraJavaOptions", JAVA_OPTIONS)
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def mock_spark_env(tmp_path, monkeypatch):
    mock_raw_dir = tmp_path / "binance_data"
    mock_docs_dir = tmp_path / "docs"
    mock_sample_dir = tmp_path / "sample"

    # Monkeypatch config paths to use our temp paths
    monkeypatch.setattr(config, "RAW_KLINES_DIR", mock_raw_dir)
    monkeypatch.setattr(config, "DOCS_DIR", mock_docs_dir)
    monkeypatch.setattr(config, "SAMPLE_DATA_DIR", mock_sample_dir)
    monkeypatch.setattr(config, "TOP_20_SYMBOLS", ["BTCUSDT", "ETHUSDT"])

    btc_dir = mock_raw_dir / "spot" / "monthly" / "klines" / "BTCUSDT" / "1m"
    eth_dir = mock_raw_dir / "spot" / "monthly" / "klines" / "ETHUSDT" / "1m"

    btc_dir.mkdir(parents=True, exist_ok=True)
    eth_dir.mkdir(parents=True, exist_ok=True)

    # Write 100 mock rows for BTCUSDT and ETHUSDT to satisfy indicator window constraints
    # (e.g. sma_50 requires at least 50 periods)
    btc_csv = btc_dir / "BTCUSDT-1m-2024-01.csv"
    with open(btc_csv, "w") as f:
        base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
        for i in range(100):
            t = base_time + i * 60000
            price = 42000.0 + i * 10.0
            f.write(
                f"{t},{price},{price + 5},{price - 5},{price},10.0,"
                f"{t + 59999},420000.0,100,5.0,210000.0,0\n"
            )

    eth_csv = eth_dir / "ETHUSDT-1m-2024-01.csv"
    with open(eth_csv, "w") as f:
        base_time = 1704067200000
        for i in range(100):
            t = base_time + i * 60000
            price = 2200.0 + i * 2.0
            f.write(
                f"{t},{price},{price + 1},{price - 1},{price},50.0,"
                f"{t + 59999},110000.0,80,25.0,55000.0,0\n"
            )

    return {
        "raw_dir": mock_raw_dir,
        "docs_dir": mock_docs_dir,
        "sample_dir": mock_sample_dir,
    }


def test_spark_profiling(mock_spark_env, spark_session, monkeypatch):
    # Monkeypatch the Spark session getter to reuse the test one
    from src.pipeline import preprocess_spark

    monkeypatch.setattr(preprocess_spark, "get_spark_session", lambda: spark_session)

    run_profiling()

    report_path = mock_spark_env["docs_dir"] / "data_profile_spark.md"
    assert report_path.exists()

    with open(report_path) as f:
        content = f.read()
        assert "Dataset Profile Report (Spark Edition)" in content
        assert "BTCUSDT" in content
        assert "ETHUSDT" in content


def test_spark_sample_generator(mock_spark_env, spark_session, monkeypatch):
    # Monkeypatch Spark session getter in sample generator
    from src.pipeline import sample_generator_spark

    monkeypatch.setattr(sample_generator_spark, "get_spark_session", lambda: spark_session)

    generate_sample()

    parquet_path = mock_spark_env["sample_dir"] / "binance_sample_spark.parquet"
    assert parquet_path.exists()


def test_spark_indicators_and_ml(mock_spark_env, spark_session, monkeypatch):
    # Monkeypatch Spark session getter in sample generator
    from src.pipeline import sample_generator_spark

    monkeypatch.setattr(sample_generator_spark, "get_spark_session", lambda: spark_session)

    # 1. Run sample generator to produce the test parquet file
    generate_sample()
    parquet_path_str = str(mock_spark_env["sample_dir"] / "binance_sample_spark.parquet").replace(
        "\\", "/"
    )

    # 2. Read back using Spark
    df = spark_session.read.parquet(parquet_path_str)
    assert df.count() == 200  # 100 rows * 2 symbols

    # 3. Compute indicators
    df_features = compute_indicators_spark(df)
    assert "rsi_14" in df_features.columns
    assert "bb_position" in df_features.columns
    assert "close_to_sma_50" in df_features.columns

    # 4. Compute ML targets
    df_labeled = compute_targets_spark(df_features)
    assert "target" in df_labeled.columns

    # We should have rows remaining after warmups are dropped (100 - 49 - 15 = 36 rows per symbol)
    total_labeled = df_labeled.count()
    assert total_labeled > 0

    # 5. Split chronologically using dynamically calculated timestamp bounds
    # to bypass local timezone offsets
    import datetime

    from pyspark.sql import functions as F

    times = df_labeled.select(F.min("open_time"), F.max("open_time")).collect()[0]
    min_time = times[0]
    max_time = times[1]

    # Calculate timezone-independent epoch bounds
    min_epoch = min_time.timestamp()
    max_epoch = max_time.timestamp()
    delta_epoch = max_epoch - min_epoch

    train_end_epoch = min_epoch + delta_epoch * 0.5
    val_end_epoch = min_epoch + delta_epoch * 0.75

    train_end_dt = datetime.datetime.fromtimestamp(train_end_epoch, datetime.UTC)
    val_end_dt = datetime.datetime.fromtimestamp(val_end_epoch, datetime.UTC)

    train_end_str = train_end_dt.strftime("%Y-%m-%d %H:%M:%S")
    val_end_str = val_end_dt.strftime("%Y-%m-%d %H:%M:%S")

    train_df, val_df, _ = split_data_chronologically_spark(
        df_labeled, train_end=train_end_str, val_end=val_end_str
    )

    # 6. Train pipelines
    from src.models.train_spark import DEFAULT_FEATURE_COLS

    feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in train_df.columns]

    trained_artifacts = train_pipeline_spark(train_df, val_df, feature_cols)

    assert trained_artifacts.logistic_regression is not None
    assert trained_artifacts.random_forest is not None
    assert trained_artifacts.metrics is not None
