import sys
import os
import pytest
import shutil
from pathlib import Path

# Point PySpark to the active virtual env Python executable to bypass Windows Microsoft Store alias
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# JVM options required on Java 11+ to allow Spark to access jdk.internal packages

java_options = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"
)
os.environ["JDK_JAVA_OPTIONS"] = java_options
os.environ["JAVA_TOOL_OPTIONS"] = java_options


pytestmark = [pytest.mark.slow, pytest.mark.spark]


import src.config as config
from pyspark.sql import SparkSession

from src.pipeline.preprocess_spark import run_profiling
from src.pipeline.sample_generator_spark import generate_sample
from src.features.indicators_spark import compute_indicators_spark
from src.models.train_spark import (
    compute_targets_spark,
    split_data_chronologically_spark,
    train_pipeline_spark,
)


@pytest.fixture(scope="module")
def spark_session():
    """Module-level SparkSession for fast testing."""
    if os.name == "nt":
        hadoop_dir = Path("data/hadoop").resolve()
        bin_dir = hadoop_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        winutils_exe = bin_dir / "winutils.exe"
        if not winutils_exe.exists():
            attrib_exe = (
                Path(os.environ.get("SystemRoot", "C:\\Windows"))
                / "System32"
                / "attrib.exe"
            )
            if attrib_exe.exists():
                try:
                    shutil.copy(str(attrib_exe), str(winutils_exe))
                except Exception:
                    pass
        os.environ["HADOOP_HOME"] = str(hadoop_dir)

        # Mock PySpark's Parquet read/write on Windows to bypass Hadoop/winutils DLL issues
        from pyspark.sql import DataFrameWriter, DataFrameReader
        import pandas as pd
        from urllib.parse import urlparse
        import urllib.request

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
                df_pandas = pd.concat(
                    [pd.read_parquet(f) for f in files], ignore_index=True
                )
            else:
                df_pandas = pd.read_parquet(p_str)
            return self._spark.createDataFrame(df_pandas)

        DataFrameWriter.parquet = mock_parquet_write
        DataFrameReader.parquet = mock_parquet_read

    java_options = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"
    )
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("TestSparkPipelines")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", java_options)
        .config("spark.executor.extraJavaOptions", java_options)
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

    # Write 60 mock rows for BTCUSDT and ETHUSDT to satisfy indicator window constraints
    # (e.g. sma_50 requires at least 50 periods)
    btc_csv = btc_dir / "BTCUSDT-1m-2024-01.csv"
    with open(btc_csv, "w") as f:
        base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
        for i in range(65):
            t = base_time + i * 60000
            price = 42000.0 + i * 10.0
            f.write(
                f"{t},{price},{price+5},{price-5},{price},10.0,{t+59999},420000.0,100,5.0,210000.0,0\n"
            )

    eth_csv = eth_dir / "ETHUSDT-1m-2024-01.csv"
    with open(eth_csv, "w") as f:
        base_time = 1704067200000
        for i in range(65):
            t = base_time + i * 60000
            price = 2200.0 + i * 2.0
            f.write(
                f"{t},{price},{price+1},{price-1},{price},50.0,{t+59999},110000.0,80,25.0,55000.0,0\n"
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

    with open(report_path, "r") as f:
        content = f.read()
        assert "Dataset Profile Report (Spark Edition)" in content
        assert "BTCUSDT" in content
        assert "ETHUSDT" in content


def test_spark_sample_generator(mock_spark_env, spark_session, monkeypatch):
    # Monkeypatch Spark session getter in sample generator
    from src.pipeline import sample_generator_spark

    monkeypatch.setattr(
        sample_generator_spark, "get_spark_session", lambda: spark_session
    )

    generate_sample()

    parquet_path = mock_spark_env["sample_dir"] / "binance_sample_spark.parquet"
    assert parquet_path.exists()


def test_spark_indicators_and_ml(mock_spark_env, spark_session, monkeypatch):
    # Monkeypatch Spark session getter in sample generator
    from src.pipeline import sample_generator_spark

    monkeypatch.setattr(
        sample_generator_spark, "get_spark_session", lambda: spark_session
    )

    # 1. Run sample generator to produce the test parquet file
    generate_sample()
    parquet_path_str = str(
        mock_spark_env["sample_dir"] / "binance_sample_spark.parquet"
    ).replace("\\", "/")

    # 2. Read back using Spark
    df = spark_session.read.parquet(parquet_path_str)
    assert df.count() == 130  # 65 rows * 2 symbols

    # 3. Compute indicators
    df_features = compute_indicators_spark(df)
    assert "rsi_14" in df_features.columns
    assert "bb_position" in df_features.columns
    assert "close_to_sma_50" in df_features.columns

    # 4. Compute ML targets
    df_labeled = compute_targets_spark(df_features)
    assert "target" in df_labeled.columns

    # We should have rows remaining after warmups are dropped (65 - 50 = 15 rows per symbol)
    total_labeled = df_labeled.count()
    assert total_labeled > 0

    # 5. Split chronologically
    # Our mock dates are all 2024-01-01. Split before/after mid-day to verify partitions.
    train_df, val_df, test_df = split_data_chronologically_spark(
        df_labeled, train_end="2024-01-01 00:30:00", val_end="2024-01-01 00:50:00"
    )

    # 6. Train pipelines
    feature_cols = [
        "close_to_sma_15",
        "close_to_sma_50",
        "close_to_ema_15",
        "close_to_ema_50",
        "bb_position",
        "macd_line_norm",
        "macd_signal_norm",
        "macd_hist_norm",
        "volatility_30",
        "rsi_14",
        "log_return",
    ]

    trained_dict = train_pipeline_spark(train_df, val_df, feature_cols)
    assert "logistic_regression" in trained_dict
    assert "random_forest" in trained_dict
    assert "metrics" in trained_dict
