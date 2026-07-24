"""
Configuration parameters for the Big Data & Scalable Machine Learning project.

Defines repository paths, target symbols, prediction horizons, S3 credentials,
and toggles for local/cloud execution modes.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.utils.helpers import normalize_path_str

# Set up module-level logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path Configurations (Relative to this configuration file)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load environment variables from the root .env file
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Data file locations
RAW_KLINES_DIR = RAW_DATA_DIR / "binance_data"
SAMPLE_PARQUET_PATH = SAMPLE_DATA_DIR / "binance_sample.parquet"

# Documentation / Output paths
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_PROFILE_PATH = DOCS_DIR / "data_profile.md"


# Helper functions to load environment variables with warnings
def get_env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    if val is None:
        logger.warning(
            f"Environment variable '{key}' is missing. Using default fallback: '{default}'"
        )
        return default
    return val


def get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        logger.warning(
            f"Environment variable '{key}' is missing. Using default fallback: {default}"
        )
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(
            f"Environment variable '{key}' has invalid integer value '{val}'. "
            f"Using default fallback: {default}"
        )
        return default


def get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        logger.warning(
            f"Environment variable '{key}' is missing. Using default fallback: {default}"
        )
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning(
            f"Environment variable '{key}' has invalid float value '{val}'. "
            f"Using default fallback: {default}"
        )
        return default


# ---------------------------------------------------------------------------
# Execution Settings
# ---------------------------------------------------------------------------
# Modes: "local_sample" (default), "local_raw", "aws_hub"
EXECUTION_MODE = get_env_str("EXECUTION_MODE", "local_sample")

# ---------------------------------------------------------------------------
# Historical Data / Download Settings
# ---------------------------------------------------------------------------
YEARS_OF_HISTORY = get_env_int("YEARS_OF_HISTORY", 3)
DATA_FREQUENCY = get_env_str("DATA_FREQUENCY", "1m")

# ---------------------------------------------------------------------------
# Symbols & Sampling Selection
# ---------------------------------------------------------------------------
# Top 20 most liquid cryptocurrency USDT pairs
TOP_20_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LINKUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "AVAXUSDT",
    "SHIBUSDT",
    "ATOMUSDT",
    "XLMUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "FILUSDT",
]

# Exclusion sets for raw data download and preprocessing
STABLECOIN_BASES = {
    "USDC",
    "BUSD",
    "TUSD",
    "FDUSD",
    "DAI",
    "USDP",
    "PAX",
    "UST",
    "USTC",
}
EXCLUDED_SUFFIXES = ("UP", "DOWN")  # Leveraged tokens

# ---------------------------------------------------------------------------
# Machine Learning Task Parameters (Option A: Binary Direction Classification)
# ---------------------------------------------------------------------------
TARGET_SYMBOL = get_env_str("TARGET_SYMBOL", "BTCUSDT")
FUTURE_HORIZON = get_env_int("FUTURE_HORIZON", 15)  # N minutes ahead
TARGET_THRESHOLD = get_env_float("TARGET_THRESHOLD", 0.0)  # Return threshold for UP direction
TRAIN_SPLIT_DATE = get_env_str("TRAIN_SPLIT_DATE", "2024-01-01")

# Canonical feature columns (11 stationary price features + 5 order-flow/time features)
FEATURE_COLS = [
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
    "taker_buy_ratio",
    "volume_z30",
    "trades_z30",
    "hour_sin",
    "hour_cos",
]


# ---------------------------------------------------------------------------
# AWS S3 Hub-and-Spoke Configurations
# ---------------------------------------------------------------------------
AWS_REGION = get_env_str("AWS_DEFAULT_REGION", "us-east-1")
AWS_S3_BUCKET_NAME = get_env_str("AWS_S3_BUCKET_NAME", "dat204m-binance-bigdata-hub-sg")
AWS_S3_RAW_PREFIX = "raw/"
AWS_S3_SAMPLE_PREFIX = "sample/"

# S3 Client Configurations (Loaded from environment variables, avoiding hardcoding)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# ---------------------------------------------------------------------------
# Spark Settings
# ---------------------------------------------------------------------------
SPARK_EXECUTION_MODE = get_env_str("SPARK_EXECUTION_MODE", "local")
# Default Hadoop-AWS jar package dependencies for S3 compatibility
SPARK_JARS_PACKAGES = (
    "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"
)

# Optional: configure JAVA_HOME to force a specific Java runtime for Spark
JAVA_HOME = os.getenv("JAVA_HOME")


def configure_java_home() -> None:
    """Lazily configures JAVA_HOME and updates PATH for Spark."""
    if JAVA_HOME:
        os.environ["JAVA_HOME"] = JAVA_HOME
        java_bin = os.path.join(JAVA_HOME, "bin")
        if java_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Dynamic Execution Data Paths
# ---------------------------------------------------------------------------
# Dynamically resolves the active data path based on execution mode
if EXECUTION_MODE == "aws_hub":
    # In S3 hub execution, load directly from S3 sample prefix
    ACTIVE_DATA_PATH = f"s3://{AWS_S3_BUCKET_NAME}/{AWS_S3_SAMPLE_PREFIX}binance_sample.parquet"
elif EXECUTION_MODE == "local_raw":
    ACTIVE_DATA_PATH = str(RAW_KLINES_DIR)
else:
    ACTIVE_DATA_PATH = str(SAMPLE_PARQUET_PATH)


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable, type-safe pipeline configuration container.

    Enables clean dependency injection across data preprocessing, feature engineering,
    model training, and evaluation pipelines.
    """

    execution_mode: str = "local_sample"
    years_of_history: int = 3
    data_frequency: str = "1m"
    target_symbol: str = "BTCUSDT"
    future_horizon: int = 15
    target_threshold: float = 0.0
    train_split_date: str = "2024-01-01"
    val_split_date: str = "2024-07-01"
    aws_region: str = "us-east-1"
    aws_s3_bucket_name: str = "dat204m-binance-bigdata-hub-sg"
    aws_s3_raw_prefix: str = "raw/"
    aws_s3_sample_prefix: str = "sample/"
    spark_execution_mode: str = "local"
    spark_jars_packages: str = (
        "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"
    )

    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_klines_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "binance_data"
    )
    sample_parquet_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "sample" / "binance_sample.parquet"
    )
    docs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "docs")
    data_profile_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "docs" / "data_profile.md"
    )

    feature_cols: tuple[str, ...] = tuple(FEATURE_COLS)
    top_20_symbols: tuple[str, ...] = tuple(TOP_20_SYMBOLS)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Instantiates PipelineConfig reading parameters from environment variables."""
        return cls(
            execution_mode=get_env_str("EXECUTION_MODE", "local_sample"),
            years_of_history=get_env_int("YEARS_OF_HISTORY", 3),
            data_frequency=get_env_str("DATA_FREQUENCY", "1m"),
            target_symbol=get_env_str("TARGET_SYMBOL", "BTCUSDT"),
            future_horizon=get_env_int("FUTURE_HORIZON", 15),
            target_threshold=get_env_float("TARGET_THRESHOLD", 0.0),
            train_split_date=get_env_str("TRAIN_SPLIT_DATE", "2024-01-01"),
            aws_region=get_env_str("AWS_DEFAULT_REGION", "us-east-1"),
            aws_s3_bucket_name=get_env_str("AWS_S3_BUCKET_NAME", "dat204m-binance-bigdata-hub-sg"),
            spark_execution_mode=get_env_str("SPARK_EXECUTION_MODE", "local"),
        )

    @classmethod
    def for_testing(cls, **kwargs: Any) -> "PipelineConfig":
        """Instantiates isolated PipelineConfig with optional overrides for unit testing."""
        default_kwargs: dict[str, Any] = {
            "execution_mode": "local_sample",
            "target_symbol": "BTCUSDT",
            "future_horizon": 15,
        }
        default_kwargs.update(kwargs)
        return cls(**default_kwargs)

    @property
    def active_data_path(self) -> str:
        """Returns normalized active data path string according to execution mode."""
        if self.execution_mode == "aws_hub":
            return (
                f"s3://{self.aws_s3_bucket_name}/{self.aws_s3_sample_prefix}binance_sample.parquet"
            )
        elif self.execution_mode == "local_raw":
            return normalize_path_str(self.raw_klines_dir)
        else:
            return normalize_path_str(self.sample_parquet_path)
