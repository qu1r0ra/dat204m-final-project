"""
Configuration parameters for the Big Data & Scalable Machine Learning project.

Defines repository paths, target symbols, prediction horizons, S3 credentials,
and toggles for local/cloud execution modes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path Configurations (Relative to this configuration file)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load environment variables from the root .env file
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Data file locations
RAW_KLINES_DIR = RAW_DATA_DIR / "binance_data"
SAMPLE_PARQUET_PATH = SAMPLE_DATA_DIR / "binance_sample.parquet"

# Documentation / Output paths
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_PROFILE_PATH = DOCS_DIR / "data_profile.md"

# ---------------------------------------------------------------------------
# Execution Settings
# ---------------------------------------------------------------------------
# Modes: "local_sample" (default), "local_raw", "aws_hub"
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "local_sample")

# ---------------------------------------------------------------------------
# Historical Data / Download Settings
# ---------------------------------------------------------------------------
YEARS_OF_HISTORY = int(os.getenv("YEARS_OF_HISTORY", "3"))
DATA_FREQUENCY = os.getenv("DATA_FREQUENCY", "1m")

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
TARGET_SYMBOL = os.getenv("TARGET_SYMBOL", "BTCUSDT")
FUTURE_HORIZON = int(os.getenv("FUTURE_HORIZON", "15"))  # N minutes ahead
TARGET_THRESHOLD = float(
    os.getenv("TARGET_THRESHOLD", "0.0")
)  # Return threshold for UP direction
TRAIN_SPLIT_DATE = os.getenv("TRAIN_SPLIT_DATE", "2024-01-01")

# ---------------------------------------------------------------------------
# AWS S3 Hub-and-Spoke Configurations
# ---------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "dat204m-binance-bigdata-hub")
AWS_S3_RAW_PREFIX = "raw/"
AWS_S3_SAMPLE_PREFIX = "sample/"

# S3 Client Configurations (Loaded from environment variables, avoiding hardcoding)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
