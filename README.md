# dat204m-final-project <!-- omit from toc -->

<!-- Refer to <https://shields.io/badges> for usage -->

![Term Course](https://img.shields.io/badge/AY2526--T3-DAT204M-blue) ![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white) ![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white) ![duckdb](https://img.shields.io/badge/DuckDB-FFF000?logo=duckdb&logoColor=black) ![polars](https://img.shields.io/badge/Polars-CD7F32?logo=polars&logoColor=white) ![pyspark](https://img.shields.io/badge/PySpark-E25A28?logo=apachespark&logoColor=white)

An end-to-end data pipeline and machine learning pipeline for descriptive and predictive analytics on historical Binance Spot 1-Minute K-Lines (~75+ GB raw CSVs, ~611 million rows, 558 trading pairs).

## Table of Contents <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. Project Structure](#2-project-structure)
- [3. Getting Started](#3-getting-started)
  - [3.1. Technical Prerequisites](#31-technical-prerequisites)
  - [3.2. Configuration Setup](#32-configuration-setup)
  - [3.3. Installation](#33-installation)
- [4. Execution Pipeline](#4-execution-pipeline)
  - [4.1. Downloading Raw Data](#41-downloading-raw-data)
  - [4.2. Running Profiling](#42-running-profiling)
  - [4.3. Generating Downsampled Parquet](#43-generating-downsampled-parquet)
  - [4.4. Running Tests](#44-running-tests)
- [5. Reproducing Analytical Results](#5-reproducing-analytical-results)
- [6. AWS Hub-and-Spoke Architecture](#6-aws-hub-and-spoke-architecture)

## 1. Introduction

This project processes and analyzes Binance Spot 1-Minute K-line data to solve a binary classification problem: **Binary Price Direction Prediction** (Option A). The model predicts whether the price return of a given cryptocurrency ticker (e.g., `BTCUSDT`) will move up or down over a future horizon $N$.

The project is structured in two main phases:

1. **Phase 1 (Descriptive Analytics):** Downloading, deduplicating, and profiling the massive raw dataset (~75+ GB CSVs) using DuckDB (or PySpark), generating a local downsampled Parquet dataset (~1.5-2 GB) of the top 20 most liquid cryptocurrency pairs, and conducting exploratory data analysis.
2. **Phase 2 (Predictive Analytics):** Extracting rolling technical indicators using Polars (or PySpark), training classification models, evaluating their performance against a baseline, and performing error analysis.

## 2. Project Structure

A high-level overview of the repository organization:

```text
.
├── .cursor/                  # Cursor workspace configurations and rules
│   ├── project/              # Data and model registry definitions
│   └── rules/                # Canonical domain rules (.mdc files)
├── aws/                      # AWS infrastructure scripts and configurations
│   └── s3_bucket_policy.json # Cross-account S3 bucket policy template
├── data/                     # Git-ignored local data directory
│   ├── raw/                  # Downloaded raw monthly CSV files
│   └── sample/               # Local compressed sample Parquet files
├── docs/                     # Written deliverables, reports, and documentation
│   ├── specs.md              # Project specifications
│   ├── team_roles.md         # Team roles and task dissemination
│   ├── data_profile.md       # Auto-generated dataset profiling report (DuckDB)
│   └── data_profile_spark.md # Auto-generated dataset profiling report (Spark)
├── notebooks/                # Jupyter Notebooks for deliverables
│   ├── 01_eda_descriptive_analytics.ipynb         # Phase 1: Descriptive profiling & visualizations
│   ├── 02_ml_feature_engineering_training.ipynb   # Phase 2: Signal features & model training
│   └── 03_ml_evaluation_error_analysis.ipynb      # Phase 2: Evaluation metrics & recommendations
├── src/                      # Source package
│   ├── __init__.py
│   ├── config.py             # Configuration parameters and environment variables loader
│   ├── features/             # Feature engineering and signal generation
│   │   ├── __init__.py
│   │   ├── indicators.py     # Rolling Polars indicators
│   │   └── indicators_spark.py # UDF-based Spark feature indicators
│   ├── models/               # Model definitions, training, and evaluation pipelines
│   │   ├── __init__.py
│   │   ├── train.py          # Scikit-learn training pipeline
│   │   └── train_spark.py    # PySpark MLlib training pipeline
│   ├── pipeline/             # Data preprocessing and ingestion pipelines
│   │   ├── __init__.py
│   │   ├── download_klines.py # Ingestion script for historical data
│   │   ├── preprocess.py      # Profiling and data cleaning script (DuckDB)
│   │   ├── preprocess_spark.py # Profiling and data cleaning script (Spark)
│   │   ├── sample_generator.py # Downsampling Parquet generator (DuckDB)
│   │   ├── sample_generator_spark.py # Downsampling Parquet generator (Spark)
│   │   └── schemas.py         # Shared PySpark schemas
│   └── utils/                # General utility modules
│       ├── __init__.py
│       ├── aws_client.py     # General AWS client helpers
│       ├── helpers.py        # Shared utility helper methods
│       └── spark_client.py   # Unified Spark Session configuration
├── tests/                    # pytest suite for validation
│   ├── __init__.py
│   ├── conftest.py           # Shared test fixtures and configuration
│   ├── test_pipelines.py     # Automated tests for ingestion and processing
│   └── test_spark_pipelines.py # Spark-specific test suite
├── AGENTS.md                 # Agent entrypoint and rules index
└── HANDOFF.md                # Workspace living handoff
```

## 3. Getting Started

### 3.1. Technical Prerequisites

Ensure you have the following installed on your local machine:

1. **Git:** Used to clone the repository.
2. **Python >=3.11**
3. **uv:** Fast Python package installer and project manager. Installation details: [Astral uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).
4. **Java JDK 21:** Required to run PySpark locally.

### 3.2. Configuration Setup

Copy the template environment file and customize it for your local environment:

```bash
cp .env.example .env
```

If configurations (like `YEARS_OF_HISTORY`, `TARGET_SYMBOL`, or AWS S3 parameters) are omitted from the `.env` file, the configuration loader will notify you and fallback to the project's safe default parameters.

### 3.3. Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/qu1r0ra/dat204m-final-project
   ```

2. Navigate to the project root and synchronize dependencies:

   ```bash
   cd dat204m-final-project
   uv sync
   ```

## 4. Execution Pipeline

### 4.1. Downloading Raw Data

Download the historical Binance spot 1-minute klines for the target history:

```bash
uv run python -m src.pipeline.download_klines
```

### 4.2. Running Profiling

Profile the downloaded raw CSV datasets, checking for gaps, nulls, and duplicate timestamps.

**DuckDB version** (generates [docs/data_profile.md](docs/data_profile.md)):

```bash
uv run python -m src.pipeline.preprocess
```

**PySpark version** (generates [docs/data_profile_spark.md](docs/data_profile_spark.md)):

```bash
uv run python -m src.pipeline.preprocess_spark
```

### 4.3. Generating Downsampled Parquet

Slice out data for the top 20 most liquid trading pairs and export a compressed Parquet sample to `data/sample/binance_sample.parquet`:

**DuckDB version**:

```bash
uv run python -m src.pipeline.sample_generator
```

**PySpark version**:

```bash
uv run python -m src.pipeline.sample_generator_spark
```

### 4.4. Running Tests

Verify pipeline logic and correctness against mock data structures:

**Standard pipeline tests**:

```bash
uv run pytest -m "not spark"
```

**Spark pipeline tests**:

```bash
uv run pytest tests/test_spark_pipelines.py
```

## 5. Reproducing Analytical Results

Execute the Jupyter Notebooks located in `notebooks/` in sequential order:

1. **`01_eda_descriptive_analytics.ipynb`**: Imports the downsampled Parquet dataset, calculates statistical metrics, and generates 5 exploratory visualizations.
2. **`02_ml_feature_engineering_training.ipynb`**: Engineers technical indicators (moving averages, returns, signals) using Polars and trains machine learning models.
3. **`03_ml_evaluation_error_analysis.ipynb`**: Computes validation and test classification metrics (accuracy, precision, recall, confusion matrix) and conducts model error analysis.

## 6. AWS Hub-and-Spoke Architecture

For cloud execution, this project uses an AWS Hub-and-Spoke model:

- **Hub Account (Central S3 Bucket):** Houses the main `/raw/` and `/sample/` datasets. S3 resource-based bucket policies allow read-only access to specific teammate spoke accounts. A Glue Crawler catalogs the tables.
- **Spoke Accounts:** Teammates query the Glue Catalog tables via Amazon Athena from their respective AWS consoles.
- **SageMaker Run:** Once code is verified locally on the sample Parquet file, it can be executed in the Hub AWS account against the full raw dataset for scale-up training.
