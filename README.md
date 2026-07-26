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
  - [4.4. Unified Pipeline CLI](#44-unified-pipeline-cli)
  - [4.5. End-to-End Pipeline Execution Guide](#45-end-to-end-pipeline-execution-guide)
    - [Option A: Local Execution (Fast Development Run)](#option-a-local-execution-fast-development-run)
    - [Option B: AWS Cloud Execution (Full Production Scale-Up)](#option-b-aws-cloud-execution-full-production-scale-up)
  - [4.6. Running Tests](#46-running-tests)
- [5. Reproducing Analytical Results](#5-reproducing-analytical-results)
- [6. AWS Hub-and-Spoke Architecture \& Cloud Deployment](#6-aws-hub-and-spoke-architecture--cloud-deployment)
  - [6.1. Infrastructure Setup](#61-infrastructure-setup)
  - [6.2. S3 Cross-Account Bucket Policy](#62-s3-cross-account-bucket-policy)
  - [6.3. SageMaker Notebook Instance Bootstrapping \& Production Run Guide](#63-sagemaker-notebook-instance-bootstrapping--production-run-guide)

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
│   ├── hub_infrastructure.yaml # CloudFormation template for hub infrastructure
│   ├── s3_bucket_policy.json # Cross-account S3 bucket policy template
│   └── sagemaker_bootstrap.sh# SageMaker Notebook lifecycle configuration script
├── data/                     # Git-ignored local data directory
│   ├── raw/                  # Downloaded raw monthly CSV files
│   └── sample/               # Local compressed sample Parquet files
├── docs/                     # Written deliverables, reports, and documentation
│   ├── audits/               # Codebase quality and final audit plans
│   │   └── final_refactors.md# Post-audit refactoring roadmap
│   ├── plans/                # Detailed feature & architecture implementation plans
│   │   ├── full_aws_execution.md    # AWS SageMaker AI production execution guide
│   │   ├── lstm_design.md           # Sequence dataset & PyTorch LSTM design spec
│   │   ├── lstm_implementation.md   # PyTorch LSTM implementation specification
│   │   └── refactoring_master_plan.md# 9-pillar architectural master plan
│   ├── data_profile.md       # Auto-generated dataset profiling report (DuckDB)
│   ├── data_profile_spark.md # Auto-generated dataset profiling report (Spark)
│   ├── session_history.md    # Historical task milestone timeline
│   ├── specs.md              # Project specifications
│   └── team_roles.md         # Team roles and task dissemination
├── notebooks/                # Jupyter Notebooks for deliverables
│   ├── 01_eda_descriptive_analytics.ipynb         # Phase 1: Descriptive profiling & visualizations
│   ├── 02_ml_feature_engineering_training.ipynb   # Phase 2: Signal features & model training
│   └── 03_ml_evaluation_error_analysis.ipynb      # Phase 2: Evaluation metrics & recommendations
├── src/                      # Source package
│   ├── __init__.py
│   ├── cli.py                # Unified Pipeline CLI entrypoint
│   ├── config.py             # Configuration parameters loader & environment loader
│   ├── exceptions.py         # Custom domain exception classes
│   ├── features/             # Feature engineering and signal generation
│   │   ├── __init__.py
│   │   ├── indicators.py     # Rolling Polars technical indicators
│   │   ├── indicators_spark.py # UDF-based Spark feature indicators
│   │   └── labels.py         # Binary price direction label generator
│   ├── models/               # Model definitions, training, and evaluation pipelines
│   │   ├── __init__.py
│   │   ├── base.py           # Base Classifier abstract base class
│   │   ├── baselines.py      # Majority Floor and OLS baseline classifiers
│   │   ├── evaluation.py     # Unified classification metrics and evaluation
│   │   ├── lstm.py           # PyTorch sequence dataset & LSTM architecture
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
│       ├── seed.py           # Global random seed utility for reproducibility
│       └── spark_client.py   # Unified Spark Session configuration
├── tests/                    # pytest suite for validation (31 unit tests)
│   ├── __init__.py
│   ├── conftest.py           # Shared test fixtures and configuration
│   ├── test_config_and_base.py # Configuration loader & base model tests
│   ├── test_evaluation.py    # Metric calculation unit tests
│   ├── test_features.py      # Technical indicator & label unit tests
│   ├── test_lstm.py          # PyTorch LSTM architecture & dataset tests
│   ├── test_models.py        # Baseline & scikit-learn model tests
│   ├── test_pipelines.py     # Automated tests for ingestion and processing
│   ├── test_refactoring.py   # Architectural integrity & abstraction tests
│   └── test_spark_pipelines.py # Spark-specific integration test suite
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

### 4.4. Unified Pipeline CLI

Alternatively, run individual steps or full pipelines using the unified CLI entrypoint:

```bash
# Data preparation steps
uv run python -m src.cli profile
uv run python -m src.cli sample

# Model training steps
uv run python -m src.cli train-sklearn
uv run python -m src.cli train-lstm
uv run python -m src.cli train-spark

# Evaluation & metrics export
uv run python -m src.cli evaluate
```

### 4.5. End-to-End Pipeline Execution Guide

#### Option A: Local Execution (Fast Development Run)

1. **Data Preparation**: Ensure `data/sample/binance_sample.parquet` exists (generate via step 4.3).
2. **Train Models**:
   ```bash
   uv run python -m src.cli train-sklearn
   uv run python -m src.cli train-lstm
   ```
3. **Run Test Evaluation**:
   ```bash
   uv run python -m src.cli evaluate
   ```

#### Option B: AWS Cloud Execution (Full Production Scale-Up)

1. **Environment Configuration**:
   Set environment variables in `.env` or your shell:
   ```bash
   EXECUTION_MODE=aws_hub
   AWS_S3_BUCKET_NAME=dat204m-binance-bigdata-hub-sg
   AWS_DEFAULT_REGION=ap-southeast-1
   ```
2. **Execute PySpark / PyTorch Cloud Pipeline**:
   - **CLI Execution**:
     ```bash
     # Distributed PySpark MLlib training on cloud dataset
     uv run python -m src.cli train-spark
     ```
   - **SageMaker Notebook Execution**:
     - `notebooks/02_ml_feature_engineering_training.ipynb` has been pre-configured for production execution (`DEV_SYMBOLS = None` for all trading pairs, `max_epochs = 20` for LSTM convergence).
     - Parquet loading automatically uses `config.load_parquet_auto()` to pull directly from S3 via `s3fs`.

### 4.6. Running Tests

Verify pipeline logic, feature engineering, baseline/ML models, PyTorch LSTM architectures, and S3 path loading against unit test fixtures:

```bash
# Full automated test suite (31 unit tests across 8 test modules)
uv run pytest

# Fast unit tests (excluding PySpark integration)
uv run pytest -m "not spark"

# PySpark integration tests
uv run pytest tests/test_spark_pipelines.py
```

## 5. Reproducing Analytical Results

Execute the Jupyter Notebooks located in `notebooks/` in sequential order:

1. **`01_eda_descriptive_analytics.ipynb`**: [notebooks/01_eda_descriptive_analytics.ipynb](notebooks/01_eda_descriptive_analytics.ipynb)
   - Profiles dataset distributions, missingness, stationarity, and feature correlations.
2. **`02_ml_feature_engineering_training.ipynb`**: [notebooks/02_ml_feature_engineering_training.ipynb](notebooks/02_ml_feature_engineering_training.ipynb)
   - Pre-configured for production: `DEV_SYMBOLS = None` (all trading pairs) and `max_epochs = 20` (full training sweep).
   - Engineers 16 canonical features, applies chronological splitting with 15-minute purging, trains 5 benchmark models, and tunes decision thresholds.
   - For fast local dev iterations, set `DEV_SYMBOLS = ["BTCUSDT", "ETHUSDT"]` and `max_epochs = 2`.
3. **`03_ml_evaluation_error_analysis.ipynb`**: [notebooks/03_ml_evaluation_error_analysis.ipynb](notebooks/03_ml_evaluation_error_analysis.ipynb)
   - Evaluates trained models on held-out test data, generates ROC curves and confusion matrices, analyzes volatility regime performance, and exports metrics.

## 6. AWS Hub-and-Spoke Architecture & Cloud Deployment

### 6.1. Infrastructure Setup

Deploy the hub infrastructure using CloudFormation template [aws/hub_infrastructure.yaml](aws/hub_infrastructure.yaml):

```bash
aws cloudformation create-stack \
  --stack-name dat204m-binance-hub-stack \
  --template-body file://aws/hub_infrastructure.yaml \
  --capabilities CAPABILITY_IAM
```

### 6.2. S3 Cross-Account Bucket Policy

Apply [aws/s3_bucket_policy.json](aws/s3_bucket_policy.json) to grant read-only access to teammate spoke accounts while preserving central encryption defaults.

### 6.3. SageMaker Notebook Instance Bootstrapping & Production Run Guide

1. **Instance Launch**: Create an **Amazon SageMaker AI** Notebook Instance choosing **`ml.g4dn.2xlarge`** (8 vCPUs, 32 GB RAM, 1x NVIDIA T4 GPU with 16 GB VRAM) with **`Amazon Linux 2023, Jupyter Lab 4`** platform and at least **35 GB** EBS volume storage. Attach IAM Execution Role `SageMaker-sagemaker-binance-hub-role-2`.
2. **Terminal Bootstrapping**:
   - Open JupyterLab and open a Terminal.
   - Clone repository and sync virtual environment:
     ```bash
     cd ~/SageMaker
     git clone https://github.com/qu1r0ra/dat204m-final-project.git
     cd dat204m-final-project
     curl -LsSf https://astral.sh/uv/install.sh | sh
     export PATH="$HOME/.local/bin:$PATH"
     uv sync
     uv run python -m ipykernel install --user --name="dat204m-final-project" --display-name="Python (DAT204M)"
     ```
   - Create `.env` configuration file:
     ```bash
     cat << 'EOF' > .env
     EXECUTION_MODE=aws_hub
     SPARK_EXECUTION_MODE=local
     AWS_DEFAULT_REGION=ap-southeast-1
     AWS_S3_BUCKET_NAME=dat204m-binance-bigdata-hub-sg
     EOF
     ```
3. **Pipeline Execution**:
   - Open Jupyter notebook `notebooks/01_eda_descriptive_analytics.ipynb` and select kernel **`Python (DAT204M)`**. Run all cells.
   - Open `notebooks/02_ml_feature_engineering_training.ipynb`. The notebook reads cloud data directly from S3, trains all 5 classifiers across all top 20 trading pairs (30.65M rows) with 20-epoch PyTorch LSTM GPU sweeps (~10-12 mins), and saves trained artifacts to `models/`.
   - Open `notebooks/03_ml_evaluation_error_analysis.ipynb` and run all cells to evaluate trained classifiers on the held-out test partition and export `docs/evaluation_report.json`.

4. **Uninterrupted Background Execution (Preventing Session Disconnects & Timeouts)**:

   > [!TIP]
   > **Preventing Browser Disconnects**: Interactive browser sessions in SageMaker JupyterLab can disconnect or require a session refresh when idle or when AWS SSO tokens expire. To run Notebook 02 headlessly in the background without keeping your browser open:
   - **Option A (Terminal - Recommended)**:
     Run the notebook in the SageMaker terminal on your instance:

     ```bash
     nohup uv run python -m jupyter nbconvert --to notebook --execute notebooks/02_ml_feature_engineering_training.ipynb --output notebooks/02_ml_feature_engineering_training_executed.ipynb > train.log 2>&1 &
     ```

     Monitor progress anytime in the terminal:

     ```bash
     # Check process status
     ps aux | grep python

     # Watch live log output
     tail -f train.log

     # Monitor GPU utilization & VRAM
     nvidia-smi
     ```

   - **Option B (Notebook Code Cell)**:
     Insert a code cell at the top of your notebook in Jupyter:

     ```python
     import subprocess

     subprocess.Popen(
         "nohup uv run python -m jupyter nbconvert --to notebook --execute notebooks/02_ml_feature_engineering_training.ipynb --output notebooks/02_ml_feature_engineering_training_executed.ipynb > train.log 2>&1 &",
         shell=True,
     )
     print("🚀 Background training started! You can safely close your browser.")
     ```
