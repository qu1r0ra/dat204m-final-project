# Implementation Plan - Refined Architecture & Repository Reorganization

This plan outlines the strategy to reorganize the repository, move datasets to the git-ignored `data/` folder, set up simple cross-account AWS S3/Athena access, and define configurable code interfaces for Phase 1 and Phase 2.

---

## User Review Required

> [!IMPORTANT]
> **Data Organization:** We will move the existing `binance_data/` folder (75+ GB) to `data/raw/` to ensure the project root remains clean and git-ignored.
>
> **Explicit Permission Rule:** All code changes and script executions will require explicit confirmation from you before we run them, as we are strictly planning.

---

## Technical Design Decisions (Aligned)

1. **Option A (Binary Classification - Price Direction):** Feature horizon and target labels will be fully configurable via parameter.
2. **Option 1 (Downsampling):** Top 20 pairs at 1-minute frequency will be extracted to `data/sample/binance_sample.parquet` using DuckDB.
3. **AWS Architecture:**
   - Central Hub S3 bucket hosts `/raw/` and `/sample/` Parquet files.
   - Resource-based S3 policy allows teammate accounts read-only access.
   - Glue Catalog database crawled in the Hub account.
   - Teammates query using Athena in their spoke accounts.
   - local development is done on the local sample Parquet file.
4. **Agent Guidelines:** Keep `docs/agent/project_context.md` and root `AGENTS.md` updated.
5. **Configuration Tooling:** Use simple `src/config.py` rather than Hydra to avoid overcomplication for teammates.

---

## Proposed Repository Structure

```text
dat204m-final-project/
├── data/                    # Local directory (git-ignored)
│   ├── raw/                 # Will hold the raw 75+ GB CSV folder
│   └── sample/              # Generated 1-2 GB sample Parquet dataset
├── src/                     # Core source code package
│   ├── __init__.py
│   ├── config.py            # Parameters for symbol selection, horizons, and run modes
│   ├── pipeline/            # Data processing scripts
│   │   ├── __init__.py
│   │   ├── preprocess.py    # Profiles and cleans data/raw/ using DuckDB
│   │   └── sample_generator.py # Deduplicates and exports sample Parquet
│   ├── features/            # Feature engineering indicators
│   │   ├── __init__.py
│   │   └── indicators.py    # Returns, SMA, EMA, RSI, Bollinger Bands
│   ├── models/              # Machine learning models & evaluation
│   │   ├── __init__.py
│   │   ├── baselines.py     # Majority class classifier baseline
│   │   └── train.py         # Trains models based on target symbol parameter
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── aws_client.py    # S3 and Glue Crawler helper scripts using boto3
├── aws/                     # AWS-specific infrastructure configuration
│   ├── s3_bucket_policy.json # JSON policy for cross-account spokes
│   └── athena_queries.sql   # SQL templates for descriptive queries in Athena
├── notebooks/               # Jupyter Notebooks for reports
│   ├── 01_eda_descriptive_analytics.ipynb         # Phase 1: Descriptive Stats & Visuals
│   ├── 02_ml_feature_engineering_training.ipynb   # Phase 2: Feature Engineering & Training
│   └── 03_ml_evaluation_error_analysis.ipynb      # Phase 2: Evaluation, Error Analysis & Business Recommendations
├── docs/                    # Written deliverables & context
│   ├── team_roles.md        # Team roles and task dissemination
│   ├── eda_report.md        # Formatted markdown report for descriptive analytics
│   ├── final_report.md      # Final written project report
│   └── agent/               # Agent-centric context files, rules, and plans
│       ├── project_context.md # Agent-agnostic project details
│       ├── implementation_plan.md # Technical implementation plan
│       ├── task.md          # Active task list tracking
│       └── plan.md          # Implementation Blueprint
├── AGENTS.md                # Rules and workflow guidelines for AI coding assistants
├── .gitignore
├── pyproject.toml           # uv project configuration
└── README.md
```

---

## Proposed Changes

### Step 1: Clean Up & Move Dataset

- Move `binance_data/` to `data/raw/`.
- Ensure `.gitignore` ignores `data/` recursively.

### Step 2: Codebase Foundation

- **[config.py](../src/config.py)**: Define configurable paths (using relative directories), TOP_20_SYMBOLS, target symbols, and S3 credentials.
- **[pyproject.toml](../pyproject.toml)**: Register `duckdb`, `polars`, `pyarrow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `boto3`, `ipykernel` dependencies.

### Step 3: Pipeline Modules (DuckDB-powered)

- **[preprocess.py](../src/pipeline/preprocess.py)**: Runs fast-profiling (using `approx_quantile` and native grouping) on the full dataset and writes `docs/data_profile.md`.
- **[sample_generator.py](../src/pipeline/sample_generator.py)**: Slices out the top 20 symbols at 1-minute interval and exports a clean Parquet.

### Step 4: AWS Scripts

- **[aws_client.py](../src/utils/aws_client.py)**: Upload scripts and bucket policy management using `boto3`.
- **[s3_bucket_policy.json](../aws/s3_bucket_policy.json)**: JSON template to grant read-only S3 access to specific teammate accounts.

### Step 5: Feature & Model Templates

- **[indicators.py](../src/features/indicators.py)**: Modular feature calculation in Polars.
- **[train.py](../src/models/train.py)**: Model training logic parameterized by symbol name, split thresholds, and metrics calculations.

### Step 6: Jupyter Notebook Deliverables

- **[01_eda_descriptive_analytics.ipynb](../notebooks/01_eda_descriptive_analytics.ipynb)**: Prepares the 5 charts on sample data.
- **[02_ml_feature_engineering_training.ipynb](../notebooks/02_ml_feature_engineering_training.ipynb)**: Orchestrates model training.
- **[03_ml_evaluation_error_analysis.ipynb](../notebooks/03_ml_evaluation_error_analysis.ipynb)**: Handles metrics comparison and error analysis.

---

## Verification Plan

### Manual Verification

- Ask user for confirmation to move the `binance_data/` directory to `data/raw/`.
- Propose code edits file-by-file, presenting them for your review and approval before writing.
- Execute data pipeline scripts and notebooks only under your direction.
