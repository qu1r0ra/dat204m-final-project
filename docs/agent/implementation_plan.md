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

The single source of truth for the repository structure is defined in [project_context.md](project_context.md).

---

## Proposed Changes

### Step 1: Clean Up & Move Dataset

- Move `binance_data/` to `data/raw/`.
- Ensure `.gitignore` ignores `data/` recursively.

### Step 2: Codebase Foundation

- **[config.py](../../src/config.py)**: Define configurable paths (using relative directories), TOP_20_SYMBOLS, target symbols, and S3 credentials.
- **[pyproject.toml](../../pyproject.toml)**: Register `duckdb`, `polars`, `pyarrow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `boto3`, `ipykernel` dependencies.

### Step 3: Pipeline Modules (DuckDB-powered)

- **[preprocess.py](../../src/pipeline/preprocess.py)**: Runs fast-profiling (using `approx_quantile` and native grouping) on the full dataset and writes `docs/data_profile.md`.
- **[sample_generator.py](../../src/pipeline/sample_generator.py)**: Slices out the top 20 symbols at 1-minute interval and exports a clean Parquet.

### Step 4: AWS Scripts

- **[aws_client.py](../../src/utils/aws_client.py)**: Upload scripts and bucket policy management using `boto3`.
- **[s3_bucket_policy.json](../../aws/s3_bucket_policy.json)**: JSON template to grant read-only S3 access to specific teammate accounts.

### Step 5: Feature & Model Templates

- **[indicators.py](../../src/features/indicators.py)**: Modular feature calculation in Polars.
- **[train.py](../../src/models/train.py)**: Model training logic parameterized by symbol name, split thresholds, and metrics calculations.

### Step 6: Jupyter Notebook Deliverables

- **[01_eda_descriptive_analytics.ipynb](../../notebooks/01_eda_descriptive_analytics.ipynb)**: Prepares the 5 charts on sample data.
- **[02_ml_feature_engineering_training.ipynb](../../notebooks/02_ml_feature_engineering_training.ipynb)**: Orchestrates model training.
- **[03_ml_evaluation_error_analysis.ipynb](../../notebooks/03_ml_evaluation_error_analysis.ipynb)**: Handles metrics comparison and error analysis.

---

## Verification Plan

### Manual Verification

- Ask user for confirmation to move the `binance_data/` directory to `data/raw/`.
- Propose code edits file-by-file, presenting them for your review and approval before writing.
- Execute data pipeline scripts and notebooks only under your direction.
