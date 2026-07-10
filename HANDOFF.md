# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                             |
| ---------------------- | ------------------------------------------------- |
| **Last updated**       | 2026-07-10                                        |
| **Last session focus** | Codebase-wide Refactoring & Comprehensive Testing |
| **Active tasks**       | None                                              |
| **Blockers**           | None                                              |

---

## 1. Quick Start (New Agent)

1. Read this file end-to-end to understand current state, then review the implementation queue in [Section 5](#5-implementation-queue).
2. Load canonical rules from [.cursor/rules/](.cursor/rules/).
3. Verify environment configurations in [.env](.env) and [src/config.py](src/config.py).
4. Run tests using `uv run pytest` to ensure environment correctness.

---

## 2. Workspace Map

| Directory/File           | Role                                                               | Domain Rules / Entrypoints      |
| ------------------------ | ------------------------------------------------------------------ | ------------------------------- |
| [aws/](aws/)             | CloudFormation infrastructure templates & policy definitions       | Hub-and-spoke security defaults |
| [data/](data/)           | Git-ignored data directory (both raw CSVs and downsampled Parquet) | Must remain git-ignored         |
| [docs/](docs/)           | Reports, profiles, specifications, and roles                       | Final course deliverables       |
| [notebooks/](notebooks/) | Jupyter deliverables for EDA, feature engineering, and evaluation  | Phase 1 & Phase 2 notebooks     |
| [src/](src/)             | Source package: pipelines, features, model routines, and S3 utils  | Source layout                   |
| [tests/](tests/)         | Automated unit tests to verify pipeline processing                 | Ingestion validation            |

---

## 3. Locked Architectural Decisions

Architectural decisions are managed canonically in `.cursor/rules/` and project registries.

- **Project Overview**: See [project-overview.mdc](.cursor/rules/project-overview.mdc).
- **Data & Directory Structure**: All raw and output data must reside in `data/` (git-ignored). See [data-organization.mdc](.cursor/rules/data-organization.mdc) and [data_registry.md](.cursor/project/data_registry.md).
- **AWS Hub-and-Spoke & Security**: Encryption and teammate access policies are defined in [data-organization.mdc](.cursor/rules/data-organization.mdc).
- **Tech Stack & Computations**: PySpark, DuckDB, Polars, and configuration guidelines are defined in [tech-stack.mdc](.cursor/rules/tech-stack.mdc).
- **Workflow & Rules**: Review [agent-workflows.mdc](.cursor/rules/agent-workflows.mdc) and [AGENTS.md](AGENTS.md).

---

## 4. Current Status

### Ingestion Pipeline

- **Raw CSVs**: Downloader script executed, all 21,932 files stored locally under `data/raw/binance_data/`.
- **Sample Generation**: Local sample Parquet (`data/sample/binance_sample.parquet`) successfully generated containing the top 20 liquid symbols at 1m interval.
- **S3 Uploads**:
  - Sample Parquet dataset uploaded to `s3://dat204m-binance-bigdata-hub-sg/sample/`.
  - Raw dataset upload is fully completed and verified parallel to local storage.
  - S3 and local data have been cross-verified (21,932 files, ~80.80 GB matched byte-for-byte).
- **Glue Catalog Database**:
  - Database `binance_hub_db` successfully created.
  - Table `raw` created pointing to `s3://dat204m-binance-bigdata-hub-sg/raw/` containing all 21,932 objects (609,491,961 records).
  - Table `sample` created pointing to `s3://dat204m-binance-bigdata-hub-sg/sample/` containing the parquet file (30,659,220 records).

### Cloud Infrastructure

- CloudFormation stack `dat204m-binance-hub-stack` deployed.
- Resource policies set up to grant read-only S3 access to teammate accounts.

### PySpark Distributed Integration

- **Profiling**: Spark-based dataset profiling script [preprocess_spark.py](src/pipeline/preprocess_spark.py) aggregates raw CSV files and writes [data_profile_spark.md](docs/data_profile_spark.md).
- **Downsampling**: Spark downsampling script [sample_generator_spark.py](src/pipeline/sample_generator_spark.py) generates the downsampled Parquet dataset.
- **Feature Engineering**: UDF-based [indicators_spark.py](src/features/indicators_spark.py) scales the original Polars rolling technical indicators in parallel across symbols.
- **Machine Learning**: Spark MLlib [train_spark.py](src/models/train_spark.py) supports distributed `LogisticRegression` and `RandomForestClassifier` training on the global dataset.
- **Testing**: Test suite split into fast unit tests (`tests/test_pipelines.py`, runs in <1s) and slow Spark integration tests (`tests/test_spark_pipelines.py`, runs in ~1m). Spark tests are fully compatible with Windows platforms using path resolution, `PYSPARK_PYTHON` configuration, parquet read/write mocks, and timezone-agnostic split boundaries.
- **Refactoring & Code Quality**: Fully completed 14-item refactoring plan to improve codebase health:
  - **Environment Hygiene**: Moved `os.environ` mutations in `spark_client.py` and `config.py` to run lazily inside a setup function rather than unconditionally at import time, preventing notebook pollution.
  - **Error Reliability**: Re-raised exceptions in pipeline functions and updated `__main__` guards in all four pipelines to exit with code 1 upon failure.
  - **Deduplication**: Extracted a shared raw data schema (`RAW_KLINE_CSV_SCHEMA` in `schemas.py`), shared Hadoop `winutils` provisioning logic, and a central list of default model feature columns.
  - **Validation & Typing**: Added input column validations to Polars features, dataclasses for ML outputs (`ModelArtifacts`, `SparkModelArtifacts`), missing return type annotations, resolved implicit optionals, and vectorized Markdown table generation in `helpers.py`.
  - **Cleanup & Linting**: Deleted dead code (`smoke_test.py`), restructured test configuration using `__init__.py` and `conftest.py`, added a strict `ruff` config in `pyproject.toml`, and fully formatted and linted the codebase to satisfy all rules.
  - **Testing**: Added `tests/test_features.py` and `tests/test_models.py` with shared fixtures to cover technical indicators, stationary features, metrics calculation, and chronological data splitting.
- **Spark Client Windows Compatibility**: Integrated dynamic `HADOOP_HOME` configuration and local `winutils.exe` provisioning directly inside [spark_client.py](src/utils/spark_client.py). Configured `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` environment variables to point directly to `sys.executable`, eliminating Python worker connection timeouts and runtime environment mismatch errors on Windows systems.
- **Notebook Migration**: Upgraded Phase 1 EDA notebook [01_eda_descriptive_analytics.ipynb](notebooks/01_eda_descriptive_analytics.ipynb) to utilize the distributed PySpark pipeline for data loading, descriptive analytics, parallel feature engineering, and binary price direction labeling. Converted Spark DataFrames back to Polars DataFrames locally to preserve contract compatibility with downstream visualization cells.

---

## 5. Implementation Queue

_No active tasks in the queue. All planned tasks have been completed._

_For archived tasks (1-10), see [session_history.md](docs/session_history.md)._
