# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                                            |
| ---------------------- | ---------------------------------------------------------------- |
| **Last updated**       | 2026-07-22                                                       |
| **Last session focus** | Spoke AWS access, Phase 2 baseline comparison, train_spark rebuild |
| **Active tasks**       | Team decision: optional cloud scale-up run on full raw (Option A) |
| **Blockers**           | None                                                             |

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

### Phase 2 Modeling Session (2026-07-22)

- **Spoke AWS access**: Second teammate account (481088927299) connected to the hub bucket (bucket policy + `AmazonS3ReadOnlyAccess` on the IAM user). Reusable connection checker added at `scripts/check_aws.py` (`--iam` flag for permission probing). Sample Parquet cached locally; `.env` runs `local_sample` mode.
- **Method comparison design**: Four-model ladder — majority class (floor), OLS return regression thresholded (traditional), Logistic Regression, Random Forest (ML). All consume the same 11 stationary features. Global model trained on all top-20 symbols (21.4M train rows).
- **New modules**: `src/features/labels.py` (null-preserving direction labels), `src/models/baselines.py` (majority + OLS artifacts), `compute_split_boundaries` + `purge_minutes` in `src/models/train.py` (70/15/15 quantile split, 15-min label embargo). Tests in `tests/test_baselines.py`.
- **Memory fixes for 16 GB machines**: float32 feature matrices, RF `n_jobs=4`, notebooks free dataframes (`del`) after each stage.
- **Results (test partition, 4.6M rows)**: majority 54.35%, OLS 51.48% (AUC 0.502), LogReg 54.77% (AUC 0.538), RF 54.96% (AUC 0.548). Models stronger in high-volatility regimes; edge concentrated in balanced pairs (ETH/BTC/BNB). Findings written into notebook 03.
- **`train_spark.py` recreated** (was missing from this copy): `compute_targets_spark`, `split_data_chronologically_spark` (with purge), `train_pipeline_spark`, `DEFAULT_FEATURE_COLS`. Full Spark test suite passes again.

### Raw-Data Inspection & Feature Expansion (2026-07-23)

- **Feature set expanded 11 → 16**: added order-flow/time features from previously unused raw columns — `taker_buy_ratio`, `volume_z30`, `trades_z30`, `hour_sin`, `hour_cos` (`compute_flow_features` in `src/features/indicators.py`, mirrored in the Spark UDF + schema). Evidence: controlled 11-vs-16 experiment on the BTCUSDT slice improved RF validation AUC 0.5247 → 0.5263; all five rank above `log_return` in importance. Documented in notebook 02 intro.
- **MATICUSDT data-quality finding**: trading ends 2024-09-10 (token migrated to POL) → MATIC has training rows but zero val/test rows (per-symbol test tables show 19 symbols). Kept and documented (hub sample is fixed, spoke access read-only); ticker-set revision deferred to the Spark scale-up.
- **Raw inventory (from Phase 1 profile)**: 545 symbols, 210 with complete 3-year coverage. Full-coverage liquid candidates for scale-up expansion: PEPE, SUI, ARB, NEAR, OP, APT, INJ, FET. Note: profile says 545 symbols / 600.0M rows vs README 558 / Glue 609.5M — reconcile one sentence in the final report.
- **Memory optimizations**: notebooks 02/03 load only needed parquet columns, compute features per symbol, and keep slim float32 frames (peak ~15+ GB → ~6-7 GB on 16 GB machines).
- **Notebooks enriched for the rubric**: intro/problem/objective narratives, feature importance + interpretation section, conclusion with business recommendation, hyperparameter table + overfitting check, threshold-tuning section with exact values, references. Notebook 01 EDA extended to the 16 features.
- **Final 16-feature results (test, 4.6M rows)**: majority 54.35% / OLS 50.84% (AUC 0.501) / LogReg 54.76% (AUC 0.537) / RF 55.04% (AUC 0.551); tuned thresholds 0.480 lift RF recall 0.19→0.48 and balanced accuracy to 0.536 (best of any method). `taker_buy_ratio` ranks 4th in RF importance — the order-flow addition paid off. All notebook 03 findings/conclusion text synced to these numbers. RF `n_jobs` reduced to 2 for memory headroom on 16 GB machines.
- **Pending**: notebook 02 needs one final run **with Ctrl+S afterwards** to persist its outputs (models are already trained and saved; only the displayed outputs are missing). Notebook 03 is saved with outputs.

## 5. Implementation Queue

1. **(Optional, team decision)** Cloud scale-up training run on the full 609M-row raw dataset via `train_spark.py` on EMR/SageMaker (Option A). Pipeline is verified; this is budget/time, not code.
2. **(Optional)** Validation-tuned decision threshold or `class_weight="balanced"` to fix the classifiers' low "up" recall (~0.18); requires a ~35 min notebook 02 re-run.
3. Notebook 01 re-run on this machine pending kernel selection (`.venv`); Spark verified working locally.

_For archived tasks (1-10), see [session_history.md](docs/session_history.md)._
