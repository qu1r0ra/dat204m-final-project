# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Last updated**       | 2026-07-24                                                                                                                             |
| **Last session focus** | PyTorch LSTM Implementation Verification & Evaluation Report ([docs/plans/lstm_implementation.md](docs/plans/lstm_implementation.md))  |
| **Active tasks**       | Finalize metrics logging for ML analysis; Run Notebook 02 & 03 execution to persist outputs; (Optional) Cloud scale-up on 609M dataset |
| **Blockers**           | None                                                                                                                                   |

---

## 1. Quick Start (New Agent)

1. Read this file end-to-end to understand current state, then review the implementation queue in [Section 5](#5-implementation-queue).
2. Review the detailed implementation plan in [docs/plans/lstm_implementation.md](docs/plans/lstm_implementation.md).
3. Load canonical rules from [.cursor/rules/](.cursor/rules/).
4. Verify environment configurations in [.env](.env) and [src/config.py](src/config.py).
5. Verify PyTorch GPU environment (`torch==2.12.1+cu132`, CUDA True on RTX 5060) and run tests using `uv run pytest`.

---

## 2. Workspace Map

| Directory/File             | Role                                                               | Domain Rules / Entrypoints          |
| -------------------------- | ------------------------------------------------------------------ | ----------------------------------- |
| [aws/](aws/)               | CloudFormation infrastructure templates & policy definitions       | Hub-and-spoke security defaults     |
| [data/](data/)             | Git-ignored data directory (both raw CSVs and downsampled Parquet) | Must remain git-ignored             |
| [docs/](docs/)             | Reports, profiles, specifications, roles, and feature plans        | Final course deliverables           |
| [docs/plans/](docs/plans/) | Detailed feature & architecture implementation plans               | `docs/plans/lstm_implementation.md` |
| [notebooks/](notebooks/)   | Jupyter deliverables for EDA, feature engineering, and evaluation  | Phase 1 & Phase 2 notebooks         |
| [src/](src/)               | Source package: pipelines, features, model routines, and S3 utils  | Source layout                       |
| [tests/](tests/)           | Automated unit tests to verify pipeline processing                 | Ingestion validation                |

---

## 3. Locked Architectural Decisions

Architectural decisions are managed canonically in `.cursor/rules/` and project registries.

- **Project Overview**: See [project-overview.mdc](.cursor/rules/project-overview.mdc).
- **Data & Directory Structure**: All raw and output data must reside in `data/` (git-ignored). See [data-organization.mdc](.cursor/rules/data-organization.mdc) and [data_registry.md](.cursor/project/data_registry.md).
- **AWS Hub-and-Spoke & Security**: Encryption and teammate access policies are defined in [data-organization.mdc](.cursor/rules/data-organization.mdc).
- **Tech Stack & Computations**: PySpark, DuckDB, Polars, PyTorch, and configuration guidelines are defined in [tech-stack.mdc](.cursor/rules/tech-stack.mdc).
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
- **Refactoring & Code Quality**: Fully completed 14-item refactoring plan to improve codebase health.
- **Spark Client Windows Compatibility**: Integrated dynamic `HADOOP_HOME` configuration and local `winutils.exe` provisioning directly inside [spark_client.py](src/utils/spark_client.py).
- **Notebook Migration**: Upgraded Phase 1 EDA notebook [01_eda_descriptive_analytics.ipynb](notebooks/01_eda_descriptive_analytics.ipynb) to utilize the distributed PySpark pipeline for data loading and feature engineering.

---

### Phase 2 Modeling Session (2026-07-22)

- **Spoke AWS access**: Second teammate account (481088927299) connected to the hub bucket.
- **Method comparison design**: Four-model ladder — majority class (floor), OLS return regression thresholded (traditional), Logistic Regression, Random Forest (ML).
- **Results (test partition, 4.6M rows)**: majority 54.35%, OLS 51.48% (AUC 0.502), LogReg 54.77% (AUC 0.538), RF 54.96% (AUC 0.548).

### Raw-Data Inspection & Feature Expansion (2026-07-23)

- **Feature set expanded 11 → 16**: added order-flow/time features from previously unused raw columns — `taker_buy_ratio`, `volume_z30`, `trades_z30`, `hour_sin`, `hour_cos`.
- **Final 16-feature results (test, 4.6M rows)**: majority 54.35% / OLS 50.84% (AUC 0.501) / LogReg 54.76% (AUC 0.537) / RF 55.04% (AUC 0.551); tuned thresholds 0.480 lift RF recall 0.19→0.48 and balanced accuracy to 0.536 (best of any method).

---

### PyTorch LSTM Sequence Classifier Implementation Completed (2026-07-24)

- **Feature Sync & Cleanup**: Centralized canonical 16-feature list in `src/config.py` as `FEATURE_COLS`. Updated `src/models/train_spark.py`, `.cursor/rules/tech-stack.mdc`, and `.cursor/project/model_registry.md`.
- **PyTorch Model & Pipeline**: Implemented `src/models/lstm.py` containing `SequenceDataset` (symbol-isolated sliding windows), `LSTMClassifier` (2-layer LSTM + Linear), `train_lstm` (AdamW, BCEWithLogitsLoss, early stopping, threshold tuning), `predict_lstm`, and artifact serialization (`save_lstm_artifacts` / `load_lstm_artifacts` with `weights_only=False` support).
- **Unit Test Suite**: Added `tests/test_lstm.py` covering sequence dataset symbol isolation, forward pass output dimensions, training convergence, and artifact round-trip. All 10 unit tests pass in <7s.
- **Notebook Integration**: Integrated PyTorch LSTM training, 5-candidate hyperparameter sensitivity sweep, threshold tuning, and validation table updates into `notebooks/02_ml_feature_engineering_training.ipynb`, and added sequence dataset evaluation on the aligned test partition in `notebooks/03_ml_evaluation_error_analysis.ipynb`.

---

## 5. Implementation Queue

1. **(Completed)** PyTorch LSTM Sequence Classifier Implementation & Evaluation Audit ([evaluation_report.md](file:///C:/Users/Quirora/.gemini/antigravity-ide/brain/e36d823b-01fc-44e7-a653-e24603f5e7f0/evaluation_report.md)).
2. **(Completed) Finalize Metrics Logging for ML Analysis**:
   - Audited metrics calculation across `src/models/evaluation.py`, `src/models/train.py`, `src/models/train_spark.py`, `src/models/lstm.py`, and notebooks `02` & `03`.
   - Added comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Balanced Accuracy, Log Loss, Specificity, NPV, Confusion Matrix breakdown, Per-Symbol performance, Volatility Regime splits, Per-Epoch Training/Validation curves, and Hyperparameter Sweep outputs).
   - Implemented structured JSON persistence (`save_metrics_json`, `lstm_sweep_metrics.json`, `test_evaluation_metrics.json`, `test_regime_metrics.json`, `test_per_symbol_metrics.json`).
3. **Notebook Cell Execution**: Re-run Notebook 02 & 03 to persist output cells with trained PyTorch LSTM model weights and updated evaluation metrics.
4. **(Optional, team decision)** Cloud scale-up training run on the full 609M-row raw dataset via `train_spark.py` on EMR/SageMaker (Option A).

_For archived tasks (1-10), see [session_history.md](docs/session_history.md)._
