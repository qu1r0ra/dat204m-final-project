# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Last updated**       | 2026-07-26                                                                                                                                                                                                                                                                                                                                                                       |
| **Last session focus** | Notebook 02 SageMaker Audit: Verified headless background execution via `jupyter nbconvert`, fixed missing `SequenceDataset` and `predict_lstm` imports in Cell 3 (preventing `NameError` on checkpoint evaluation fallback), verified path resolution, S3 IAM credentials, Polars Float32 memory optimization, and test suite green (32/32 tests pass). |
| **Active tasks**       | SageMaker Background Notebook Execution (`notebooks/02_ml_feature_engineering_training.ipynb` background training sweep & `notebooks/03_ml_evaluation_error_analysis.ipynb` evaluation).                                                                                                                                                                                         |
| **Blockers**           | None.                                                                                                                                                                                                                                                                                                                                                                            |

---

## 1. Quick Start (New Agent)

1. Read this file end-to-end to understand current state, then review active tasks in [Section 5](#5-implementation-queue) and the final refactoring plan in [docs/audits/final_refactors.md](docs/audits/final_refactors.md).
2. Review the feature implementation plans under [docs/plans/](docs/plans/).
3. Load canonical rules from [.cursor/rules/](.cursor/rules/).
4. Verify environment configurations in [.env](.env) and [src/config.py](src/config.py).
5. Verify PyTorch GPU environment (`torch==2.12.1+cu132`, CUDA True on RTX 5060) and run test suite using `uv run pytest`.

---

## 2. Workspace Map

| Directory/File               | Role                                                               | Domain Rules / Entrypoints                                                        |
| ---------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| [aws/](aws/)                 | CloudFormation infrastructure templates & policy definitions       | Hub-and-spoke security defaults                                                   |
| [data/](data/)               | Git-ignored data directory (both raw CSVs and downsampled Parquet) | Must remain git-ignored                                                           |
| [docs/](docs/)               | Reports, profiles, specifications, roles, and feature plans        | Final course deliverables; see [docs/session_history.md](docs/session_history.md) |
| [docs/audits/](docs/audits/) | Comprehensive codebase quality and final audit plans               | [final_refactors.md](docs/audits/final_refactors.md)                              |
| [docs/plans/](docs/plans/)   | Detailed feature & architecture implementation plans               | [refactoring_master_plan.md](docs/plans/refactoring_master_plan.md)               |
| [notebooks/](notebooks/)     | Jupyter deliverables for EDA, feature engineering, and evaluation  | Phase 1 & Phase 2 notebooks                                                       |
| [src/](src/)                 | Source package: pipelines, features, model routines, and S3 utils  | Source layout                                                                     |
| [tests/](tests/)             | Automated unit tests to verify pipeline processing                 | Ingestion validation                                                              |

---

## 3. Locked Architectural Decisions

Architectural decisions are managed canonically in `.cursor/rules/` and project registries.

- **Project Overview**: See [project-overview.mdc](.cursor/rules/project-overview.mdc).
- **Data & Directory Structure**: All raw and output data must reside in `data/` (git-ignored). See [data-organization.mdc](.cursor/rules/data-organization.mdc) and [data_registry.md](.cursor/project/data_registry.md).
- **AWS Hub-and-Spoke & Security**: Encryption and teammate access policies are defined in [data-organization.mdc](.cursor/rules/data-organization.mdc).
- **AWS Cloud Execution Strategy**: Full ML dataset training (609M rows / 80.8GB) is configured to run inside AWS (SageMaker Notebook Instance or SageMaker/EMR job) using `EXECUTION_MODE=aws_hub` in `.env`. Manual terminal bootstrapping (`uv sync`, `ipykernel install`) is documented in `docs/plans/full_aws_execution.md`. Notebook pathing dynamically resolves via `src/config.py`.
- **Tech Stack & Computations**: PySpark, DuckDB, Polars, PyTorch, and configuration guidelines are defined in [tech-stack.mdc](.cursor/rules/tech-stack.mdc).
- **Workflow & Rules**: Review [agent-workflows.mdc](.cursor/rules/agent-workflows.mdc) and [AGENTS.md](AGENTS.md).

---

## 4. Current Status Overview

- **Ingestion & S3 Pipeline**: 21,932 raw CSV files (~80.8 GB, 609M records) downloaded, stored in `data/raw/`, synced to S3, and cataloged in AWS Glue database (`binance_hub_db`). Downsampled 20-symbol Parquet sample (30.6M records) generated and cataloged.
- **Cloud Infrastructure**: CloudFormation stack `dat204m-binance-hub-stack` deployed with cross-account spoke access policy. SageMaker Notebook setup streamlined for manual terminal bootstrapping.
- **Distributed PySpark Engine**: PySpark pipelines operational for profiling, sample generation, feature engineering, and MLlib distributed training. Dynamic `winutils.exe` provisioning integrated for Windows compatibility.
- **Modeling & Feature Engineering Audit & Fixes**:
  - **`ModelArtifacts`**: Extended in `src/models/train.py` to include `lr_threshold` and `rf_threshold`.
  - **Notebook 02 Structure & Imports**: Restored Cell 1 Markdown explanation ("The Question We Are Answering"), resolved missing `load_lstm_artifacts`, `SequenceDataset`, and `predict_lstm` imports in Cell 3 (preventing `NameError` on resume-loading and fallback evaluation), and added `import numpy as np` in Cell 3.
  - **Emoji Removal**: Stripped all non-standard emojis from notebooks (`01`, `02`, `03`) and Python source modules to maintain clean production styling.
  - **LSTM Checkpoint Probability Fallback**: Added defensive fallback in Cell 17 of Notebook 02 to recompute validation probabilities if cached checkpoint history does not contain `val_probs` and `val_targets`.
  - **Validation Decision Threshold Tuning**: Implemented in Cell 19 of `notebooks/02_ml_feature_engineering_training.ipynb` to grid search cutoffs (0.40–0.60) for maximum balanced accuracy on validation partition.
  - **Per-Candidate Checkpoint Persistence & Caching**: Added automatic saving/loading of `.pt` checkpoints (`models/lstm_checkpoint_<slug>.pt`) in `src/models/lstm.py` and Notebook 02 (Cells 15 & 17). If a training run crashes or is interrupted, pre-trained candidates load from disk in <1 second instead of retraining from scratch.
  - **Memory & OS Resource Safeguards**: Configured `max_samples=0.2` and `n_jobs=min(os.cpu_count() or 4, 8)` in `RandomForestClassifier` inside `src/models/train.py` to eliminate `MemoryError` and `WinError 1450` thread handle limits on large matrices (21.4M rows).
  - **Notebook 03 Variable Alignment**: Fixed missing baseline imports and variable definitions (`baselines`, `lr`, `rf`) in `notebooks/03_ml_evaluation_error_analysis.ipynb` so all evaluation cells run cleanly.
- **Code Quality & Verification**: 32/32 unit tests pass in `pytest` cleanly with zero `ruff` lint or formatting errors.

_For detailed historical progress logs and completed task timelines, see [docs/session_history.md](docs/session_history.md)._

---

## 5. Implementation Queue (Handoff for Next Agent)

1. **[COMPLETED] Pipeline Audit, Checkpoint Caching & Memory Optimization**: Thoroughly audited Notebook 02 and 03, added per-candidate LSTM checkpoint caching, memory safeguards for Random Forest, and validated test suite.
2. **Next Agent Action Guide (GPU-Accelerated SageMaker Background Execution)**:
   - **Step 1**: Guide the user to run Notebook 02 headlessly in the background via the SageMaker terminal:
     ```bash
     nohup uv run python -m jupyter nbconvert --to notebook --execute notebooks/02_ml_feature_engineering_training.ipynb --output notebooks/02_ml_feature_engineering_training_executed.ipynb > train.log 2>&1 &
     ```
   - **Step 2**: Monitor background progress (`ps aux | grep python`, `tail -f train.log`, `nvidia-smi`) until `models/ml_artifacts.pkl` and `models/lstm_checkpoint.pt` are generated. Note: Any previously trained LSTM candidate checkpoint in `models/` will load instantly from disk without retraining.
   - **Step 3**: Execute `notebooks/03_ml_evaluation_error_analysis.ipynb` headlessly or interactively to evaluate model metrics and export `data/sample/test_evaluation_metrics.json`.

---

## 6. Code Quality & Architectural Refactoring Master Plan

The complete 9-pillar architectural design specification is maintained in [docs/plans/refactoring_master_plan.md](docs/plans/refactoring_master_plan.md). The final post-audit refactoring items are defined in [docs/audits/final_refactors.md](docs/audits/final_refactors.md).

_Archived tasks (1-15) and detailed milestone logs are stored in [docs/session_history.md](docs/session_history.md)._
