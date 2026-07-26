# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Last updated**       | 2026-07-26                                                                                                                                                                                                                                                                                                         |
| **Last session focus** | SageMaker Session Persistence & Background Execution: Solved SageMaker session timeout issue during Notebook 02 training. Provided `nohup`/`tmux` headless background execution steps and updated `README.md` and `docs/plans/full_aws_execution.md`. Guided user on setting IAM Role Max Session Duration to 12h. |
| **Active tasks**       | SageMaker Background Notebook Execution (`notebooks/02_ml_feature_engineering_training.ipynb` background training sweep & `notebooks/03_ml_evaluation_error_analysis.ipynb` evaluation).                                                                                                                           |
| **Blockers**           | None.                                                                                                                                                                                                                                                                                                              |

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
- **Modeling & Feature Engineering**: 16-feature set configured. Classifiers pipeline operational across all 5 models (Majority Floor, OLS, LogReg, RF, and PyTorch LSTM with live `tqdm` progress and `models/lstm_training_log.jsonl` logging). Final metrics will be logged upon completing the AWS SageMaker AI production run.
- **Code Quality & Verification**: All 31 unit tests pass in `pytest` cleanly in 76s with zero ruff lint or formatting errors.

_For detailed historical progress logs and completed task timelines, see [docs/session_history.md](docs/session_history.md)._

---

## 5. Implementation Queue (Handoff for Next Agent)

1. **[COMPLETED] SageMaker Background Execution & Documentation**: Documented background headless execution commands (`nohup`/`tmux`) in `README.md` and `docs/plans/full_aws_execution.md` to prevent browser timeouts/session refreshes during Notebook 02 training.
2. **Next Agent Action Guide (GPU-Accelerated SageMaker Background Execution)**:
   - **Step 1**: Verify the user has set the IAM Role Max Session Duration to 12h.
   - **Step 2**: Guide the user to run Notebook 02 headlessly in the background via the SageMaker terminal on their right panel:
     ```bash
     nohup uv run python -m jupyter nbconvert --to notebook --execute notebooks/02_ml_feature_engineering_training.ipynb --output notebooks/02_ml_feature_engineering_training_executed.ipynb > train.log 2>&1 &
     ```
   - **Step 3**: Assist the user in monitoring background progress (`ps aux | grep python`, `tail -f train.log`, `nvidia-smi`) until `models/sklearn/ml_artifacts.pkl` and `models/lstm_model.pt` are generated.
   - **Step 4**: Execute `notebooks/03_ml_evaluation_error_analysis.ipynb` to evaluate model metrics and export `docs/evaluation_report.json`.

---

## 6. Code Quality & Architectural Refactoring Master Plan

The complete 9-pillar architectural design specification is maintained in [docs/plans/refactoring_master_plan.md](docs/plans/refactoring_master_plan.md). The final post-audit refactoring items are defined in [docs/audits/final_refactors.md](docs/audits/final_refactors.md).

_Archived tasks (1-15) and detailed milestone logs are stored in [docs/session_history.md](docs/session_history.md)._
