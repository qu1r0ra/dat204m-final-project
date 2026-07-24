# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                                                                                                                                                                                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Last updated**       | 2026-07-25                                                                                                                                                                                                                                                                             |
| **Last session focus** | Environment Variable & AWS Production Readiness Verification: Verified AWS IAM STS credentials (`arn:aws:iam::872891100013:user/qu1r0ra`), set `EXECUTION_MODE=aws_hub` in `.env`, fixed dummy session token conflict, validated live S3 data ingestion (30.65M rows), added `SPARK_EXECUTION_MODE=local`, verified 30/30 unit tests pass (71.7s), and synced all documentation (`/sync-docs`). |
| **Active tasks**       | SageMaker Notebook Instance Option B Production Execution (`notebooks/02_ml_feature_engineering_training.ipynb` & `notebooks/03_ml_evaluation_error_analysis.ipynb`). |
| **Blockers**           | None                                                                                                                                                                                                                                                                                   |

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
- **AWS Cloud Execution Strategy**: Full ML dataset training (609M rows / 80.8GB) is configured to run inside AWS (SageMaker Notebook Instance or SageMaker/EMR job) using `EXECUTION_MODE=aws_hub` in `.env` and `aws/sagemaker_bootstrap.sh`. Notebook pathing dynamically resolves via `src/config.py`.
- **Tech Stack & Computations**: PySpark, DuckDB, Polars, PyTorch, and configuration guidelines are defined in [tech-stack.mdc](.cursor/rules/tech-stack.mdc).
- **Workflow & Rules**: Review [agent-workflows.mdc](.cursor/rules/agent-workflows.mdc) and [AGENTS.md](AGENTS.md).

---

## 4. Current Status Overview

- **Ingestion & S3 Pipeline**: 21,932 raw CSV files (~80.8 GB, 609M records) downloaded, stored in `data/raw/`, synced to S3, and cataloged in AWS Glue database (`binance_hub_db`). Downsampled 20-symbol Parquet sample (30.6M records) generated and cataloged.
- **Cloud Infrastructure**: CloudFormation stack `dat204m-binance-hub-stack` deployed with cross-account spoke access policy. `aws/sagemaker_bootstrap.sh` ready for SageMaker Notebook kernel registration and dependency sync.
- **Distributed PySpark Engine**: PySpark pipelines operational for profiling, sample generation, feature engineering, and MLlib distributed training. Dynamic `winutils.exe` provisioning integrated for Windows compatibility.
- **Modeling & Feature Engineering**: 16-feature set configured. Classifiers evaluated (Majority Floor 54.35%, OLS 50.84%, LogReg 54.76%, RF 55.04% AUC 0.551, PyTorch LSTM Sequence Classifier passing all unit tests).
- **Code Quality & Verification**: All 30 unit tests pass in `pytest` cleanly in 75s with zero ruff lint or formatting errors.

_For detailed historical progress logs and completed task timelines, see [docs/session_history.md](docs/session_history.md)._

---

## 5. Implementation Queue (Handoff for Next Agent)

1. **[COMPLETED] Code Check & Unit Test Verification**: Executed `uv run pytest`, verifying 30/30 unit tests pass cleanly. Confirmed AWS cloud execution strategy via SageMaker Notebook Instance and dynamic `src/config.py` path switching.
2. **[COMPLETED] Notebook Presentation & Explanatory Refinement**: Refactored `notebooks/02_ml_feature_engineering_training.ipynb` and `notebooks/03_ml_evaluation_error_analysis.ipynb` with plain-language feature breakdowns, non-technical commentary, and Mermaid visual diagrams.
3. **[COMPLETED] End-to-End Pipeline Documentation**: Updated `README.md` with step-by-step local and AWS cloud execution guides (including SageMaker Lifecycle Configuration bootstrapping).
4. **[COMPLETED] Independent Audit of End-to-End ML Pipeline Completeness**: Conducted full code-level audit of all source files, notebooks, and AWS infrastructure. Fixed 6 critical issues (S3 path handling, missing flow features in CLI, duplicated label logic, notebook guards) and 4 moderate issues (scaler scope, lint, readability). Added `s3fs` dependency, `config.load_parquet_auto()` utility, and plain-language notebook content for non-technical audiences.
5. **[COMPLETED] Pre-Flight Production Configuration**: Configured `notebooks/02_ml_feature_engineering_training.ipynb` to use production defaults (`DEV_SYMBOLS = None` for all 20 pairs, `max_epochs = 20` for LSTM convergence). Added S3-aware data loading via `config.load_parquet_auto()`.
6. **Next Agent Action Guide (AWS Production Execution)**:
   - **Step 1**: Confirm AWS credentials & environment variables in `.env` (`EXECUTION_MODE=aws_hub`, `AWS_S3_BUCKET_NAME=dat204m-binance-bigdata-hub-sg`, `AWS_DEFAULT_REGION=ap-southeast-1`).
   - **Step 2**: If running on SageMaker, guide the user to launch a Notebook Instance (`ml.g5.xlarge` for GPU or `ml.m5.4xlarge` for CPU) using `aws/sagemaker_bootstrap.sh`.
   - **Step 3**: Execute `notebooks/02_ml_feature_engineering_training.ipynb` (or CLI `uv run python -m src.cli train-spark`) to train across all 20 symbols with 20 epochs.
   - **Step 4**: Execute `notebooks/03_ml_evaluation_error_analysis.ipynb` (or CLI `uv run python -m src.cli evaluate`) to generate final test partition evaluation metrics and visualizations.

---

## 6. Code Quality & Architectural Refactoring Master Plan

The complete 9-pillar architectural design specification is maintained in [docs/plans/refactoring_master_plan.md](docs/plans/refactoring_master_plan.md). The final post-audit refactoring items are defined in [docs/audits/final_refactors.md](docs/audits/final_refactors.md).

_Archived tasks (1-15) and detailed milestone logs are stored in [docs/session_history.md](docs/session_history.md)._
