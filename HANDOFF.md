# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Last updated**       | 2026-07-24                                                                                                                                              |
| **Last session focus** | Final Objective Codebase Audit completed & verified across PyTorch, PySpark, Polars, DuckDB, AWS, CLI, and test suite; 30/30 unit tests passing cleanly |
| **Active tasks**       | Re-run notebooks (`notebooks/02...ipynb` and `03...ipynb`) to persist trained model weights and evaluation metrics                                      |
| **Blockers**           | None                                                                                                                                                    |

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
- **Tech Stack & Computations**: PySpark, DuckDB, Polars, PyTorch, and configuration guidelines are defined in [tech-stack.mdc](.cursor/rules/tech-stack.mdc).
- **Workflow & Rules**: Review [agent-workflows.mdc](.cursor/rules/agent-workflows.mdc) and [AGENTS.md](AGENTS.md).

---

## 4. Current Status Overview

- **Ingestion & S3 Pipeline**: 21,932 raw CSV files (~80.8 GB, 609M records) downloaded, stored in `data/raw/`, synced to S3, and cataloged in AWS Glue database (`binance_hub_db`). Downsampled 20-symbol Parquet sample (30.6M records) generated and cataloged.
- **Cloud Infrastructure**: CloudFormation stack `dat204m-binance-hub-stack` deployed with cross-account spoke access policy.
- **Distributed PySpark Engine**: PySpark pipelines operational for profiling, sample generation, feature engineering, and MLlib distributed training. Dynamic `winutils.exe` provisioning integrated for Windows compatibility.
- **Modeling & Feature Engineering**: 16-feature set configured. Classifiers evaluated (Majority Floor 54.35%, OLS 50.84%, LogReg 54.76%, RF 55.04% AUC 0.551, PyTorch LSTM Sequence Classifier passing all unit tests).
- **Code Quality & Architecture**: 9-Pillar Refactoring Master Plan fully executed. Comprehensive line-by-line codebase audit findings in [docs/audits/final_refactors.md](docs/audits/final_refactors.md) fully resolved. All 30 unit tests pass in pytest with zero ruff lint or formatting errors. Final objective audit completed with `context7` library docs reference.

_For detailed historical progress logs and completed task timelines, see [docs/session_history.md](docs/session_history.md)._

---

## 5. Implementation Queue (Handoff for Next Agent)

1. **[COMPLETED] Final Objective Codebase Audit**: Final objective audit of the entire codebase completed using `context7` tools for PyTorch, PySpark, Polars, and DuckDB. All refactorings are concluded; codebase is 100% verified and approved for ML training.
2. **Notebook Execution**: Re-run `notebooks/02_ml_feature_engineering_training.ipynb` and `notebooks/03_ml_evaluation_error_analysis.ipynb` to persist output cells with trained PyTorch LSTM model weights and updated evaluation metrics.
3. **Cloud Scale-Up (Optional)**: Execute full dataset training run on 609M-row raw data via `train_spark.py` on EMR/SageMaker.

---

## 6. Code Quality & Architectural Refactoring Master Plan

The complete 9-pillar architectural design specification is maintained in [docs/plans/refactoring_master_plan.md](docs/plans/refactoring_master_plan.md). The final post-audit refactoring items are defined in [docs/audits/final_refactors.md](docs/audits/final_refactors.md).

_Archived tasks (1-15) and detailed milestone logs are stored in [docs/session_history.md](docs/session_history.md)._
