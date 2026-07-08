# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                                            |
| ---------------------- | ---------------------------------------------------------------- |
| **Last updated**       | 2026-07-08                                                       |
| **Last session focus** | PySpark JVM Configuration & Java 26 Patch Cleanup                |
| **Active tasks**       | Verify Spark test execution and generate Spark profiling reports |
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
- **Testing**: Unified Spark test suite implemented in [test_spark_pipelines.py](tests/test_spark_pipelines.py).

---

## 5. Implementation Queue

| P   | Task                                                               | Component | Status      |
| --- | ------------------------------------------------------------------ | --------- | ----------- |
| 8   | Verify local Java 21 execution for Spark and generate data profile | Spark ML  | In Progress |

_For archived tasks (1-7), see [session_history.md](docs/session_history.md)._
