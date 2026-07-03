# Binance K-Lines Analytics Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the Binance Spot K-Lines data and machine learning pipeline workspace.

| Field                  | Value                                                                   |
| ---------------------- | ----------------------------------------------------------------------- |
| **Last updated**       | 2026-07-03                                                              |
| **Last session focus** | Raw Dataset S3 Verification & Glue Catalog Validation                   |
| **Active tasks**       | None (All Phase 1 pipeline, indexing, and verification tasks completed) |
| **Blockers**           | None                                                                    |

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

| Topic                 | Decision                                                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Thesis Context**    | Canonical rules defined under `.cursor/rules/`.                                                                 |
| **Data Directory**    | All data (raw CSV files and Parquet files) must reside in the git-ignored `data/` directory.                    |
| **AWS Hub-and-Spoke** | Central Hub account hosts `s3://dat204m-binance-bigdata-hub-sg/` with spoke accounts granted read-only access.  |
| **Teammate Access**   | Buckets configured with bucket policies allowing teammate IDs `481088927299` and `989211373646` access.         |
| **Upload Pipeline**   | Parallelized python-boto3 multi-threaded uploader is used for raw CSV datasets to bypass missing local AWS CLI. |
| **Testing**           | Standard pytest checks run against mock CSV arrays in `tests/test_pipelines.py`.                                |

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

---

## 5. Implementation Queue

| P   | Task                                                              | Component  | Status   |
| --- | ----------------------------------------------------------------- | ---------- | -------- |
| 1   | Complete raw dataset upload to central S3 bucket                  | Data       | Complete |
| 2   | Reorganize agent context files and rules under `.cursor/rules/`   | Agent Docs | Complete |
| 3   | Execute Glue Crawler (`binance_raw_crawler`) to index raw dataset | AWS        | Complete |
| 4   | Run descriptive analysis & profile reports                        | Notebook 1 | Complete |
| 5   | Train machine learning classifiers on local sample data           | Notebook 2 | Complete |
| 6   | Conduct evaluation and error analysis                             | Notebook 3 | Complete |
