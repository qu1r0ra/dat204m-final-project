# Project Context: Big Data & Scalable Machine Learning

This document serves as an agent-agnostic project context file to align team members and future AI agents working on this project.

---

## 1. Project Overview & Requirements

- **Course:** Introductory Big Data and Scalable Computing
- **Dataset:** Binance Spot 1-Minute K-Lines (~75+ GB raw CSVs, ~611 million rows, 558 trading pairs)
- **Goal:**
  - **Phase 1 (Descriptive Analytics):** Profile, clean, and analyze the dataset, producing descriptive stats and 5 key visualizations.
  - **Phase 2 (Predictive Analytics):** Frame a machine learning problem, train 2+ models, and compare them against baseline performance.
- **Hosting:** AWS Cloud Services

---

## 2. Selected Choices & Scope

1. **Machine Learning Task (Option A):** Binary Price Direction Classification.
   - **Target:** Predict if the return of a selected cryptocurrency ticker (configured via a parameter) will be positive (UP = 1) or negative/flat (DOWN = 0) over a future horizon $N$.
   - **Features & Horizon:** To be decided and configured later.
2. **Sampling Strategy (Option 1):**
   - Create a local sample dataset containing the **top 20 most liquid cryptocurrency pairs** at the raw **1-minute frequency** (estimated size: 1.5–2 GB in compressed Parquet format).
   - This local sample resides in `data/sample/` (git-ignored) for local development and rapid prototyping.
3. **Storage Policy:**
   - **No raw data files** are allowed in the project root. All data (raw CSVs and processed Parquet) must be stored in the git-ignored `data/` directory.
4. **Configuration Tooling:**
   - Plain Python configuration (`src/config.py`) is used instead of Hydra to avoid learning curves and debugging complexities for teammates.

---

## 3. AWS Hub-and-Spoke Architecture

To keep billing independent and simplify the experience for teammates with limited AWS knowledge:

- **The Central Hub (User's AWS Account):**
  - CENTRAL S3 BUCKET: Central S3 bucket (e.g., `dat204m-binance-bigdata-hub`).
  - `/raw/` prefix: Stores the master deduplicated dataset.
  - `/sample/` prefix: Stores the downsampled Parquet dataset.
  - Cross-Account Bucket Policy: Resource-based policy allowing teammates' specific spoke AWS Account IDs read-only access (`s3:GetObject`, `s3:ListBucket`).
  - Glue Crawler & Catalog: Crawls the S3 prefixes and creates schema-defined tables.
- **The Spokes (Teammates' AWS Accounts):**
  - Teammates query the Glue Catalog tables using **Amazon Athena** from their own AWS consoles.
  - Queries are billed to the teammates' spoke accounts, but the S3 read operations are performed against the central hub. (Free within the same region, e.g., `us-east-1`).
  - Local Prototyping: Teammates download the `/sample/` Parquet locally to write models.
- **Scale-Up Execution:**
  - Code is written using a configuration toggle (e.g., local sample vs. AWS raw).
  - Teammates run local code on `/sample/`.
  - When verified, the final code is pulled to the Hub account (e.g., in a SageMaker notebook or local runner) and run against `/raw/` to complete the scale-up.

---

## 4. Current Repository Layout

```text
dat204m-final-project/
├── data/                    # Git-ignored local data directory
│   ├── raw/                 # Symlinks or raw CSV directories (e.g., binance_data/)
│   └── sample/              # Compressed sample Parquet files
├── docs/                    # Written deliverables & report context
│   ├── specs.md             # Professor's Project Specifications
│   ├── team_roles.md        # Team roles and task dissemination
│   ├── eda_report.md        # Formatted descriptive analytics report
│   ├── final_report.md      # Final project report
│   └── agent/               # Agent-centric context files, rules, and plans
│       ├── project_context.md # This document
│       ├── implementation_plan.md # Technical implementation plan
│       ├── task.md          # Active task list tracking
│       └── plan.md          # Implementation Blueprint
├── notebooks/               # Jupyter Notebooks for deliverables
│   ├── 01_eda_descriptive_analytics.ipynb         # Phase 1: Profiling, stats, and 5 charts
│   ├── 02_ml_feature_engineering_training.ipynb   # Phase 2: Rolling indicators, model training
│   └── 03_ml_evaluation_error_analysis.ipynb      # Phase 2: Metrics, confusion matrix, recommendations
├── src/                     # Core source code package
│   ├── __init__.py
│   ├── config.py            # Parameters for symbol selection, horizons, and run modes
│   ├── pipeline/            # Data processing scripts
│   │   ├── __init__.py
│   │   ├── preprocess.py    # Profiles and cleans data/raw/ using DuckDB
│   │   ├── sample_generator.py # Deduplicates and exports sample Parquet
│   │   ├── download_klines.py # Bulk downloads raw Binance data
│   │   └── smoke_test.py    # Downloads a subset of data for testing
│   ├── features/            # Feature engineering indicators
│   │   └── __init__.py
│   ├── models/              # Machine learning models & evaluation
│   │   └── __init__.py
│   └── utils/               # Utility functions
│       └── __init__.py
├── AGENTS.md                # Rules and workflow guidelines for AI coding assistants
├── pyproject.toml           # uv project configuration
├── uv.lock                  # uv lockfile
└── README.md                # Project README
```
