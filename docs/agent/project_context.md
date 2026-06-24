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
4. **Configuration Tooling & Dynamic Paths:**
   - Plain Python configuration (`src/config.py`) is used instead of Hydra to avoid learning curves and debugging complexities for teammates. Local configuration overrides are managed via a `.env` file (loaded using `python-dotenv`). It dynamically exposes `ACTIVE_DATA_PATH` depending on the `EXECUTION_MODE` parameter so that all notebooks switch data sources automatically.
5. **Timestamp Heterogeneity & Normalization:**
   - Raw Binance CSV data contains mixed formats: older files use millisecond timestamps (13 digits), while recent files use microsecond timestamps (16 digits).
   - The preprocessing and sample generation pipelines automatically detect and normalize raw timestamps to millisecond-based datetimes on-the-fly.
6. **Testing and Verification:**
   - Unit and integration tests are located in the `tests/` folder and run using `pytest` against mock datasets to ensure pipeline correctness.
7. **SageMaker & Environment Bootstrapping**:
   - A shell script `aws/sagemaker_bootstrap.sh` is provided to automate standard SageMaker instance setup, installing dependencies via `uv sync` and linking the custom Jupyter kernel.
8. **AWS Infrastructure Security & Compliance**:
   - The CloudFormation deployment `aws/hub_infrastructure.yaml` utilizes S3-managed encryption (SSE-S3 / `AES256`) for simplified cross-account teammate read access, and mandates bucket access logging, Object Lock (Compliance mode, 90 days), and Customer Managed IAM Policies to satisfy standard AWS security checks.
9. **Model Binary Boundaries**:
   - The `models/` directory is git-ignored. The codebase remains clean of intermediate `.pkl` model artifacts; teammates train or load them locally or via S3.

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
├── aws/                     # AWS infrastructure scripts and configurations
│   ├── s3_bucket_policy.json # Cross-account S3 bucket policy template
│   ├── hub_infrastructure.yaml # secure CloudFormation hub deployment template
│   └── sagemaker_bootstrap.sh # SageMaker bootstrap environment configuration script
├── data/                    # Git-ignored local data directory
│   ├── raw/                 # Symlinks or raw CSV directories (e.g., binance_data/)
│   └── sample/              # Compressed sample Parquet files
├── docs/                    # Written deliverables & report context
│   ├── specs.md             # Professor's Project Specifications
│   ├── team_roles.md        # Team roles and task dissemination
│   ├── data_profile.md      # Generated dataset profiling report
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
│   │   ├── __init__.py
│   │   └── indicators.py    # Polars calculations of technical indicators and stationary features
│   ├── models/              # Machine learning models & evaluation
│   │   ├── __init__.py
│   │   ├── train.py         # Chronological data split and Scikit-learn model train pipelines
│   │   └── evaluation.py    # Performance calculations, confusion matrices, and ROC curves
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── aws_client.py    # AWS S3, CloudFormation, and Glue crawler helper client
├── tests/                   # pytest suite for pipeline validation
│   └── test_pipelines.py    # Unit tests for preprocessing and sample generation
├── .env.example             # Template file for environment variable overrides
├── AGENTS.md                # Rules and workflow guidelines for AI coding assistants
├── pyproject.toml           # uv project configuration
├── uv.lock                  # uv lockfile
└── README.md                # Project README
```

---

## 5. Team Roles & Task Dissemination

To execute this project with 4 group members, roles have been structured to ensure fair distribution of work. These assignments are tentative, subject to change as the project progresses, and detailed in [team_roles.md](../team_roles.md).

- **Role 1: Cloud Infrastructure & Repository Lead** (CJ) - AWS environment configuration (S3 central hub, Glue Crawlers, cross-account resource policies), Git repository maintenance, code quality reviews, and production pipeline scale-up execution.
- **Role 2: Data Engineering & Pipeline Lead** (Teammate A) - Ingestion pipeline execution, data cleaning/deduplication with DuckDB, sampling generator implementation, and data profiling documentation.
- **Role 3: Machine Learning & Modeling Lead** (Teammate B) - ML problem definition, baseline classifier creation, technical indicator feature engineering (Polars), model training, tuning, and comparison metrics.
- **Role 4: Exploratory Analytics & Comms Lead** (Teammate C) - Running descriptive statistics, producing 5+ analytical charts, authoring the EDA and final reports, and managing presentation preparation and deadlines.
