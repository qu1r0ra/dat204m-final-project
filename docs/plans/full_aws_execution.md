# AWS SageMaker AI Production Execution Guide

This guide provides step-by-step, unambiguous instructions for executing the full production pipeline on **AWS SageMaker AI** (the current official console name for Amazon SageMaker).

> [!NOTE]
> **Service Clarification**: **Amazon SageMaker AI** and **Amazon SageMaker** refer to the exact same AWS service. AWS updated the console branding to **Amazon SageMaker AI**.

---

## Target Hardware & Environment Configuration

- **Instance Type**: `ml.g4dn.2xlarge` (8 vCPUs, 32 GB RAM, 1 NVIDIA T4 GPU with 16 GB VRAM) — _Accelerated Computing_
- **Platform Identifier**: `Amazon Linux 2023, Jupyter Lab 4`
- **Execution Mode**: Cloud Hub (`EXECUTION_MODE=aws_hub`)
- **S3 Bucket**: `dat204m-binance-bigdata-hub-sg`
- **AWS Region**: `ap-southeast-1`
- **SageMaker IAM Role**: `SageMaker-sagemaker-binance-hub-role-2`
- **Role ARN**: `arn:aws:iam::872891100013:role/service-role/SageMaker-sagemaker-binance-hub-role-2`

### Projected Execution Timeline (`ml.g4dn.2xlarge`)

| Pipeline Phase                                     | Description                                                                                                                                    | Computing Acceleration                | Projected Time    |
| :------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------ | :---------------- |
| **Phase 1: EDA & Descriptive Analytics**           | Data loading, missingness checks, volume/volatility profiling (`01_eda_descriptive_analytics.ipynb`)                                           | 8 vCPUs (Polars / PySpark)            | ~3 – 5 mins       |
| **Phase 2: Feature Engineering & Baseline Models** | 16-indicator engineering, chronological train/val/test split, Logistic Regression & Random Forest (`02_ml_feature_engineering_training.ipynb`) | 8 vCPUs multi-threading (`n_jobs=-1`) | ~3 – 4 mins       |
| **Phase 2: PyTorch LSTM Deep Learning**            | 20-epoch training sweep on ~30.6M sequence windows with early stopping & threshold tuning (`02_ml_feature_engineering_training.ipynb`)         | NVIDIA T4 GPU (`cuda`)                | ~10 – 12 mins     |
| **Phase 3: Model Evaluation & Error Analysis**     | Predictions, ROC/PR curves, confusion matrices, regime summary & JSON export (`03_ml_evaluation_error_analysis.ipynb`)                         | 8 vCPUs / T4 GPU                      | ~2 – 3 mins       |
| **Total Pipeline**                                 | **End-to-end notebook pipeline execution**                                                                                                     | **Accelerated GPU + CPU**             | **~18 – 25 mins** |

---

## Dataset Architecture & S3 Storage

The project dataset in AWS S3 is structured into two main tiers:

| Tier                           | Path                                                                | Contents & Size                                                                                                                                     | Target Pipeline Phase                                                                                                                                                           |
| :----------------------------- | :------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Full Raw CSV Dataset**       | `s3://dat204m-binance-bigdata-hub-sg/raw/`                          | **21,932 raw CSV files (~80.8 GB, ~609 million rows, 558 trading pairs)** downloaded over 3 years.                                                  | **Phase 1 (Descriptive Profiling & Big Data Sampling)**: Processed by PySpark (`preprocess_spark.py`) to generate data profiling metrics across all coins.                      |
| **Curated ML Parquet Dataset** | `s3://dat204m-binance-bigdata-hub-sg/sample/binance_sample.parquet` | **30,650,000 rows (~1.03 GB compressed Parquet)** containing all 1-minute k-lines for the **top 20 liquid cryptocurrency USDT pairs** over 3 years. | **Phase 2 (Predictive Analytics & Model Training)**: Filters out stablecoins and illiquid dead coins to train canonical ML models (Random Forest, PySpark MLlib, PyTorch LSTM). |

---

## Step-by-Step AWS SageMaker AI Pipeline Execution

### Step 0: Set Up SageMaker IAM Role via Amazon SageMaker AI Role Manager

To create a persona-based IAM Execution Role with bucket access using **Amazon SageMaker Role Manager**:

1. Log in to the **Amazon SageMaker AI Console**.
2. In the left navigation menu (under **Admin configurations** or **Governance**), click **Role Manager**.
3. Click **Create role**.
4. **Step 1: Role Details**:
   - **Role name**: `SageMaker-sagemaker-binance-hub-role-2`
   - **Persona**: Select **Data Scientist** (pre-configures ML activities for notebook access, S3 data loading, model training, and experiment tracking).
   - Click **Next**.
5. **Step 2: Configure ML Activities**:
   - Ensure the **Access S3 buckets** and **Run processing/training jobs** activities are enabled.
   - Click **Next**.
6. **Step 3: Network and Encryption**:
   - Keep default VPC/encryption settings. Click **Next**.
7. **Step 4: Add Additional Policies & S3 Access**:
   - Under **S3 bucket access**, add custom bucket access for `dat204m-binance-bigdata-hub-sg`.
   - Alternatively, attach a Customer Managed Policy granting explicit read/write access:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Action": [
             "s3:GetObject",
             "s3:PutObject",
             "s3:ListBucket",
             "s3:DeleteObject"
           ],
           "Resource": [
             "arn:aws:s3:::dat204m-binance-bigdata-hub-sg",
             "arn:aws:s3:::dat204m-binance-bigdata-hub-sg/*"
           ]
         }
       ]
     }
     ```
8. **Step 5: Review & Submit**:
   - Review permissions and click **Submit**. Created role ARN: `arn:aws:iam::872891100013:role/service-role/SageMaker-sagemaker-binance-hub-role-2`.

---

### Step 1: Launch Notebook Instance in AWS SageMaker AI

1. Go to **Notebook** -> **Notebook instances** -> Click **Create notebook instance**.
2. Settings:
   - **Notebook instance name**: `dat204m-production-node`
   - **Notebook instance type**: `ml.g4dn.2xlarge` (8 vCPUs, 32 GB RAM, 1 NVIDIA T4 GPU)
   - **Platform identifier**: `Amazon Linux 2023, Jupyter Lab 4`
   - **Volume size (GB)**: Set to **`35 GB`** (or `50 GB` - required for PyTorch CUDA wheels and PySpark packages).
3. Under **Git repositories** (optional automated clone):
   - Select **Clone a public Git repository** and enter `https://github.com/qu1r0ra/dat204m-final-project.git`.
4. Under **Permissions and encryption**:
   - **IAM Role**: Select `SageMaker-sagemaker-binance-hub-role-2` (`arn:aws:iam::872891100013:role/service-role/SageMaker-sagemaker-binance-hub-role-2`).
5. Click **Create notebook instance**.
6. Wait ~3–5 minutes until Status changes to **InService**.

---

### Step 2: Open JupyterLab & Bootstrap Environment

1. Next to `dat204m-production-node`, click **Open JupyterLab**.
2. Open a Terminal in JupyterLab (`File` -> `New` -> `Terminal`).
3. If the repository was not automatically attached, clone it into persistent storage:
   ```bash
   cd ~/SageMaker
   git clone https://github.com/qu1r0ra/dat204m-final-project.git
   ```
4. Navigate into the project and install dependencies / register the Jupyter kernel:

   ```bash
   cd ~/SageMaker/dat204m-final-project
   export PATH="$HOME/.local/bin:$PATH"

   # 1. Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

   # 2. Sync virtual environment & register Jupyter kernel
   uv sync
   uv run python -m ipykernel install --user --name="dat204m-final-project" --display-name="Python (DAT204M)"

   # 3. Create .env configuration file for AWS Hub execution mode
   cat << 'EOF' > .env
   EXECUTION_MODE=aws_hub
   SPARK_EXECUTION_MODE=local
   AWS_DEFAULT_REGION=ap-southeast-1
   AWS_S3_BUCKET_NAME=dat204m-binance-bigdata-hub-sg
   YEARS_OF_HISTORY=3
   DATA_FREQUENCY=1m
   TARGET_SYMBOL=BTCUSDT
   FUTURE_HORIZON=15
   TARGET_THRESHOLD=0.0
   TRAIN_SPLIT_DATE=2024-01-01
   EOF
   ```

5. Open your target notebook (e.g. `notebooks/01_eda_descriptive_analytics.ipynb`).
6. In the top-right kernel selector dropdown, select **`Python (DAT204M)`**.

---

### Step 4: Execute Phase 1 (Full 609M Row Dataset Profiling & Visualizations)

1. Open notebook [`notebooks/01_eda_descriptive_analytics.ipynb`](../../notebooks/01_eda_descriptive_analytics.ipynb).
2. Select **Run** -> **Run All Cells**.
3. Profiles dataset metrics, volume distributions, volatility regimes, and cross-asset correlations across all pairs.

---

### Step 5: Execute Phase 2 (Machine Learning Feature Engineering & Training)

1. Open notebook [`notebooks/02_ml_feature_engineering_training.ipynb`](../../notebooks/02_ml_feature_engineering_training.ipynb).
2. Confirm production parameters in Cell 2:
   - `EXECUTION_MODE = "aws_hub"`
   - `DEV_SYMBOLS = None` (processes all 20 liquid cryptocurrency trading pairs: 30.65M rows)
   - `max_epochs = 20` (full PyTorch LSTM sweep; automatically uses CPU execution)
3. Select **Run** -> **Run All Cells**.
4. Model artifacts are saved to `models/sklearn/ml_artifacts.pkl` and `models/lstm_model.pt`.

---

### Step 6: Execute Phase 2 Evaluation & Error Analysis

1. Open notebook [`notebooks/03_ml_evaluation_error_analysis.ipynb`](../../notebooks/03_ml_evaluation_error_analysis.ipynb).
2. Select **Run** -> **Run All Cells**.
3. Renders ROC curves, confusion matrices, and exports metrics to `docs/evaluation_report.json`.

---

## Output Verification Checklist

Upon completion, verify the following files are updated in the workspace:

- [x] Model Artifact: `models/sklearn/ml_artifacts.pkl`
- [x] PyTorch Checkpoint: `models/lstm_model.pt`
- [x] Metrics Report: `docs/evaluation_report.json`
