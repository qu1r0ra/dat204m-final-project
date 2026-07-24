# AWS SageMaker AI Production Execution Guide

This guide provides step-by-step, unambiguous instructions for executing the full production pipeline on **AWS SageMaker AI** (the current official console name for Amazon SageMaker).

> [!NOTE]
> **Service Clarification**: **Amazon SageMaker AI** and **Amazon SageMaker** refer to the exact same AWS service. AWS updated the console branding to **Amazon SageMaker AI**.

---

## Dataset Architecture & S3 Storage

The project dataset in AWS S3 is structured into two main tiers:

| Tier                           | Path                                                                | Contents & Size                                                                                                                                     | Target Pipeline Phase                                                                                                                                                           |
| :----------------------------- | :------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Full Raw CSV Dataset**       | `s3://dat204m-binance-bigdata-hub-sg/raw/`                          | **21,932 raw CSV files (~80.8 GB, ~609 million rows, 558 trading pairs)** downloaded over 3 years.                                                  | **Phase 1 (Descriptive Profiling & Big Data Sampling)**: Processed by PySpark (`preprocess_spark.py`) to generate data profiling metrics across all coins.                      |
| **Curated ML Parquet Dataset** | `s3://dat204m-binance-bigdata-hub-sg/sample/binance_sample.parquet` | **30,650,000 rows (~1.03 GB compressed Parquet)** containing all 1-minute k-lines for the **top 20 liquid cryptocurrency USDT pairs** over 3 years. | **Phase 2 (Predictive Analytics & Model Training)**: Filters out stablecoins and illiquid dead coins to train canonical ML models (Random Forest, PySpark MLlib, PyTorch LSTM). |

---

## Environment & S3 Status

- **AWS Execution Mode**: Cloud Hub (`EXECUTION_MODE=aws_hub`).
- **AWS Account Identity**: `arn:aws:iam::872891100013:user/qu1r0ra` (Region: `ap-southeast-1`).
- **S3 Bucket**: `dat204m-binance-bigdata-hub-sg`

---

## Step-by-Step AWS SageMaker AI Pipeline Execution

### Step 1: Create Lifecycle Configuration in AWS SageMaker AI

1. Log in to the **AWS Management Console** and search for **Amazon SageMaker AI**.
2. In the left-hand navigation pane under **Admin configurations** (or **Notebook**), click **Lifecycle configurations**.
3. Click **Create configuration**.
4. Set the details:
   - **Type**: Select **Notebook instance**.
   - **Name**: `dat204m-bootstrap-lcc`
5. In the **Start notebook** script tab (or **Create notebook**), paste the exact script contents of [`aws/sagemaker_bootstrap.sh`](../../aws/sagemaker_bootstrap.sh).
6. Click **Create configuration**.

> [!TIP]
> `aws/sagemaker_bootstrap.sh` automatically installs `uv`, syncs project dependencies (including `s3fs` and `pyspark`), and registers the custom **`Python (DAT204M)`** kernel inside Jupyter.

---

### Step 2: Launch Notebook Instance in AWS SageMaker AI

1. In the **Amazon SageMaker AI** console left menu, go to **Notebook** -> **Notebook instances**.
2. Click **Create notebook instance**.
3. Configure the instance:
   - **Notebook instance name**: `dat204m-production-node`
   - **Notebook instance type**: `ml.g5.xlarge` (for GPU acceleration) or `ml.m5.4xlarge` (for high-CPU compute).
   - **Platform identifier**: `Amazon Linux 2, Jupyter Lab`
4. Expand **Additional configuration**:
   - **Lifecycle configuration**: Select `dat204m-bootstrap-lcc`.
5. Under **Permissions and encryption**:
   - **IAM Role**: Choose an IAM role with S3 read/write permissions to bucket `dat204m-binance-bigdata-hub-sg`.
6. Click **Create notebook instance**.
7. Wait ~3–5 minutes until the Instance Status turns **InService**.

---

### Step 3: Open JupyterLab & Select Kernel

1. Next to `dat204m-production-node`, click **Open JupyterLab**.
2. Open the repository folder: `dat204m-final-project`.
3. In JupyterLab, check the kernel selector dropdown (top right) and select **`Python (DAT204M)`**.

---

### Step 4: Execute Phase 1 (Full 609M Row Dataset Profiling & Visualizations)

1. Open notebook [`notebooks/01_eda_descriptive_analytics.ipynb`](../../notebooks/01_eda_descriptive_analytics.ipynb).
2. Select **Run** -> **Run All Cells**.
3. The notebook profiles the full dataset metrics, volume distributions, volatility regimes, and cross-asset correlations across all pairs.

---

### Step 5: Execute Phase 2 (Machine Learning Feature Engineering & Training)

1. Open notebook [`notebooks/02_ml_feature_engineering_training.ipynb`](../../notebooks/02_ml_feature_engineering_training.ipynb).
2. Confirm production configuration parameters in Cell 2:
   - `EXECUTION_MODE = "aws_hub"`
   - `DEV_SYMBOLS = None` (processes all 20 liquid cryptocurrency trading pairs: 30.65M rows)
   - `max_epochs = 20` (full PyTorch LSTM convergence sweep)
3. Select **Run** -> **Run All Cells**.
4. The notebook will:
   - Fetch data directly from `s3://dat204m-binance-bigdata-hub-sg/sample/binance_sample.parquet` using `load_parquet_auto()`.
   - Calculate 16 technical features across 30.65M rows.
   - Train baseline classifiers, Scikit-Learn models, PySpark MLlib, and PyTorch LSTM.
   - Save trained model checkpoints to `models/sklearn/ml_artifacts.pkl` and `models/lstm_model.pt`.

---

### Step 6: Execute Phase 2 Evaluation & Error Analysis

1. Open notebook [`notebooks/03_ml_evaluation_error_analysis.ipynb`](../../notebooks/03_ml_evaluation_error_analysis.ipynb).
2. Select **Run** -> **Run All Cells**.
3. The notebook will:
   - Evaluate all models on held-out test data (Jan 2024 - July 2024).
   - Render ROC curves, Precision-Recall curves, confusion matrices, and feature importance bar charts.
   - Export evaluation metrics to `docs/evaluation_report.json`.

---

## Output Verification Checklist

Upon completion, verify the following files are updated in the workspace:

- [x] Model Artifact: `models/sklearn/ml_artifacts.pkl`
- [x] PyTorch Checkpoint: `models/lstm_model.pt`
- [x] Metrics Report: `docs/evaluation_report.json`
