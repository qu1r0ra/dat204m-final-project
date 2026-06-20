# Big Data Analytics & Machine Learning Project

- **Minimum Data Size:** 50 GB
- **Structure:** Group Activity (3-5 Members per group)
- **Total Points:** 100 points

---

## Project Overview

Goal: Each group selects ONE large public dataset (at least 50 GB raw) and completes two phases: Descriptive Analytics to understand the data, and Machine Learning to build a predictive model. Both phases must address a coherent business or research problem.

### Phase 1: Describe (Descriptive Analytics)

- Understand and profile the dataset
- Compute descriptive statistics and identify data quality issues
- Visualize patterns and distributions
- State key exploratory findings

### Phase 2: Predict (Machine Learning)

- Define a Machine Learning target variable and engineer relevant features
- Train and evaluate a model
- Interpret predictions and errors
- Connect results to the business or research problem

### Core Deliverables

- Project proposal
- EDA report
- Final report + reproducible notebook
- Team presentation
- Peer evaluation form

---

## Dataset Requirements

- **Key Rule:** Only one group may use each dataset.
- **Raw Input Data:** Must be $>= 50\text{ GB}$. You may work on a smaller subset for analysis but must explicitly document the full dataset size.
- **Must Have:** Must be publicly accessible (free download), possess a real-world origin (not synthetic), contain multiple variables for EDA/ML, and feature a column or target suitable for machine learning prediction.
- **Consider Checking:** Schema and column documentation, update frequency (static vs streaming), licensing terms (commercial use allowed?), file formats (CSV, JSON, Parquet, logs), and download feasibility.

---

## Phase 1 Workflow Details: Descriptive Analytics

1. **Acquire & Profile:** Download or access a subset. Document row count, column types, file format, and total storage size.
2. **Clean & Prepare:** Handle missing values, duplicates, and inconsistent formats. Document every data quality decision made.
3. **Summarize:** Compute central tendency (mean, median, mode), dispersion (std dev, IQR), and distribution shape for all numeric columns.
4. **Visualize:** Produce at least 5 meaningful charts (histogram, bar, time series, heatmap, or scatter plot) suited to your dataset.
5. **Interpret Findings:** State 3 to 5 key findings that answer a specific business or research question, connecting numbers to real-world meaning.

---

## Phase 2 Workflow Details: Machine Learning

1. **Define the ML Problem:** Identify the target variable (what to predict). Determine if it is a classification, regression, or clustering problem, and state a clear prediction goal tied to the business problem.
2. **Feature Engineering:** Select, encode, and transform input features. Handle categorical variables via mechanisms like one-hot or label encoding, and scale numeric features where required by the algorithm.
3. **Train a Model:** Split data into train/test sets ($80/20$ minimum ratio). Train at least two algorithms and compare their metrics. Document the specific hyperparameters used.
4. **Evaluate Performance:** For classification, document accuracy, precision, recall, F1, and a confusion matrix. For regression, document MAE, RMSE, and $R^2$. Always compare your results to a baseline model (such as majority class or mean).
5. **Interpret & Recommend:** Explain which features matter most, discuss model limitations, and state a concrete business recommendation based on the predictive outputs.

---

## Grading Rubrics

### Phase 1 Evaluation Criteria

- **Dataset Selection (15%):** Dataset clearly justified, size verified, and problem well-defined.
- **Data Profiling (20%):** All columns documented, types verified, and an initial data sample shown.
- **Descriptive Statistics (20%):** Mean, median, std dev, IQR, and distribution shape computed for all numeric columns.
- **Visualizations (25%):** 5+ relevant charts, properly labelled, paired with thorough analytical interpretations.
- **Findings & Insights (20%):** 3 to 5 specific findings explicitly tied back to the overarching business problem.

### Phase 2 & Presentation Evaluation Criteria

- **ML Problem Definition (15%):** Target variable clearly defined, problem type justified, and baseline explicitly stated.
- **Feature Engineering (20%):** Features selected with clear rationale; encoding and scaling applied correctly.
- **Model Training & Evaluation (25%):** 2+ models compared, metrics appropriate to problem type, and a confusion matrix or error analysis included.
- **Interpretation (20%):** Feature importance discussed, errors analyzed, and concrete business recommendations stated.
- **Reproducibility & Code Quality (10%):** Notebook runs cleanly end-to-end, comments explain each step, and programmatic outputs match the written report.
- **Presentation (10%):** All members must speak, presenting a clear narrative while confidently handling the Q&A segment.

_Note: Peer evaluation is factored into individual scores; group grades are shared but individual grades may vary._

---

## Group Structure and Roles

While these roles serve as an execution guide, all members are expected to contribute across all phases of the project.

- **Data Lead:** Downloads and stores the raw dataset, profiles columns, documents the data schema, builds the data cleaning pipeline, and manages the reproducible notebook.
- **Analytics Lead:** Defines the core business or research question, runs the EDA/descriptive statistics, trains and evaluates the machine learning models, and interprets/communicates overall findings.
- **Comms Lead:** Writes the project proposal, produces the final written report, coordinates the presentation deck, and manages the team's submission deadlines.

---

## Tips for Success

- **Define the problem before touching data:** Write down _'We want to predict X because Y.'_ Every statistic and model should answer that specific question directly.
- **Descriptive first, ML second:** You cannot build a good machine learning model without understanding your data profile first. Do not skip Phase 1 to rush into modeling.
- **Use AWS for scale:** 50 GB will not fit in local RAM. Store your raw data assets in Amazon S3 and process them with SageMaker notebooks or Spark on EMR clusters.
- **Compare your ML model to a baseline:** A complex model predicting with 70% accuracy is useless if a naive majority class baseline natively yields 69%. Always state your baseline metrics.
- **Document every decision:** Explain why you removed specific columns or utilized specific techniques (e.g., median imputation). Graders evaluate your structural reasoning, not just the code outputs.
