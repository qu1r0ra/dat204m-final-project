# Model Registry

Registry tracking trained classifiers, features, baselines, and performance metrics.

---

## 1. Machine Learning Task Configuration

- **Task Class**: Binary Price Direction Classification (Option A).
- **Target Variable**: Predict if the future return of a symbol will be positive (1) or negative/flat (0) over a horizon of $N$ minutes.
- **Horizon $N$**: Configured dynamically via `src/config.py` (default: 15 minutes).
- **Train/Test split**: Chronological split based on threshold date (default split date: `2024-01-01`).

---

## 2. Feature Engineering Pipeline

Features are engineered in Polars using:

- **Rolling Windows**: Technical indicators (SMAs, EMAs, RSI, Bollinger Bands).
- **Returns**: Historical log returns of open, high, low, close prices.
- **Stationarity**: Differencing and normalization to ensure stationary inputs.

---

## 3. Algorithm Inventory

We compare the following algorithms:

- **Baseline Classifier**: Dummy classifier predicting the majority class or coin-flip probabilities.
- **Model 1**: Logistic Regression (Linear baseline).
- **Model 2**: Random Forest Classifier (Tree-based ensemble).
- **Model 3**: XGBoost / LightGBM (Gradient boosting, optional scale-up).

---

## 4. Performance Log

_To be updated as models are evaluated against the test set:_

| Model         | Symbol  | Horizon | Hyperparameters  | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| ------------- | ------- | ------- | ---------------- | -------- | --------- | ------ | -------- | ------- |
| Baseline      | BTCUSDT | 15m     | Strategy: Prior  | -        | -         | -      | -        | -       |
| Logistic Reg  | BTCUSDT | 15m     | C=1.0, L2        | -        | -         | -      | -        | -       |
| Random Forest | BTCUSDT | 15m     | n_estimators=100 | -        | -         | -      | -        | -       |
