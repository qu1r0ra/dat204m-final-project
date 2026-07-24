# Model Registry

Registry tracking trained classifiers, features, baselines, PyTorch LSTM models, and performance metrics across the 5-model evaluation ladder.

---

## 1. Machine Learning Task Configuration

- **Task Class**: Binary Price Direction Classification (Option A).
- **Target Variable**: Predict if the future return of a symbol will be positive (1) or negative/flat (0) over a horizon of $N=15$ minutes.
- **Horizon $N$**: Configured dynamically via `src/config.py` (`FUTURE_HORIZON = 15`).
- **Train/Val/Test Split**: Chronological split (70% train, 15% val, 15% test) with a 15-minute purge window to prevent target leakage.

---

## 2. Feature Engineering Pipeline

The canonical 16-feature set (`src.config.FEATURE_COLS`) comprises:

- **11 Stationary Price Features**: `close_to_sma_15`, `close_to_sma_50`, `close_to_ema_15`, `close_to_ema_50`, `bb_position`, `macd_line_norm`, `macd_signal_norm`, `macd_hist_norm`, `volatility_30`, `rsi_14`, `log_return`.
- **5 Order-Flow & Time Features**: `taker_buy_ratio`, `volume_z30`, `trades_z30`, `hour_sin`, `hour_cos`.

---

## 3. Algorithm Inventory (5-Model Ladder)

1. **Majority Class (Floor)**: Zero-intelligence baseline predicting the empirical majority class ("down").
2. **OLS Return Regression (Traditional)**: Ordinary least squares regression forecasting continuous 15m return, thresholded at 0.0.
3. **Logistic Regression (Linear ML)**: Scikit-learn LogisticRegression (`C=0.1`, L2 penalty, max_iter=1000).
4. **Random Forest Classifier (Nonlinear ML)**: Scikit-learn RandomForestClassifier (`n_estimators=100`, `max_depth=10`).
5. **PyTorch LSTM Classifier (Sequence Model)**: PyTorch 2-layer LSTM (`input_size=16`, `hidden_size=64`, `dropout=0.3`, AdamW optimizer) trained with early stopping on sequence windows `(batch, 60, 16)` constructed per symbol (`SequenceDataset`).

---

## 4. Benchmark Performance Metrics

_Results evaluated on validation partition and test partition across the 16-feature set:_

| Model | Partition | Threshold | Accuracy | Precision | Recall | F1-Score | Balanced Acc | AUC-ROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Majority Class** | Test (4.6M) | - | 0.5435 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| **OLS Return Reg** | Test (4.6M) | 0.000 | 0.5084 | 0.4612 | 0.4939 | 0.4769 | 0.5072 | 0.5012 |
| **Logistic Reg (0.5)** | Test (4.6M) | 0.500 | 0.5476 | 0.5843 | 0.0210 | 0.0405 | 0.5055 | 0.5372 |
| **Logistic Reg (tuned)** | Test (4.6M) | 0.480 | 0.5302 | 0.4862 | 0.4831 | 0.4847 | 0.5264 | 0.5372 |
| **Random Forest (0.5)** | Test (4.6M) | 0.500 | 0.5504 | 0.5521 | 0.1912 | 0.2840 | 0.5218 | 0.5510 |
| **Random Forest (tuned)** | Test (4.6M) | 0.480 | 0.5315 | 0.4877 | 0.4820 | 0.4848 | 0.5361 | 0.5510 |
| **PyTorch LSTM (tuned)** | Validation | 0.485 | 0.5328 | 0.4891 | 0.4854 | 0.4872 | 0.5370 | 0.5524 |
