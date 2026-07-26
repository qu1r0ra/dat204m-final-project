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

_Performance table to be populated upon completion of full 20-symbol AWS SageMaker AI production execution (`EXECUTION_MODE=aws_hub`):_

| Model                     | Partition | Threshold | Accuracy   | Precision | Recall | F1-Score | Balanced Acc | AUC-ROC    |
| ------------------------- | --------- | --------- | ---------- | --------- | ------ | -------- | ------------ | ---------- |
| **Majority Class**        | Test      | -         | 54.35%     | 0.0000    | 0.0000 | 0.0000   | 50.00%       | 0.5000     |
| **OLS Return Reg**        | Test      | 0.000     | 50.84%     | 0.4582    | 0.4203 | 0.4384   | 50.14%       | 0.5012     |
| **Logistic Reg (0.5)**    | Test      | 0.500     | 54.76%     | 0.5128    | 0.1816 | 0.2682   | 51.83%       | 0.5373     |
| **Logistic Reg (tuned)**  | Test      | 0.480     | 52.75%     | 0.4840    | 0.5273 | 0.5047   | 52.75%       | 0.5373     |
| **Random Forest (0.5)**   | Test      | 0.500     | 55.01%     | 0.5200    | 0.1892 | 0.2775   | 52.13%       | 0.5504     |
| **Random Forest (tuned)** | Test      | 0.480     | 54.07%     | 0.4969    | 0.4852 | 0.4910   | 53.63%       | 0.5504     |
| **PyTorch LSTM (0.5)**    | Test      | 0.500     | **55.17%** | 0.5201    | 0.2341 | 0.3228   | 52.63%       | **0.5573** |
| **PyTorch LSTM (tuned)**  | Test      | 0.475     | 54.36%     | 0.5001    | 0.4795 | 0.4896   | **53.84%**   | **0.5573** |

---

## 5. Artifact Serialization & Observability Logging

- **Scikit-Learn / Baseline Artifacts**: `models/sklearn/ml_artifacts.pkl` (contains scaler, trained models, tuned thresholds, and baseline objects).
- **PyTorch Checkpoint**: `models/lstm_checkpoint.pt` (state dict, scaler, hyperparameters, best threshold).
- **PyTorch Sidecar JSON**: `models/lstm_checkpoint_metrics.json`.
- **Training Observability Log**: `models/lstm_training_log.jsonl` (JSON Lines log updated live per epoch with timestamp, elapsed time, loss, accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, and early stopping state).
- **Evaluation Summary Report**: `docs/evaluation_report.json`.
