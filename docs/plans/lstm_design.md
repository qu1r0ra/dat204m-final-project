# Implementation Plan — PyTorch LSTM Sequence Classifier (Revised)

Add a **PyTorch LSTM** as the 5th model in the evaluation ladder (Majority → OLS → LogReg → RF → **LSTM**) to benchmark deep learning against classical ML on 15-minute price direction prediction.

## Key Design Decisions (Revised After Code Review)

1. **GPU-accelerated**: Your machine has an **RTX 5060 with CUDA 13.2** — the LSTM will train on GPU automatically, making the ~21M-row training set feasible in minutes rather than hours.
2. **No dependency on missing modules**: Your groupmate hasn't pushed `src/features/labels.py`, `src/models/baselines.py`, or the updated `train.py` (with `compute_split_boundaries`) yet. The LSTM module will be **self-contained** — it consumes only the NumPy feature matrices (`X_train`, `X_val`) and Polars DataFrames that the notebooks already produce, so it works regardless of which `.py` files are checked in.
3. **Symbol-aware windowing from Polars DataFrames**: Instead of windowing from flat NumPy arrays (which would create cross-symbol sequences at partition boundaries), the `SequenceDataset` takes the **full Polars DataFrame** containing `symbol`, `open_time`, and feature columns, builds per-symbol index maps, and produces clean `(seq_len, 16)` windows. This avoids the subtle data leakage bug of windowing across symbols.
4. **Sequence length = 60 minutes**: Changed from 15 to **60** to give the LSTM a full hour of temporal context. The prediction target (15-min future return) doesn't constrain the lookback length — the LSTM should see enough history for its recurrent state to learn meaningful patterns (e.g., multi-candle momentum, volatility regime transitions).

> [!NOTE]
> The `seq_len=60` default is configurable in the module and notebook. If training is too slow or you prefer a shorter lookback, it can be changed to 30 or 15 with a single parameter.

---

## Proposed Changes

### Source Package (`src/`)

#### [NEW] [lstm.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/lstm.py)

Core module containing:

| Component                                               | Description                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LSTMClassifier(nn.Module)`                             | 2-layer LSTM (`input_size=16`, `hidden_size=64`, `dropout=0.3`) → Linear(64, 1). Takes `(batch, seq_len, 16)` → scalar logit.                                                                                                                                                                                              |
| `SequenceDataset(Dataset)`                              | Builds per-symbol sliding windows from a Polars DataFrame. Given df with columns `[symbol, open_time, *feature_cols, target]`, produces `(X_seq, y)` tuples where `X_seq` is `(seq_len, n_features)` and `y` is the target at the last timestep. No cross-symbol contamination.                                            |
| `train_lstm(...)`                                       | Full training loop: DataLoader creation (batch_size=2048, pin_memory for GPU), AdamW optimizer (lr=1e-3, weight_decay=1e-4), BCEWithLogitsLoss, **early stopping** on validation loss (patience=3). Returns the trained model + tuned decision threshold (identical approach to RF/LogReg threshold tuning on validation). |
| `predict_lstm(...)`                                     | Runs inference on a `SequenceDataset`, returns `(predictions, probabilities)` NumPy arrays.                                                                                                                                                                                                                                |
| `save_lstm_artifacts(...)` / `load_lstm_artifacts(...)` | `torch.save` / `torch.load` to `models/lstm_checkpoint.pt` containing model state_dict, feature_names, seq_len, threshold, and hyperparameters.                                                                                                                                                                            |

---

### Notebook Deliverables (`notebooks/`)

#### [MODIFY] [02_ml_feature_engineering_training.ipynb](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/notebooks/02_ml_feature_engineering_training.ipynb)

Add cells **after Section 5** (LogReg & RF) and **before Section 6** (Threshold Tuning):

**New Section 5.1: LSTM Deep Learning Model**

- Import `src.models.lstm` functions.
- Build `SequenceDataset` from `train_df` and `val_df` Polars DataFrames (symbol-aware windowing, `seq_len=60`).
- Train LSTM with early stopping; print epoch-by-epoch loss/accuracy progress.
- Tune decision threshold on validation probabilities (same method as LogReg/RF).
- Save checkpoint to `models/lstm_checkpoint.pt`.
- Add LSTM row to the validation comparison table.

**Update Section 6** (Threshold Tuning):

- Include LSTM threshold in the printed summary and comparison DataFrame.

**Update Section 7** (Hyperparameters & Overfitting Check):

- Add LSTM hyperparameter row to the table and train-vs-val accuracy comparison.

---

#### [MODIFY] [03_ml_evaluation_error_analysis.ipynb](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/notebooks/03_ml_evaluation_error_analysis.ipynb)

**Cell 1 (imports)**: Add LSTM imports and load `models/lstm_checkpoint.pt`.

**Section 2 (Metrics)**: Add LSTM (default 0.5) and LSTM (tuned) rows to `results_df`.

**Section 3 (Visualizations)**: Add LSTM confusion matrix; add LSTM probabilities to the ROC curve comparison.

**Section 4 (Volatility Regime)**: Add `lstm_correct` column to the regime analysis.

**Section 5 (Per-Symbol)**: Add `lstm_correct` / `lstm_acc` to the per-symbol accuracy table.

**Section 7 (Conclusion)**: Update narrative to include LSTM findings.

---

### Test Suite (`tests/`)

#### [NEW] [test_lstm.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/tests/test_lstm.py)

| Test                                    | What it verifies                                                                    |
| --------------------------------------- | ----------------------------------------------------------------------------------- |
| `test_sequence_dataset_no_cross_symbol` | Sliding windows respect symbol boundaries; no sequence mixes two symbols.           |
| `test_lstm_forward_pass_dimensions`     | Model output shape matches batch size; logits are finite floats.                    |
| `test_lstm_training_loop_convergence`   | `train_lstm` on a small synthetic dataset reduces loss and returns valid artifacts. |

Uses the existing `sample_ohlcv_df` fixture from `conftest.py`, extended with synthetic feature columns and targets.

---

## Verification Plan

### Automated Tests

```bash
uv run pytest tests/test_lstm.py -v
uv run pytest  # full suite still passes
```

### Manual Verification

- Notebook 02: LSTM trains on GPU, loss decreases, threshold is tuned, checkpoint saved.
- Notebook 03: LSTM appears in all evaluation tables, confusion matrices, ROC curves, and regime analysis alongside the existing 4 models.
