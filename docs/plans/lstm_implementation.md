# Implementation Plan -- PyTorch LSTM Sequence Classifier + Codebase Cleanup

Add a PyTorch LSTM as the 5th model in the evaluation ladder (Majority, OLS, LogReg, RF, LSTM), with a 5-configuration hyperparameter sweep. Includes codebase maintainability improvements identified during audit.

---

## Codebase Audit Findings

The following issues were identified during full codebase review and will be fixed alongside the LSTM work:

### Issue 1: Feature column list duplicated in 4+ places

The 16-feature list is copy-pasted independently in:

- [02_ml_feature_engineering_training.ipynb Cell 1](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/notebooks/02_ml_feature_engineering_training.ipynb) (16 features)
- [03_ml_evaluation_error_analysis.ipynb](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/notebooks/03_ml_evaluation_error_analysis.ipynb) (loaded from artifacts, but redefined in Notebook 01)
- [01_eda_descriptive_analytics.ipynb](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/notebooks/01_eda_descriptive_analytics.ipynb) (hardcoded)
- [train_spark.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/train_spark.py#L36-L48) (`DEFAULT_FEATURE_COLS` -- only 11 features, missing the 5 order-flow/time features)

**Fix**: Add `FEATURE_COLS` as a canonical constant in [config.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/config.py), then import it in the LSTM module and notebooks. This prevents drift and makes the feature set a single-source-of-truth.

### Issue 2: Model registry is stale

[model_registry.md](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/.cursor/project/model_registry.md) still lists "XGBoost / LightGBM (optional)" as Model 3, references only the original 11-feature set implicitly, and has empty performance cells. It does not reflect the current 5-model ladder, the 16-feature expansion, or the actual test results from Session 2026-07-22/23.

**Fix**: Update to reflect the full Majority, OLS, LogReg, RF, LSTM ladder with actual hyperparameters, features, and test metrics.

### Issue 3: Tech stack rules don't mention PyTorch

[tech-stack.mdc](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/.cursor/rules/tech-stack.mdc) lists scikit-learn for ML but does not include PyTorch, even though `torch==2.12.1+cu132` is already in `pyproject.toml`.

**Fix**: Add a PyTorch entry under Approved Technologies.

### Issue 4: `DEFAULT_FEATURE_COLS` in `train_spark.py` is outdated

[train_spark.py L36-48](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/train_spark.py#L36-L48) defines `DEFAULT_FEATURE_COLS` with only the original 11 features. The 5 order-flow/time features (`taker_buy_ratio`, `volume_z30`, `trades_z30`, `hour_sin`, `hour_cos`) added in the 2026-07-23 session are missing.

**Fix**: Replace the inline list with an import from `config.FEATURE_COLS`. This also fixes Issue 1.

> [!IMPORTANT]
> The Spark feature pipeline in `indicators_spark.py` would also need to compute the 5 new features before `train_spark.py` can use them. Since the Spark pipeline is not in scope for this session (it's the "optional scale-up" item in the queue), I will update the `DEFAULT_FEATURE_COLS` import but add a clear comment noting that `indicators_spark.py` needs the matching UDFs before a Spark full-data run. This avoids breaking the existing Spark tests while keeping the constant in sync.

---

## Proposed Changes

### 0. Codebase Cleanup (pre-LSTM)

---

#### [MODIFY] [config.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/config.py)

Add a canonical `FEATURE_COLS` constant (the 16-feature list) in the Machine Learning Task Parameters section, between `TARGET_THRESHOLD` and the AWS section. This becomes the single source of truth.

---

#### [MODIFY] [train_spark.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/train_spark.py)

Replace the inline `DEFAULT_FEATURE_COLS` list with `from src.config import FEATURE_COLS` and alias `DEFAULT_FEATURE_COLS = FEATURE_COLS`. Add a comment noting that `indicators_spark.py` must compute the 5 new flow/time features before a Spark run can use them.

---

#### [MODIFY] [tech-stack.mdc](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/.cursor/rules/tech-stack.mdc)

Add PyTorch line: `**Deep Learning**: Use PyTorch (version 2.x, CUDA-accelerated) for sequence models (LSTM)`

---

#### [MODIFY] [model_registry.md](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/.cursor/project/model_registry.md)

Full rewrite of Sections 2-4 to reflect: 16-feature set with order-flow/time additions, 5-model ladder (Majority, OLS, LogReg, RF, LSTM), actual test metrics from the 2026-07-22/23 sessions, LSTM entry marked as "pending" until trained.

---

### 1. Source Package (`src/models/`)

---

#### [NEW] [lstm.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/lstm.py)

Self-contained module. Imports `FEATURE_COLS` from `config.py` as the default feature list.

| Component                                                     | Detail                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LSTMClassifier(nn.Module)`                                   | 2-layer LSTM (`input_size`, `hidden_size`, `num_layers=2`, `dropout`) followed by `nn.Linear(hidden_size, 1)`. Input `(batch, seq_len, n_features)` produces a logit `(batch, 1)`. `input_size` is inferred from feature count, not hardcoded to 16.                                                                                                                                                                                                                            |
| `SequenceDataset(Dataset)`                                    | Takes a Polars DataFrame with `[symbol, open_time, *feature_cols, target_col]`. Builds per-symbol contiguous index windows of length `seq_len`. Feature values are pre-scaled via a provided `StandardScaler`. Returns `(X_seq: FloatTensor[seq_len, n_features], y: FloatTensor)`.                                                                                                                                                                                             |
| `train_lstm(train_df, val_df, feature_cols, target_col, ...)` | Accepts Polars DataFrames directly. Fits `StandardScaler` on training features. Constructs `DataLoader` (batch_size=2048, pin_memory for GPU). AdamW (`lr=1e-3`, `weight_decay=1e-4`), `BCEWithLogitsLoss`. Early stopping on val loss (patience=3). After training: scans thresholds 0.40 to 0.60 (step 0.005) on val probabilities, picks best balanced accuracy. Returns `(model, scaler, threshold, history)` where `history` is a list of per-epoch metrics for reporting. |
| `predict_lstm(model, dataset, device)`                        | Returns `(y_probs, y_preds_default)` at threshold 0.5. Caller applies tuned threshold separately.                                                                                                                                                                                                                                                                                                                                                                               |
| `save_lstm_artifacts(...)` / `load_lstm_artifacts(...)`       | `torch.save`/`torch.load` to `models/lstm_checkpoint.pt`. Stores `state_dict`, scaler, feature_cols, seq_len, threshold, hparams dict.                                                                                                                                                                                                                                                                                                                                          |

Key hyperparameters are function arguments with defaults:

```python
def train_lstm(
    train_df,
    val_df,
    feature_cols,
    target_col="target",
    seq_len=60,
    hidden_size=64,
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=2048,
    max_epochs=20,
    patience=3,
    device=None,
): ...
```

---

#### [MODIFY] [**init**.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/__init__.py)

Re-export: `LSTMClassifier`, `SequenceDataset`, `train_lstm`, `predict_lstm`, `save_lstm_artifacts`, `load_lstm_artifacts`.

---

### 2. Test Suite (`tests/`)

---

#### [NEW] [test_lstm.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/tests/test_lstm.py)

Uses `sample_ohlcv_df` fixture extended with multi-symbol synthetic data inline. Tests run on CPU with small data (seq_len=10, 4 features, 200 rows) so they complete in seconds.

| Test                                    | Verifies                                                                                                                                                              |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_sequence_dataset_no_cross_symbol` | 2-symbol DataFrame; every window contains a single symbol; no boundary-crossing sequences.                                                                            |
| `test_lstm_forward_pass_dimensions`     | `LSTMClassifier(input_size=4, hidden_size=16)` on `(batch=8, seq_len=10, 4)` yields `(8, 1)` finite logits.                                                           |
| `test_lstm_training_loop_convergence`   | `train_lstm` on synthetic DataFrame (200 rows, 2 symbols, 4 features, seq_len=10, 3 epochs) reduces loss; threshold in [0.40, 0.60]; save/load round-trips correctly. |

---

### 3. Notebook 02 -- Training + Sweep

---

#### [MODIFY] [02_ml_feature_engineering_training.ipynb](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/notebooks/02_ml_feature_engineering_training.ipynb)

**Cell 1 (imports)**: Add `from src.models.lstm import train_lstm, predict_lstm, save_lstm_artifacts, SequenceDataset` and `import src.config as config` (already present). Replace hardcoded `feature_cols` list with `feature_cols = config.FEATURE_COLS`.

**New cells between Cell 5 (split) and Cell 7 (feature matrix extraction)** -- this is critical because `train_df` and `val_df` are deleted in Cell 7:

- **New markdown cell -- Section 5.1: PyTorch LSTM Sequence Classifier**
- **New code cell -- LSTM baseline training**: Call `train_lstm(train_df, val_df, feature_cols, "target", seq_len=60)`. Print epoch-by-epoch loss/accuracy. Save result.
- **New markdown cell -- Section 5.2: LSTM Hyperparameter Sensitivity**
- **New code cell -- Sweep loop**: Loop through 4 alternative configs (B-E from below), training each. Collect val balanced accuracy and AUC. Print comparison table. Select best config. Save winning checkpoint to `models/lstm_checkpoint.pt`.

Sweep configurations:

| Config       | `seq_len` | `hidden_size` | `dropout` | Rationale                                  |
| ------------ | --------- | ------------- | --------- | ------------------------------------------ |
| A (baseline) | 60        | 64            | 0.3       | Default: 1hr lookback, moderate capacity   |
| B            | 30        | 64            | 0.3       | Shorter context: is 30 min sufficient?     |
| C            | 120       | 64            | 0.3       | Longer context: does 2 hours help?         |
| D            | 60        | 128           | 0.3       | More capacity: is the model underfitting?  |
| E            | 60        | 64            | 0.5       | Heavier dropout: is the model overfitting? |

**Cell 11 (LR/RF comparison table)**: Add LSTM (best config) row to `comparison_df`.

**Cell 13 (threshold tuning table)**: Add LSTM threshold row.

**Cell 15 (overfitting check)**: Add LSTM train-vs-val accuracy.

**Section 7 markdown (hyperparameters table)**: Add LSTM row with the selected sweep winner's hyperparameters.

---

### 4. Notebook 03 -- Evaluation

---

#### [MODIFY] [03_ml_evaluation_error_analysis.ipynb](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/notebooks/03_ml_evaluation_error_analysis.ipynb)

**Cell 1 (imports)**: Add `from src.models.lstm import load_lstm_artifacts, predict_lstm, SequenceDataset`. Load checkpoint. Import `config.FEATURE_COLS` for consistency.

**Cell 3 (test partition)**: After building `test_df`, construct `SequenceDataset` for the test partition using the loaded scaler. Run `predict_lstm` to get `lstm_probs`. Derive `lstm_preds` (default 0.5) and `lstm_tuned_preds` (loaded threshold). Note: because `SequenceDataset` drops the first `seq_len-1` rows per symbol, `lstm_probs` will be shorter than `y_test`. The code must align indices. The cleanest approach: track which test rows have LSTM predictions (those with enough history) and use the aligned subset for LSTM metrics, while the full `y_test` is used for all other models as before.

**Cell 5 (metrics table)**: Add `"LSTM (0.5)"` and `"LSTM (tuned)"` rows to `results_df` using the aligned subset.

**Cell 8 (visualizations)**: Add LSTM confusion matrix. Add `"LSTM": lstm_probs` to `plot_roc_curves(...)`.

**Cell 10 (volatility regime)**: Add `lstm_correct` column and print LSTM low/high volatility accuracy.

**Cell 13 (per-symbol)**: Add `lstm_correct` and `lstm_acc` to per-symbol aggregation.

**Cell 18 (conclusion markdown)**: Add LSTM findings paragraph. Preserve existing text.

**Cell 19 (references)**: Add PyTorch citation.

---

### 5. Documentation Updates

---

#### [MODIFY] [HANDOFF.md](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/HANDOFF.md)

- Update "Last updated", "Last session focus", "Active tasks"
- Add implementation completion entry under the PyTorch LSTM session section
- Update Implementation Queue: mark item 1 as done, keep items 2-3
- Note the sweep results and winning config

#### [MODIFY] [docs/plans/lstm.md](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/docs/plans/lstm.md)

Add the hyperparameter sweep section (5 configs) and the codebase cleanup items to the plan for traceability.

#### [MODIFY] [model_registry.md](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/.cursor/project/model_registry.md)

Rewrite to reflect current state: 5-model ladder, 16 features, actual metrics, LSTM hyperparameters.

---

## Design Decision: `seq_len=60` as default

Going with `seq_len=60` (1 hour) as the default baseline. The sweep covers `30` and `120` as alternatives. The winning configuration from the sweep becomes the final saved checkpoint.

---

## Design Decision: Index alignment for LSTM evaluation in Notebook 03

`SequenceDataset` needs `seq_len - 1` rows of history before producing the first prediction. For `seq_len=60`, this means the first 59 rows per symbol in the test partition have no LSTM prediction. Two approaches:

1. **Aligned subset**: LSTM metrics are computed on the subset of test rows that have predictions. Other models' metrics remain on the full test set. Results are reported with a note on the different sample sizes.
2. **Recompute all models on the aligned subset**: All models are re-evaluated on the same rows that LSTM can predict on. Ensures apples-to-apples comparison but changes the existing reported numbers slightly.

Going with **option 2** (aligned subset for all models) for fairness. The LSTM drops at most 59 rows per symbol from a ~4.6M row test set (19 symbols \* 59 = 1,121 rows, or 0.02%). The impact on other models' metrics is negligible, but the comparison is clean.

---

## Verification Plan

### Automated Tests

```bash
uv run ruff format src/models/lstm.py tests/test_lstm.py
uv run ruff check src/models/lstm.py tests/test_lstm.py
uv run pytest tests/test_lstm.py -v
uv run pytest  # full suite still passes
```

### Manual Verification

- Notebook 02: LSTM trains on GPU, sweep table printed, best checkpoint saved to `models/lstm_checkpoint.pt`, LSTM row in all comparison tables
- Notebook 03: LSTM in metrics table, confusion matrix, ROC curve, volatility regime, per-symbol breakdown, conclusion
- `config.FEATURE_COLS` imported successfully by LSTM module and notebooks
- Model registry reflects the current 5-model ladder
