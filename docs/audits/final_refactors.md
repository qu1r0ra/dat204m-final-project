# Final Codebase Audit & Refactoring Plan (Revised)

Comprehensive line-by-line audit of every source file, test file, and configuration after completing the 9-Pillar refactoring. This is the final quality gate.

## Audit Methodology

Reviewed all 19 Python source files (4,400+ LOC), 9 test files (30 tests), `pyproject.toml`, CloudFormation template, and the refactoring master plan. Cross-referenced every public function signature against its call sites.

---

## Findings (Ordered by Severity)

### Finding 1 — CLI `evaluate --model-type lstm` Is Broken (🔴 HIGH)

The `EvaluateCommandHandler` in [cli.py:L278-285](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/cli.py#L278-L285) calls `predict_lstm` with this signature:

```python
y_prob, y_true_seq = predict_lstm(
    test_df,
    model,
    scaler,
    feature_cols,
    target_col="target",
    seq_len=seq_len,
)
```

But `predict_lstm` in [lstm.py:L359-365](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/lstm.py#L359-L365) actually expects:

```python
def predict_lstm(
    model: LSTMClassifier,
    dataset: SequenceDataset,
    batch_size: int = 2048,
    ...
) -> tuple[np.ndarray, np.ndarray]:
```

**Impact**: `python -m src.cli evaluate --model-type lstm` will crash with a `TypeError` at runtime. The CLI passes 6 positional args to a function expecting 2. Additionally, `predict_lstm` only returns `(y_probs, y_preds_default)` — it never returns ground-truth targets, but the CLI assigns the second return to `y_true_seq`.

**Fix**: Construct a `SequenceDataset` from `test_df` using the loaded `scaler`, then call `predict_lstm(model, dataset)`. Extract ground-truth targets from the dataset's `targets` array instead.

---

### Finding 2 — Spark Sessions Never Cleaned Up (🟡 MEDIUM)

`spark.stop()` is **never called** in production code. The only `spark.stop()` in the entire codebase is in the test fixture at [test_spark_pipelines.py:L82](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/tests/test_spark_pipelines.py#L82).

Affected files:

- [preprocess_spark.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/pipeline/preprocess_spark.py) — calls `get_spark_session()` at L43, never stops
- [sample_generator_spark.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/pipeline/sample_generator_spark.py) — calls `get_spark_session()` at L38, never stops
- [cli.py `TrainSparkCommandHandler`](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/cli.py#L186-L222) — calls `get_spark_session()` at L200, never stops

**Impact**: Lingering JVM processes on Windows after CLI exits. On repeated CLI invocations, orphaned Java processes consume memory.

**Fix**: Wrap Spark operations in `try ... finally: spark.stop()` in all three locations.

---

### Finding 3 — `normalize_path_str` Defined but Never Used (🟢 LOW)

[helpers.py:L13-18](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/utils/helpers.py#L13-L18) defines `normalize_path_str()`, which was created as Pillar 5 of the refactoring plan. However, **10 call sites** across the source code still use inline `.replace("\\", "/")` instead of this helper:

| File                        | Line     | Inline pattern                      |
| --------------------------- | -------- | ----------------------------------- |
| `preprocess.py`             | 45       | `csv_pattern.replace("\\", "/")`    |
| `sample_generator.py`       | 41, 43   | `str(p).replace("\\", "/")`         |
| `sample_generator_spark.py` | 44, 48   | `str(p).replace("\\", "/")`         |
| `preprocess_spark.py`       | 48       | `str(p).replace("\\", "/")`         |
| `config.py`                 | 293, 295 | `str(...).replace("\\", "/")`       |
| `cli.py`                    | 201      | `str(Path(...)).replace("\\", "/")` |

The only import of `normalize_path_str` is in [test_config_and_base.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/tests/test_config_and_base.py#L13) for testing purposes.

**Fix**: Replace all 10 inline `.replace("\\", "/")` calls with `normalize_path_str()` to fulfill Pillar 5's original intent.

---

### Finding 4 — `AWSError` Bypasses Exception Hierarchy (🟢 LOW)

[aws_client.py:L22-29](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/utils/aws_client.py#L22-L29) defines `AWSError(Exception)` and `CrawlerTimeoutError(AWSError)` as standalone exceptions that inherit from bare `Exception`, not from the Pillar 7 `BinanceAnalyticsError` hierarchy in [exceptions.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/exceptions.py).

**Fix**: Move `AWSError` into `src/exceptions.py` under `BinanceAnalyticsError`, and have `aws_client.py` import from there. This means a single `except BinanceAnalyticsError` at top-level CLI boundaries can catch all domain errors uniformly.

---

### Finding 5 — `torch.load(weights_only=False)` Warning (🔵 INFO)

[lstm.py:L457](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/lstm.py#L457) uses `weights_only=False`, which PyTorch ≥2.6 emits deprecation warnings about. This is **intentionally correct** here because the checkpoint contains a `StandardScaler` (pickle object), not just state_dict tensors. No code change needed — just adding a code comment documenting the rationale.

---

## Items I Previously Flagged That I'm Now Dropping

| Previous Item                           | Why It's Dropped                                                                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Structured JSON Logging**             | The current `%(asctime)s [%(levelname)s] %(message)s` format is perfectly adequate for an academic project. JSON structured logging adds complexity with zero benefit here.                                                           |
| **Logging Granularity Audit**           | Logging levels are already used consistently: `logger.info` for progress, `logger.error` for failures, `logger.warning` for degraded states. No actionable changes.                                                                   |
| **AWS Security Audit**                  | The CloudFormation template is solid: S3 buckets have AES256 encryption, public access fully blocked, logging bucket enabled, IAM uses managed policies (not inline), Glue crawler role is properly scoped. No security issues found. |
| **Memory Profiling (Polars vs DuckDB)** | This is an optimization exercise, not a code quality issue. Out of scope for a final refactoring audit.                                                                                                                               |
| **Model Checkpoint Migration**          | `weights_only=False` is correct and necessary (see Finding 5). No backward-compatibility issue exists with the current PyTorch 2.12.1 version.                                                                                        |

---

## Proposed Changes

### CLI Bug Fix & Interface Alignment

#### [MODIFY] [cli.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/cli.py)

- Import `SequenceDataset` from `src.models.lstm`
- In `EvaluateCommandHandler`, construct `SequenceDataset(test_df, feature_cols, target_col, seq_len, scaler)`
- Call `predict_lstm(model, test_dataset)` with correct 2-arg signature
- Extract `y_true` from `test_dataset.targets[test_dataset.valid_end_indices]`
- In `TrainSparkCommandHandler`, wrap Spark session in `try ... finally: spark.stop()`

---

### Spark Session Lifecycle Cleanup

#### [MODIFY] [preprocess_spark.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/pipeline/preprocess_spark.py)

- Wrap `spark = get_spark_session()` and all subsequent operations in `try ... finally: spark.stop()`

#### [MODIFY] [sample_generator_spark.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/pipeline/sample_generator_spark.py)

- Same `try ... finally: spark.stop()` pattern

---

### Adopt `normalize_path_str` Across Codebase

#### [MODIFY] [preprocess.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/pipeline/preprocess.py)

#### [MODIFY] [sample_generator.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/pipeline/sample_generator.py)

#### [MODIFY] [sample_generator_spark.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/pipeline/sample_generator_spark.py)

#### [MODIFY] [preprocess_spark.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/pipeline/preprocess_spark.py)

#### [MODIFY] [config.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/config.py)

#### [MODIFY] [cli.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/cli.py)

- Replace all inline `.replace("\\", "/")` with `normalize_path_str()`

---

### Unify Exception Hierarchy

#### [MODIFY] [exceptions.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/exceptions.py)

- Add `AWSError(BinanceAnalyticsError)` and `CrawlerTimeoutError(AWSError)`

#### [MODIFY] [aws_client.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/utils/aws_client.py)

- Remove local `AWSError` and `CrawlerTimeoutError` class definitions
- Import from `src.exceptions`

---

### Add Rationale Comment for `weights_only=False`

#### [MODIFY] [lstm.py](file:///c:/Users/Quirora/Documents/GitHub/dat204m-final-project/src/models/lstm.py)

- Add inline comment at L457 explaining why `weights_only=False` is intentional (checkpoint contains sklearn `StandardScaler`)

---

## Verification Plan

### Automated Tests

```bash
uv run pytest              # All 30 tests must pass
uv run ruff check .        # Zero lint violations
uv run ruff format --check # Formatting compliance
```

### Manual Verification

- Verify `python -m src.cli evaluate --help` parses correctly
- Confirm no orphaned Java processes after Spark CLI commands
