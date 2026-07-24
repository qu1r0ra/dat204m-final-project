# Code Quality & Architectural Refactoring Master Plan

This document defines the architectural refactoring targets and completed design specifications to maximize overall code quality, maintainability, type safety, testability, and software design principles across the Binance K-Lines Analytics workspace.

---

## 9-Pillar Architectural Refactoring Overview

### Pillar 1: Type-Safe Immutable Configuration & Dependency Injection (`src/config.py`)

- **Problem**: `src/config.py` relies on module-level global variables initialized on import via `os.getenv()`. Changing variables requires monkeypatching module globals during tests, which can introduce state leakage and side effects.
- **Solution**:
  - Implement an immutable `@dataclass(frozen=True)` `PipelineConfig` class.
  - Support `PipelineConfig.from_env()`, `PipelineConfig.for_testing()`, and explicit parameter overrides.
  - Refactor pipeline functions (`generate_sample(config)`, `run_profiling(config)`, `train_pipeline(..., config)`) to take `config` as an explicit parameter (Dependency Injection).

---

### Pillar 2: Unified Model Trainer Contract & Artifact Protocol (`src/models/base.py`)

- **Problem**: `train.py` (scikit-learn), `train_spark.py` (PySpark MLlib), and `lstm.py` (PyTorch) use different function signatures and return different objects (`ModelArtifacts`, `SparkModelArtifacts`, tuple `(model, scaler, threshold, history)`).
- **Solution**:
  - Create `src/models/base.py` defining an abstract trainer protocol (`BaseModelTrainer`).
  - Standardize a unified `TrainerResult` dataclass containing `model`, `scaler`, `metrics`, `provenance`, `threshold`, `feature_names`, and `hparams`.
  - Provide unified model saving and loading routines (`save_trainer_result`, `load_trainer_result`).

---

### Pillar 3: Cross-Engine Feature Parity & Verification Suite (`src/features/`)

- **Problem**: Rolling technical indicators are calculated in Polars ([indicators.py](../../src/features/indicators.py)) and in PySpark ([indicators_spark.py](../../src/features/indicators_spark.py)) independently without runtime cross-validation.
- **Solution**:
  - Add `validate_feature_parity()` in `src/features/indicators.py` to assert mathematical equivalence between Polars rolling calculations and PySpark SQL/UDF calculations within tight numerical tolerances ($10^{-5}$).
  - Enforce explicit dtypes schemas on all CSV reads to prevent runtime type coercion overhead.

---

### Pillar 4: Command Pattern for CLI Subcommands (`src/cli.py`)

- **Problem**: `src/cli.py` mixes argument parsing definitions with inline subcommand execution functions.
- **Solution**:
  - Decouple subcommand handlers into modular command handler classes (`ProfileCommand`, `SampleCommand`, `TrainCommand`, `EvaluateCommand`).
  - Keep `src/cli.py` purely responsible for `argparse` configuration and command routing.

---

### Pillar 5: Centralized Cross-Platform Path Normalization (`src/utils/helpers.py`)

- **Problem**: Inconsistent handling of Windows backslashes (`\`) vs POSIX forward slashes (`/`) when reading S3 URIs, Spark paths, or DuckDB tables across platforms.
- **Solution**:
  - Implement `normalize_path_str()` helper in `src/utils/helpers.py` to convert path inputs into normalized POSIX-style strings across Windows, Linux, DuckDB, PySpark, and S3 URIs.

---

### Pillar 6: Extended Unit Test Suite (`tests/test_config_and_base.py`)

- **Problem**: Lack of automated tests verifying immutable configuration, trainer interfaces, and cross-engine feature parity.
- **Solution**:
  - Add `tests/test_config_and_base.py` covering `PipelineConfig` initialization, `TrainerResult` serialization, `normalize_path_str`, and feature parity assertions across test matrices.

---

### Pillar 7: Exception Hierarchy & Defensive Validation (`src/exceptions.py`)

- **Problem**: Pipeline routines throw standard `ValueError` or generic exceptions without structured domain error context.
- **Solution**:
  - Define custom hierarchy under `BinanceAnalyticsError` (`ConfigurationError`, `DataValidationError`, `SparkPipelineError`, `ModelTrainingError`).
  - Enforce defensive input checks (sequence length validations, missing feature columns, empty dataset assertions).

---

### Pillar 8: Seed Reproducibility & Provenance Tracking (`src/utils/seed.py`)

- **Problem**: Scattered `random.seed()` or `np.random.seed()` calls lead to non-deterministic runs across PyTorch, PySpark, NumPy, Python, and scikit-learn.
- **Solution**:
  - Implement centralized `set_seed()` locking random seeds across all libraries.
  - Implement `get_provenance_metadata()` recording system details, timestamps, and dependency package versions into model checkpoints.

---

### Pillar 9: Consolidated Metric Reports (`src/models/evaluation.py`)

- **Problem**: Evaluation reports were generated ad-hoc in notebooks without a unified output schema.
- **Solution**:
  - Implement `generate_evaluation_report()` producing standardized multi-dimensional metric summaries (overall, per-symbol, volatility regime splits, confusion matrix).
