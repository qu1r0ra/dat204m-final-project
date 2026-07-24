"""Command Line Interface (CLI) entrypoint for the Binance K-Lines analytics pipeline.

Supports subcommands for dataset profiling, sample generation, scikit-learn model training,
PyTorch LSTM model training, PySpark MLlib training, and comprehensive evaluation.
"""

import argparse
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

import src.config as config
from src.exceptions import ConfigurationError, DataValidationError
from src.features.indicators import compute_indicators, compute_stationary_features
from src.models.evaluation import (
    generate_evaluation_report,
    log_metrics,
    save_metrics_json,
)
from src.models.lstm import (
    SequenceDataset,
    load_lstm_artifacts,
    predict_lstm,
    save_lstm_artifacts,
    train_lstm,
)
from src.models.train import (
    prepare_features_and_targets,
    save_model_artifacts,
    split_data_chronologically,
    train_pipeline,
)
from src.pipeline.preprocess import run_profiling
from src.pipeline.sample_generator import generate_sample
from src.utils.helpers import normalize_path_str
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("src.cli")


def _load_and_prepare_data(
    data_path: str | Path | None = None, target_symbol: str = "BTCUSDT"
) -> pl.DataFrame:
    """Helper to load parquet dataset, filter by target symbol, compute indicators and targets."""
    path_to_load = Path(data_path) if data_path else Path(config.ACTIVE_DATA_PATH)
    if not path_to_load.exists():
        logger.error(
            f"Parquet dataset file not found at {path_to_load}. "
            "Run 'python -m src.cli sample' first or provide a valid --data-path."
        )
        raise ConfigurationError(f"Dataset path does not exist: {path_to_load}")

    logger.info(f"Loading data from {path_to_load}...")
    df = pl.read_parquet(path_to_load)
    df_target = df.filter(pl.col("symbol") == target_symbol) if "symbol" in df.columns else df

    logger.info(f"Computing technical indicators for target symbol '{target_symbol}'...")
    df_ind = compute_indicators(df_target)
    df_stat = compute_stationary_features(df_ind)

    # Compute binary classification target (future horizon)
    horizon = config.FUTURE_HORIZON
    threshold = config.TARGET_THRESHOLD
    df_prepared = (
        df_stat.with_columns(
            (pl.col("close").shift(-horizon).over("symbol") / pl.col("close") - 1.0).alias(
                "future_return"
            )
        )
        .with_columns((pl.col("future_return") > threshold).cast(pl.Int32).alias("target"))
        .drop_nulls()
    )

    logger.info(f"Prepared dataset shape: {df_prepared.shape}")
    return df_prepared


class BaseCommandHandler(ABC):
    """Abstract Base Class for CLI subcommand handlers."""

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Executes the subcommand logic."""
        pass


class ProfileCommandHandler(BaseCommandHandler):
    """Handler for dataset profiling subcommand."""

    def execute(self, args: argparse.Namespace) -> None:
        logger.info("Executing dataset profiling subcommand...")
        run_profiling()


class SampleCommandHandler(BaseCommandHandler):
    """Handler for downsampled sample generator subcommand."""

    def execute(self, args: argparse.Namespace) -> None:
        logger.info("Executing downsampled sample generator subcommand...")
        generate_sample()


class TrainSklearnCommandHandler(BaseCommandHandler):
    """Handler for scikit-learn model training subcommand."""

    def execute(self, args: argparse.Namespace) -> None:
        set_seed(args.seed)
        df = _load_and_prepare_data(data_path=args.data_path, target_symbol=args.symbol)

        train_df, val_df, _test_df = split_data_chronologically(
            df, train_end=args.train_end, val_end=args.val_end
        )

        feature_cols = config.FEATURE_COLS
        X_train, y_train = prepare_features_and_targets(train_df, feature_cols, "target")
        X_val, y_val = prepare_features_and_targets(val_df, feature_cols, "target")

        logger.info(f"Training scikit-learn classifiers with seed {args.seed}...")
        artifacts = train_pipeline(X_train, y_train, X_val, y_val, feature_cols, seed=args.seed)

        output_dir = Path(args.output_dir)
        save_model_artifacts(artifacts, output_dir)
        logger.info(
            f"Scikit-learn model training completed successfully. Artifacts saved to {output_dir}"
        )


class TrainLstmCommandHandler(BaseCommandHandler):
    """Handler for PyTorch LSTM sequence classifier subcommand."""

    def execute(self, args: argparse.Namespace) -> None:
        set_seed(args.seed)
        df = _load_and_prepare_data(data_path=args.data_path, target_symbol=args.symbol)

        train_df, val_df, _test_df = split_data_chronologically(
            df, train_end=args.train_end, val_end=args.val_end
        )

        feature_cols = config.FEATURE_COLS
        logger.info(
            f"Training PyTorch LSTM (seq_len={args.seq_len}, "
            f"epochs={args.epochs}, seed={args.seed})..."
        )

        model, scaler, best_threshold, history = train_lstm(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col="target",
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )

        hparams = {
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "seed": args.seed,
        }

        output_path = Path(args.output_file)
        save_lstm_artifacts(
            model=model,
            scaler=scaler,
            threshold=best_threshold,
            feature_cols=feature_cols,
            seq_len=args.seq_len,
            hparams=hparams,
            filepath=output_path,
            history=history,
            seed=args.seed,
        )
        logger.info(f"LSTM training completed successfully. Checkpoint saved to {output_path}")


class TrainSparkCommandHandler(BaseCommandHandler):
    """Handler for PySpark MLlib distributed training subcommand."""

    def execute(self, args: argparse.Namespace) -> None:
        set_seed(args.seed)
        from src.features.indicators_spark import compute_indicators_spark
        from src.models.train_spark import (
            compute_targets_spark,
            save_spark_models,
            split_data_chronologically_spark,
            train_pipeline_spark,
        )
        from src.utils.spark_client import get_spark_session

        spark = get_spark_session()
        try:
            data_path = normalize_path_str(Path(args.data_path or config.ACTIVE_DATA_PATH))
            logger.info(f"Loading PySpark data from {data_path}...")

            df_spark = spark.read.parquet(data_path)
            if args.symbol and "symbol" in df_spark.columns:
                df_spark = df_spark.filter(df_spark.symbol == args.symbol)

            logger.info("Computing Spark technical indicators and targets...")
            df_features = compute_indicators_spark(df_spark)
            df_labeled = compute_targets_spark(df_features)

            train_df, val_df, _test_df = split_data_chronologically_spark(
                df_labeled, train_end=args.train_end, val_end=args.val_end
            )

            feature_cols = [c for c in config.FEATURE_COLS if c in train_df.columns]
            logger.info(f"Training PySpark MLlib models with seed {args.seed}...")
            artifacts = train_pipeline_spark(train_df, val_df, feature_cols, seed=args.seed)

            output_dir = Path(args.output_dir)
            save_spark_models(artifacts, output_dir)
            logger.info(f"PySpark model training complete. Saved artifacts to {output_dir}")
        finally:
            spark.stop()


class EvaluateCommandHandler(BaseCommandHandler):
    """Handler for evaluation subcommand."""

    def execute(self, args: argparse.Namespace) -> None:
        set_seed(args.seed)
        df = _load_and_prepare_data(data_path=args.data_path, target_symbol=args.symbol)

        _train_df, _val_df, test_df = split_data_chronologically(
            df, train_end=args.train_end, val_end=args.val_end
        )

        if len(test_df) == 0:
            raise DataValidationError("Test set partition is empty!")

        feature_cols = config.FEATURE_COLS
        output_path = Path(args.output_file)

        if args.model_type == "sklearn":
            import pickle

            model_path = Path(args.model_path)
            if not model_path.exists():
                raise ConfigurationError(f"Sklearn artifact file not found: {model_path}")

            logger.info(f"Loading scikit-learn artifacts from {model_path}...")
            with open(model_path, "rb") as f:
                artifacts = pickle.load(f)

            X_test, y_test = prepare_features_and_targets(test_df, feature_cols, "target")
            X_test_scaled = artifacts.scaler.transform(X_test)

            rf_model = artifacts.random_forest
            y_pred = rf_model.predict(X_test_scaled)
            y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

            report = generate_evaluation_report(
                test_df, y_test, y_pred, y_prob, model_name="Sklearn Random Forest"
            )
            log_metrics(report["overall"], model_name="Sklearn Random Forest (Test Set)")
            save_metrics_json(report, output_path)

        elif args.model_type == "lstm":
            model_path = Path(args.model_path)
            if not model_path.exists():
                raise ConfigurationError(f"LSTM checkpoint file not found: {model_path}")

            logger.info(f"Loading LSTM checkpoint from {model_path}...")
            lstm_artifacts = load_lstm_artifacts(model_path)
            model = lstm_artifacts["model"]
            scaler = lstm_artifacts["scaler"]
            seq_len = lstm_artifacts["seq_len"]
            threshold = lstm_artifacts["threshold"]

            test_dataset = SequenceDataset(
                test_df,
                feature_cols=feature_cols,
                target_col="target",
                seq_len=seq_len,
                scaler=scaler,
            )
            y_prob, _ = predict_lstm(model, test_dataset)
            y_pred = (y_prob >= threshold).astype(int)
            y_true_seq = test_dataset.targets[test_dataset.valid_end_indices]

            test_df_seq = test_df.sort(["symbol", "open_time"])[test_dataset.valid_end_indices]
            report = generate_evaluation_report(
                test_df_seq, y_true_seq, y_pred, y_prob, model_name="PyTorch LSTM"
            )
            log_metrics(report["overall"], model_name="PyTorch LSTM (Test Set)")
            save_metrics_json(report, output_path)
        else:
            raise ConfigurationError(f"Unsupported model_type: {args.model_type}")

        logger.info(f"Evaluation report successfully written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binance K-Lines Scalable Analytics Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Command handlers mapping
    handlers: dict[str, BaseCommandHandler] = {
        "profile": ProfileCommandHandler(),
        "sample": SampleCommandHandler(),
        "train-sklearn": TrainSklearnCommandHandler(),
        "train-lstm": TrainLstmCommandHandler(),
        "train-spark": TrainSparkCommandHandler(),
        "evaluate": EvaluateCommandHandler(),
    }

    # 1. Profile command
    p_profile = subparsers.add_parser("profile", help="Run dataset profiling")
    p_profile.set_defaults(handler=handlers["profile"])

    # 2. Sample command
    p_sample = subparsers.add_parser("sample", help="Generate downsampled Parquet dataset")
    p_sample.set_defaults(handler=handlers["sample"])

    # 3. Train Sklearn command
    p_sklearn = subparsers.add_parser("train-sklearn", help="Train scikit-learn models")
    p_sklearn.add_argument("--data-path", default=None, help="Custom input Parquet path")
    p_sklearn.add_argument("--symbol", default="BTCUSDT", help="Target symbol")
    p_sklearn.add_argument("--train-end", default="2024-01-01", help="Train split end date")
    p_sklearn.add_argument("--val-end", default="2024-07-01", help="Validation split end date")
    p_sklearn.add_argument("--seed", type=int, default=42, help="Random seed")
    p_sklearn.add_argument(
        "--output-dir", default="models/sklearn", help="Artifacts output directory"
    )
    p_sklearn.set_defaults(handler=handlers["train-sklearn"])

    # 4. Train LSTM command
    p_lstm = subparsers.add_parser("train-lstm", help="Train PyTorch LSTM model")
    p_lstm.add_argument("--data-path", default=None, help="Custom input Parquet path")
    p_lstm.add_argument("--symbol", default="BTCUSDT", help="Target symbol")
    p_lstm.add_argument("--train-end", default="2024-01-01", help="Train split end date")
    p_lstm.add_argument("--val-end", default="2024-07-01", help="Validation split end date")
    p_lstm.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    p_lstm.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden dimensions")
    p_lstm.add_argument("--num-layers", type=int, default=2, help="LSTM layer count")
    p_lstm.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    p_lstm.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    p_lstm.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    p_lstm.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    p_lstm.add_argument("--seed", type=int, default=42, help="Random seed")
    p_lstm.add_argument(
        "--output-file",
        default="models/lstm_model.pt",
        help="Checkpoint output file path",
    )
    p_lstm.set_defaults(handler=handlers["train-lstm"])

    # 5. Train Spark command
    p_spark = subparsers.add_parser("train-spark", help="Train distributed PySpark models")
    p_spark.add_argument("--data-path", default=None, help="Custom input Parquet path")
    p_spark.add_argument("--symbol", default="BTCUSDT", help="Target symbol")
    p_spark.add_argument("--train-end", default="2024-01-01", help="Train split end date")
    p_spark.add_argument("--val-end", default="2024-07-01", help="Validation split end date")
    p_spark.add_argument("--seed", type=int, default=42, help="Random seed")
    p_spark.add_argument(
        "--output-dir", default="models/spark", help="Spark model output directory"
    )
    p_spark.set_defaults(handler=handlers["train-spark"])

    # 6. Evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model artifact on test partition")
    p_eval.add_argument(
        "--model-type",
        choices=["sklearn", "lstm"],
        default="sklearn",
        help="Model architecture",
    )
    p_eval.add_argument(
        "--model-path",
        default="models/sklearn/ml_artifacts.pkl",
        help="Path to trained model artifact or checkpoint",
    )
    p_eval.add_argument("--data-path", default=None, help="Custom input Parquet path")
    p_eval.add_argument("--symbol", default="BTCUSDT", help="Target symbol")
    p_eval.add_argument("--train-end", default="2024-01-01", help="Train split end date")
    p_eval.add_argument("--val-end", default="2024-07-01", help="Validation split end date")
    p_eval.add_argument("--seed", type=int, default=42, help="Random seed")
    p_eval.add_argument(
        "--output-file",
        default="docs/evaluation_report.json",
        help="Evaluation report output JSON file",
    )
    p_eval.set_defaults(handler=handlers["evaluate"])

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler.execute(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
