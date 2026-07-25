"""
Distributed machine learning model training pipeline using PySpark MLlib.

Implements chronological splitting, feature assembling, scaling, and training
of distributed classifiers (Logistic Regression and Random Forest) on Spark.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import src.config as config
from src.exceptions import DataValidationError, SparkPipelineError
from src.utils.seed import get_provenance_metadata, set_seed

logger = logging.getLogger(__name__)


@dataclass
class SparkModelArtifacts:
    logistic_regression: PipelineModel
    random_forest: PipelineModel
    feature_names: list[str]
    metrics: dict[str, dict[str, Any]]
    provenance: dict[str, Any] | None = None

    def to_trainer_result(self) -> Any:
        from src.models.base import TrainerResult

        return TrainerResult(
            model_name="Spark MLlib Ensemble",
            model={
                "logistic_regression": self.logistic_regression,
                "random_forest": self.random_forest,
            },
            metrics=self.metrics,
            provenance=self.provenance or {},
            feature_names=self.feature_names,
        )


from src.config import FEATURE_COLS

# Default feature columns sourced from canonical config.
# Note: indicators_spark.py must compute the 5 order-flow/time UDF features before a full Spark run.
DEFAULT_FEATURE_COLS = FEATURE_COLS


def compute_targets_spark(
    df_features: DataFrame, feature_cols: list[str] | None = None
) -> DataFrame:
    """Computes binary movement target label using lead window function."""
    logger.info("Computing target labels using Spark Window lead...")

    window_spec = Window.partitionBy("symbol").orderBy("open_time")

    # Get future price shift
    df_labeled = df_features.withColumn(
        "future_close", F.lead("close", config.FUTURE_HORIZON).over(window_spec)
    )

    # Calculate returns
    df_labeled = df_labeled.withColumn(
        "future_return", (F.col("future_close") / F.col("close")) - 1.0
    )

    # Create binary class: 1 if return > threshold else 0
    df_labeled = df_labeled.withColumn(
        "target",
        F.when(F.col("future_return") > config.TARGET_THRESHOLD, 1).otherwise(0),
    )

    # Drop rows with null values resulting from lead or warm-up periods
    # To be conservative, we drop any rows containing nulls in available features or target column
    cols = feature_cols if feature_cols is not None else DEFAULT_FEATURE_COLS
    available_cols = [c for c in cols if c in df_features.columns]

    drop_cols = list(set(available_cols + ["target"]))
    df_clean = df_labeled.dropna(subset=drop_cols)

    return df_clean


def split_data_chronologically_spark(
    df: DataFrame, train_end: str, val_end: str
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Splits Spark DataFrame chronologically using timestamp filters."""
    logger.info(
        f"Splitting Spark data chronologically: train_end={train_end}, val_end={val_end}"
    )

    train_end_ts = F.lit(train_end).cast("timestamp")
    val_end_ts = F.lit(val_end).cast("timestamp")

    train_df = df.filter(F.col("open_time") < train_end_ts)
    val_df = df.filter(
        (F.col("open_time") >= train_end_ts) & (F.col("open_time") < val_end_ts)
    )
    test_df = df.filter(F.col("open_time") >= val_end_ts)

    return train_df, val_df, test_df


def train_pipeline_spark(
    train_df: DataFrame,
    val_df: DataFrame,
    feature_cols: list[str],
    seed: int = 42,
) -> SparkModelArtifacts:
    """Trains Spark MLlib Logistic Regression and Random Forest models in a pipeline."""
    if train_df.isEmpty():
        raise DataValidationError("Training DataFrame is empty!")
    if val_df.isEmpty():
        raise DataValidationError("Validation DataFrame is empty!")

    set_seed(seed)
    logger.info("Building Spark ML Pipeline (Assembler + Scaler)...")

    # 1. Assemble features into a single vector column
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip"
    )

    # 2. Scale features
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withStd=True, withMean=True
    )

    # 3. Classifiers
    logger.info("Configuring Logistic Regression and Random Forest estimators...")
    lr = LogisticRegression(
        featuresCol="features", labelCol="target", maxIter=100, regParam=0.1
    )
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="target", numTrees=100, maxDepth=10, seed=seed
    )

    # Create pipelines
    lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
    rf_pipeline = Pipeline(stages=[assembler, scaler, rf])

    # Train Logistic Regression
    logger.info("Training distributed Logistic Regression model...")
    lr_model = lr_pipeline.fit(train_df)

    # Train Random Forest
    logger.info("Training distributed Random Forest model...")
    rf_model = rf_pipeline.fit(train_df)

    # Make predictions on validation set
    logger.info("Evaluating models on validation partition...")
    lr_predictions = lr_model.transform(val_df)
    rf_predictions = rf_model.transform(val_df)

    # Setup evaluators
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="accuracy"
    )
    prec_evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="weightedPrecision"
    )
    rec_evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="weightedRecall"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="f1"
    )
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )

    lr_accuracy = acc_evaluator.evaluate(lr_predictions)
    lr_prec = prec_evaluator.evaluate(lr_predictions)
    lr_rec = rec_evaluator.evaluate(lr_predictions)
    lr_f1 = f1_evaluator.evaluate(lr_predictions)
    lr_auc = auc_evaluator.evaluate(lr_predictions)
    lr_pr = pr_evaluator.evaluate(lr_predictions)

    logger.info(
        f"Logistic Regression Validation - Acc: {lr_accuracy:.4f}, Prec: {lr_prec:.4f}, Rec: {lr_rec:.4f}, F1: {lr_f1:.4f}, AUC-ROC: {lr_auc:.4f}, PR-AUC: {lr_pr:.4f}"
    )

    rf_accuracy = acc_evaluator.evaluate(rf_predictions)
    rf_prec = prec_evaluator.evaluate(rf_predictions)
    rf_rec = rec_evaluator.evaluate(rf_predictions)
    rf_f1 = f1_evaluator.evaluate(rf_predictions)
    rf_auc = auc_evaluator.evaluate(rf_predictions)
    rf_pr = pr_evaluator.evaluate(rf_predictions)

    logger.info(
        f"Random Forest Validation - Acc: {rf_accuracy:.4f}, Prec: {rf_prec:.4f}, Rec: {rf_rec:.4f}, F1: {rf_f1:.4f}, AUC-ROC: {rf_auc:.4f}, PR-AUC: {rf_pr:.4f}"
    )

    spark_metrics = {
        "logistic_regression": {
            "accuracy": lr_accuracy,
            "precision": lr_prec,
            "recall": lr_rec,
            "f1": lr_f1,
            "auc": lr_auc,
            "pr_auc": lr_pr,
        },
        "random_forest": {
            "accuracy": rf_accuracy,
            "precision": rf_prec,
            "recall": rf_rec,
            "f1": rf_f1,
            "auc": rf_auc,
            "pr_auc": rf_pr,
        },
    }

    provenance = get_provenance_metadata(seed)

    return SparkModelArtifacts(
        logistic_regression=lr_model,
        random_forest=rf_model,
        feature_names=feature_cols,
        metrics=spark_metrics,
        provenance=provenance,
    )


def save_spark_models(trained_artifacts: SparkModelArtifacts, dest_dir: Path) -> None:
    """Saves the trained Spark pipelines to disk."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    lr_path = dest_dir / "spark_logistic_regression"
    rf_path = dest_dir / "spark_random_forest"

    logger.info(f"Saving Logistic Regression Spark Pipeline to {lr_path}...")
    trained_artifacts.logistic_regression.write().overwrite().save(
        str(lr_path).replace("\\", "/")
    )

    logger.info(f"Saving Random Forest Spark Pipeline to {rf_path}...")
    trained_artifacts.random_forest.write().overwrite().save(
        str(rf_path).replace("\\", "/")
    )

    if trained_artifacts.metrics:
        from src.models.evaluation import save_metrics_json

        out_metrics = dict(trained_artifacts.metrics)
        if trained_artifacts.provenance:
            out_metrics["provenance"] = trained_artifacts.provenance

        save_metrics_json(out_metrics, dest_dir / "spark_metrics.json")

    logger.info("Spark MLlib models saved successfully.")
