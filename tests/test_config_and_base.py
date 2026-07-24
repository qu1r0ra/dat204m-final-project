"""Unit tests for PipelineConfig, TrainerResult, and feature parity abstractions."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from src.config import PipelineConfig
from src.exceptions import DataValidationError
from src.features.indicators import validate_feature_parity
from src.models.base import TrainerResult, save_trainer_result
from src.utils.helpers import normalize_path_str


def test_pipeline_config_defaults():
    config = PipelineConfig.from_env()
    assert config.execution_mode in ["local_sample", "local_raw", "aws_hub"]
    assert config.target_symbol == "BTCUSDT"
    assert config.future_horizon == 15
    assert isinstance(config.feature_cols, tuple)
    assert len(config.feature_cols) == 16


def test_pipeline_config_testing_overrides():
    config = PipelineConfig.for_testing(target_symbol="ETHUSDT", future_horizon=30)
    assert config.target_symbol == "ETHUSDT"
    assert config.future_horizon == 30
    assert config.execution_mode == "local_sample"


def test_normalize_path_str():
    raw_path = Path("data") / "sample" / "test.parquet"
    normalized = normalize_path_str(raw_path)
    assert "\\" not in normalized
    assert normalized.endswith("data/sample/test.parquet")


def test_trainer_result_serialization(tmp_path):
    result = TrainerResult(
        model_name="Test Model",
        model={"type": "dummy"},
        metrics={"accuracy": 0.85, "f1": 0.82},
        threshold=0.55,
        feature_names=["f1", "f2"],
    )

    result_dict = result.to_dict()
    assert result_dict["model_name"] == "Test Model"
    assert result_dict["threshold"] == 0.55
    assert result_dict["metrics"]["accuracy"] == 0.85

    save_trainer_result(result, tmp_path)
    json_path = tmp_path / "test_model_result.json"
    assert json_path.exists()


def test_validate_feature_parity_success():
    polars_df = pl.DataFrame(
        {
            "close_to_sma_15": [0.01, 0.02, -0.01],
            "rsi_14": [50.0, 55.0, 45.0],
        }
    )

    pandas_df = pd.DataFrame(
        {
            "close_to_sma_15": [0.0100001, 0.0200001, -0.0100001],
            "rsi_14": [50.000001, 55.000001, 45.000001],
        }
    )

    assert validate_feature_parity(
        polars_df, pandas_df, feature_cols=["close_to_sma_15", "rsi_14"], tol=1e-4
    )


def test_validate_feature_parity_failure():
    polars_df = pl.DataFrame({"rsi_14": [50.0, 55.0]})
    pandas_df = pd.DataFrame({"rsi_14": [60.0, 65.0]})

    with pytest.raises(DataValidationError):
        validate_feature_parity(polars_df, pandas_df, feature_cols=["rsi_14"], tol=1e-4)
