"""Base model trainer abstraction and unified training result contracts.

Provides standardized TrainerResult container and BaseModelTrainer interface
across scikit-learn, PyTorch, and PySpark MLlib model pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.evaluation import save_metrics_json
from src.utils.seed import get_provenance_metadata


@dataclass
class TrainerResult:
    """Unified container holding model training outputs, metrics, and metadata."""

    model_name: str
    model: Any
    scaler: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=get_provenance_metadata)
    threshold: float = 0.5
    feature_names: list[str] = field(default_factory=list)
    hparams: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts metadata and metrics to serializable dictionary format."""
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "provenance": self.provenance,
            "hparams": self.hparams,
        }


class BaseModelTrainer(ABC):
    """Abstract base class for all machine learning model trainers."""

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> TrainerResult:
        """Trains the machine learning model and returns a standardized TrainerResult."""
        pass


def save_trainer_result(result: TrainerResult, output_dir: Path | str) -> None:
    """Serializes TrainerResult metrics and metadata to a target output directory."""
    dest_path = Path(output_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    json_path = dest_path / f"{result.model_name.lower().replace(' ', '_')}_result.json"
    save_metrics_json(result.to_dict(), json_path)
