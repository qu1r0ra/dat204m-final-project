"""
Reproducibility and seed management module.

Provides centralized random seed setting across Python random, NumPy, PyTorch,
and scikit-learn, along with provenance metadata generation for serializing
experiment conditions into model checkpoints.
"""

import logging
import os
import random
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Sets random seed across all libraries to ensure 100% reproducible execution.

    Args:
        seed: Integer random seed value.
        deterministic_cudnn: If True, forces CUDNN deterministic execution for PyTorch.
    """
    logger.info(f"Setting global random seed to {seed}")

    # 1. Python standard library
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch (if installed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("PyTorch CUDNN configured for deterministic execution.")
    except ImportError:
        pass


def get_provenance_metadata(seed: int = 42) -> dict[str, Any]:
    """Generates a provenance metadata dictionary tracking execution conditions.

    Args:
        seed: Random seed used for the experiment run.

    Returns:
        Dictionary containing library versions, python environment info, timestamp, and seed.
    """
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "random_seed": seed,
        "packages": {},
    }

    packages_to_check = [
        "torch",
        "polars",
        "pyspark",
        "sklearn",
        "numpy",
        "pandas",
        "duckdb",
    ]
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            metadata["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            metadata["packages"][pkg] = "not_installed"

    return metadata
