"""
Shared test configuration and fixtures.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_ohlcv_df():
    """Provides a small synthetic OHLCV Polars DataFrame for testing."""
    np.random.seed(42)
    # Generate 100 rows of monotonic synthetic data
    base_time = datetime(2024, 1, 1)
    times = [base_time + timedelta(minutes=i) for i in range(100)]

    # Create somewhat realistic trending prices
    close_prices = 100.0 + np.cumsum(np.random.normal(0, 1, 100))

    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 100,
            "open_time": times,
            "open": close_prices - 0.5,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.random.uniform(10, 100, 100),
        }
    )
