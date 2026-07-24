"""
Domain-specific exception hierarchy for the Binance K-Lines analytics pipeline.
"""


class BinanceAnalyticsError(Exception):
    """Base exception class for all errors in the Binance K-Lines analytics pipeline."""

    pass


class DataValidationError(BinanceAnalyticsError):
    """Raised when dataset schemas, shapes, or sequence lengths fail validation checks."""

    pass


class ModelTrainingError(BinanceAnalyticsError):
    """Raised when model training or hyperparameter sweeps fail or produce invalid outputs."""

    pass


class SparkPipelineError(BinanceAnalyticsError):
    """Raised when PySpark session creation, job execution, or S3 IO encounters errors."""

    pass


class ConfigurationError(BinanceAnalyticsError):
    """Raised when environment configuration or required paths are missing or invalid."""

    pass


class AWSError(BinanceAnalyticsError):
    """Base exception class for AWS client operations and infrastructure failures."""

    pass


class CrawlerTimeoutError(AWSError):
    """Raised when Glue Crawler execution exceeds the configured timeout limit."""

    pass
