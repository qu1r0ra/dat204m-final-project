"""
Spark session manager client.

Configures and initializes a SparkSession for local or AWS EMR execution,
setting appropriate options for reading/writing S3 data via HDFS/S3A connector.
"""

import logging
import os

# JVM options required on Java 11+ to allow Spark to access jdk.internal packages
JAVA_OPTIONS = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"
)
from pyspark.sql import SparkSession  # noqa: E402

import src.config as config  # noqa: E402

logger = logging.getLogger(__name__)


def setup_spark_env() -> None:
    """Configures required environment variables for Spark."""
    import sys

    os.environ["JDK_JAVA_OPTIONS"] = JAVA_OPTIONS
    os.environ["JAVA_TOOL_OPTIONS"] = JAVA_OPTIONS
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    config.configure_java_home()


def ensure_hadoop_home() -> None:
    """Provisions winutils.exe for Windows and configures HADOOP_HOME."""
    if os.name == "nt":
        import shutil
        from pathlib import Path

        hadoop_dir = (config.DATA_DIR / "hadoop").resolve()
        bin_dir = hadoop_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        winutils_exe = bin_dir / "winutils.exe"
        if not winutils_exe.exists():
            attrib_exe = (
                Path(os.environ.get("SYSTEMROOT", "C:\\Windows")) / "System32" / "attrib.exe"
            )
            if attrib_exe.exists():
                try:
                    shutil.copy(str(attrib_exe), str(winutils_exe))
                except OSError as e:
                    logger.warning(
                        f"Could not provision winutils.exe from {attrib_exe}: {e}. "
                        "Spark may fail with Hadoop-related errors on Windows."
                    )
        os.environ["HADOOP_HOME"] = str(hadoop_dir)


def get_spark_session() -> SparkSession:
    """Initializes and returns a unified SparkSession based on environment config."""
    logger.info("Initializing Spark session...")

    setup_spark_env()
    ensure_hadoop_home()

    builder = (
        SparkSession.builder.appName("BinanceKLinesAnalytics")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", JAVA_OPTIONS)
        .config("spark.executor.extraJavaOptions", JAVA_OPTIONS)
    )

    # Configure master local mode or defer to cluster resource manager
    if config.SPARK_EXECUTION_MODE == "local":
        logger.info("Configuring Spark for local standalone execution...")
        builder = (
            builder.master("local[*]")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
        )
    else:
        logger.info("Configuring Spark for cluster/EMR execution...")

    # Configure AWS S3A access if AWS credentials are provided
    if config.AWS_ACCESS_KEY_ID and not config.AWS_ACCESS_KEY_ID.startswith("your_"):
        logger.info("Configuring Spark S3A credentials...")
        builder = (
            builder.config("spark.jars.packages", config.SPARK_JARS_PACKAGES)
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.access.key", config.AWS_ACCESS_KEY_ID)
            .config("spark.hadoop.fs.s3a.secret.key", config.AWS_SECRET_ACCESS_KEY)
        )

        if config.AWS_SESSION_TOKEN and not config.AWS_SESSION_TOKEN.startswith("your_"):
            builder = builder.config(
                "spark.hadoop.fs.s3a.session.token", config.AWS_SESSION_TOKEN
            ).config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider",
            )
        else:
            builder = builder.config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            )

    try:
        spark = builder.getOrCreate()
    except Exception as e:
        logger.error(
            f"Failed to initialize Spark session. Verify your local Java runtime, "
            f"environment variables, and Java versions (11/17/21/26 support): {e}"
        )
        raise e

    # Adjust log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Spark session initialized successfully.")
    return spark
