"""
Spark session manager client.

Configures and initializes a SparkSession for local or AWS EMR execution,
setting appropriate options for reading/writing S3 data via HDFS/S3A connector.
"""

import logging
import os

# JVM options required on Java 11+ to allow Spark to access jdk.internal packages
java_options = (
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
os.environ["JDK_JAVA_OPTIONS"] = java_options
os.environ["JAVA_TOOL_OPTIONS"] = java_options


import src.config as config
from pyspark.sql import SparkSession


logger = logging.getLogger(__name__)


def get_spark_session() -> SparkSession:
    """Initializes and returns a unified SparkSession based on environment config."""
    logger.info("Initializing Spark session...")

    # JVM options required on Java 11+ to allow Spark to access jdk.internal packages
    java_options = (
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

    builder = (
        SparkSession.builder.appName("BinanceKLinesAnalytics")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", java_options)
        .config("spark.executor.extraJavaOptions", java_options)
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
    if config.AWS_ACCESS_KEY_ID and config.AWS_ACCESS_KEY_ID not in (
        "your_access_key_id_if_any",
        "",
    ):
        logger.info("Configuring Spark S3A credentials...")
        builder = (
            builder.config("spark.jars.packages", config.SPARK_JARS_PACKAGES)
            .config(
                "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
            )
            .config("spark.hadoop.fs.s3a.access.key", config.AWS_ACCESS_KEY_ID)
            .config("spark.hadoop.fs.s3a.secret.key", config.AWS_SECRET_ACCESS_KEY)
        )

        if config.AWS_SESSION_TOKEN and config.AWS_SESSION_TOKEN not in (
            "your_session_token_if_any",
            "",
        ):
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

    spark = builder.getOrCreate()

    # Adjust log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Spark session initialized successfully.")
    return spark
