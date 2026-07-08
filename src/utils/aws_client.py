"""
AWS S3, CloudFormation, and Glue Catalog orchestration client.

Provides functions to deploy the Hub CloudFormation stack, upload processed sample Parquet
files, and execute Glue Crawlers for cataloging.
"""

import argparse
import logging
import sys
import time
import boto3
import botocore.exceptions

import src.config as config

# Configure logging
logger = logging.getLogger(__name__)


class AWSError(Exception):
    """Base exception for AWS client operations."""

    pass


class CrawlerTimeoutError(AWSError):
    """Exception raised when Glue Crawler times out."""

    pass


def get_boto3_session() -> boto3.Session:
    """Initializes and returns a boto3 Session using configurations from config.py."""
    session_kwargs = {}
    if config.AWS_ACCESS_KEY_ID and config.AWS_ACCESS_KEY_ID not in (
        "your_access_key_id_if_any",
        "",
    ):
        session_kwargs["aws_access_key_id"] = config.AWS_ACCESS_KEY_ID
    if config.AWS_SECRET_ACCESS_KEY and config.AWS_SECRET_ACCESS_KEY not in (
        "your_secret_access_key_if_any",
        "",
    ):
        session_kwargs["aws_secret_access_key"] = config.AWS_SECRET_ACCESS_KEY
    if config.AWS_SESSION_TOKEN and config.AWS_SESSION_TOKEN not in (
        "your_session_token_if_any",
        "",
    ):
        session_kwargs["aws_session_token"] = config.AWS_SESSION_TOKEN
    if config.AWS_REGION:
        session_kwargs["region_name"] = config.AWS_REGION

    return boto3.Session(**session_kwargs)


def deploy_stack(teammate_ids: list[str] = None) -> None:
    """Deploys or updates the Hub CloudFormation stack containing buckets and databases."""
    session = get_boto3_session()
    cf = session.client("cloudformation")
    stack_name = "dat204m-binance-hub-stack"
    template_path = config.PROJECT_ROOT / "aws" / "hub_infrastructure.yaml"

    if not template_path.exists():
        raise AWSError(f"CloudFormation template not found at: {template_path}")

    with open(template_path, "r") as f:
        template_body = f.read()

    teammates_str = ",".join(teammate_ids) if teammate_ids else ""
    parameters = [
        {
            "ParameterKey": "CentralBucketName",
            "ParameterValue": config.AWS_S3_BUCKET_NAME,
        },
        {"ParameterKey": "TeammateAccountIds", "ParameterValue": teammates_str},
    ]

    logger.info(f"Checking status of stack '{stack_name}'...")
    exists = False
    try:
        cf.describe_stacks(StackName=stack_name)
        exists = True
    except botocore.exceptions.ClientError as e:
        if "does not exist" not in str(e):
            logger.error(f"Failed to describe CloudFormation stacks: {e}")
            raise e

    try:
        if exists:
            logger.info("Stack already exists. Initiating update...")
            cf.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            logger.info("Waiting for stack update to complete...")
            waiter = cf.get_waiter("stack_update_complete")
            waiter.wait(StackName=stack_name)
            logger.info("Stack update completed successfully!")
        else:
            logger.info("Stack does not exist. Initiating creation...")
            cf.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            logger.info("Waiting for stack creation to complete...")
            waiter = cf.get_waiter("stack_create_complete")
            waiter.wait(StackName=stack_name)
            logger.info("Stack creation completed successfully!")
    except botocore.exceptions.ClientError as e:
        if "No updates are to be performed" in str(e):
            logger.info("No stack modifications or updates needed.")
        else:
            raise AWSError(f"CloudFormation deployment failed: {e}") from e


def upload_sample_parquet() -> None:
    """Uploads the local sample Parquet dataset to the Hub S3 bucket under /sample/."""
    session = get_boto3_session()
    s3 = session.client("s3")
    local_path = config.SAMPLE_PARQUET_PATH

    if not local_path.exists():
        raise AWSError(
            f"Local sample Parquet file not found at: {local_path}. Please run sample_generator first."
        )

    s3_key = f"{config.AWS_S3_SAMPLE_PREFIX}{local_path.name}"
    logger.info(
        f"Uploading local file '{local_path.name}' to s3://{config.AWS_S3_BUCKET_NAME}/{s3_key}..."
    )

    try:
        # Upload the file enforcing SSE-S3 encryption
        s3.upload_file(
            Filename=str(local_path),
            Bucket=config.AWS_S3_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        logger.info(
            f"Successfully uploaded dataset to: s3://{config.AWS_S3_BUCKET_NAME}/{s3_key}"
        )
    except Exception as e:
        raise AWSError(f"Failed to upload sample Parquet to S3: {e}") from e


def run_glue_crawler(
    timeout_seconds: int = 600, poll_interval_seconds: int = 15
) -> None:
    """Triggers the Glue Crawler and monitors its execution state until completion."""
    session = get_boto3_session()
    glue = session.client("glue")
    crawler_name = "binance_sample_crawler"

    logger.info(f"Triggering Glue Crawler '{crawler_name}'...")
    try:
        glue.start_crawler(Name=crawler_name)
        logger.info("Crawler run started.")
    except glue.exceptions.CrawlerRunningException:
        logger.warning("Crawler is already executing.")
    except Exception as e:
        raise AWSError(f"Failed to start crawler: {e}") from e

    logger.info("Waiting for Glue Crawler execution to complete...")
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_seconds:
            raise CrawlerTimeoutError(
                f"Glue Crawler execution timed out after {timeout_seconds} seconds."
            )
        try:
            status_res = glue.get_crawler(Name=crawler_name)
            state = status_res["Crawler"]["State"]
            logger.info(f"Crawler state: {state}")
            if state == "READY":
                last_run = status_res["Crawler"].get("LastCrawl", {})
                run_status = last_run.get("Status")
                logger.info(
                    f"Crawler execution completed. Final Crawl Status: {run_status}"
                )
                if run_status == "FAILED":
                    raise AWSError("Glue Crawler execution failed.")
                break
        except Exception as e:
            if isinstance(e, AWSError):
                raise e
            raise AWSError(f"Failed to check crawler state: {e}") from e
        time.sleep(poll_interval_seconds)


def main() -> None:
    """Main CLI entry point for the AWS Client utility."""
    parser = argparse.ArgumentParser(
        description="DAT204M AWS Hub deployment and data upload tool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: deploy-stack
    deploy_parser = subparsers.add_parser(
        "deploy-stack", help="Deploy or update the CloudFormation infrastructure."
    )
    deploy_parser.add_argument(
        "--teammates",
        nargs="*",
        help="AWS Account IDs of teammates to grant read access.",
    )

    # Subcommand: upload-sample
    subparsers.add_parser("upload-sample", help="Upload the sample Parquet file to S3.")

    # Subcommand: run-crawler
    subparsers.add_parser(
        "run-crawler", help="Start the Glue Crawler to catalog the S3 data."
    )

    # Subcommand: deploy-all
    all_parser = subparsers.add_parser(
        "deploy-all",
        help="Run deploy-stack, upload-sample, and run-crawler sequentially.",
    )
    all_parser.add_argument(
        "--teammates",
        nargs="*",
        help="AWS Account IDs of teammates to grant read access.",
    )

    args = parser.parse_args()

    # Configure root logging inside the main guard to prevent side effects on import
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        if args.command == "deploy-stack":
            deploy_stack(args.teammates)
        elif args.command == "upload-sample":
            upload_sample_parquet()
        elif args.command == "run-crawler":
            run_glue_crawler()
        elif args.command == "deploy-all":
            deploy_stack(args.teammates)
            upload_sample_parquet()
            run_glue_crawler()
    except AWSError as e:
        logger.error(f"AWS operations failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
