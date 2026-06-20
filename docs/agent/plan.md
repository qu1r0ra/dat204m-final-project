# Project Implementation Blueprint (Tentative & Subject to Change)

## Local Repo & Git Collaboration Plan

- **Platform:** Use a standard shared **GitHub repository** instead of AWS-native code hosting tools.
- **Development Model:** Teammates write, test, and debug all data cleaning and ML pipeline code **locally** on their own machines (VS Code, Jupyter, etc.).
- **Risk Mitigation:** Local coding eliminates cloud compute runtime costs during the initial messy debugging phase and shields your team from needing immediate AWS console access.
- **Git Workflow:** Maintain a simple `main`/`feature-branch` model. Teammates submit Pull Requests for your review. Once verified and merged into `main`, you pull the production code straight into the AWS environment using a standard `git pull`.

## AWS Infrastructure Plan

To keep billing completely independent and isolated across your individual AWS accounts, we are implementing a **Cross-Account Hub-and-Spoke Architecture**:

- **The Central Hub (Your Account):**
  - **Amazon S3:** Hosts the static core data split into two folders:
    - `/raw/`: The master uncompressed dataset (targeting 50+ GB of 1-Second Binance Klines).
    - `/sample/`: A downsampled, highly compressed 1–2 GB Parquet file (e.g., 1-minute or 5-minute aggregates for just a few major pairs like BTC and ETH).
  - **AWS Glue & Catalog:** A single Glue Crawler infers the schema and publishes the metadata table so it can be exposed across accounts.
  - **Costs:** Your account only bears the minimal flat storage cost for S3 (approx. \$1.15/month for 50 GB).
- **The Spokes (Teammate Accounts):**
  - **Cross-Account Access:** You apply a resource-based S3 bucket policy and Glue Catalog permissions allowing your teammates' specific AWS Account IDs to read data.
  - **Transient Compute:** Teammates use their own accounts to query data via **Amazon Athena** or spin up **Amazon SageMaker** instances to prototype models.
  - **Billing:** Independent. Whoever initiates the compute instance pays for the instance hours. To avoid cross-region data egress charges, all members must create their accounts and resources in the exact same AWS Region (e.g., `us-east-1`).
- **The Scale-Up Execution:** Hardcode identical S3 paths in your code configuration. Teammates write code pointing to the `/sample/` directory. When their local code is fully verified and ready, you will pull their final verified code into your account's SageMaker instance, switch the path string to the full `/raw/` directory, and run the 50+ GB processing scale-up.

_Note: This architecture and workflow are tentative and subject to change depending on what may best suit our needs as the project progresses._
