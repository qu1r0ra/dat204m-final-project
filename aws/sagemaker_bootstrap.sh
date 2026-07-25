#!/bin/bash
# ==============================================================================
# SageMaker Lifecycle Configuration / Bootstrapping Script
#
# Automatically installs astral 'uv', synchronizes project python dependencies,
# and registers the custom environment kernel inside Jupyter.
# ==============================================================================

set -e

echo "=== AWS SageMaker Bootstrapping Init ==="

# Execute bootstrapping as ec2-user in the background to prevent SageMaker 5-minute timeout limit
sudo -u ec2-user -i bash -c '
set -e
export PATH="$HOME/.local/bin:$PATH"

# 1. Install Astral uv (if not installed)
if ! command -v uv &> /dev/null; then
    echo "Installing astral uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Navigate to repo if cloned and sync dependencies
REPO_DIR="$HOME/SageMaker/dat204m-final-project"
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    if [ -f "pyproject.toml" ]; then
        echo "Synchronizing workspace dependencies..."
        uv sync
        echo "Registering custom environment as Jupyter Kernel..."
        uv run python -m ipykernel install --user --name="dat204m-final-project" --display-name="Python (DAT204M)"
    fi
fi
' > /home/ec2-user/bootstrap.log 2>&1 &

echo "Bootstrapping process launched in background. Check /home/ec2-user/bootstrap.log for progress."
echo "=== SageMaker Bootstrapping Initiated Successfully ==="
