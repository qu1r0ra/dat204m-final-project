#!/bin/bash
# ==============================================================================
# SageMaker Lifecycle Configuration / Bootstrapping Script
#
# Automatically installs astral 'uv', synchronizes project python dependencies,
# and registers the custom environment kernel inside Jupyter.
# ==============================================================================

set -e

echo "=== AWS SageMaker Bootstrapping Init ==="

# 1. Install Astral uv (if not installed)
if ! command -v uv &> /dev/null; then
    echo "Installing astral uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to local path
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed successfully."
else
    echo "uv is already installed."
fi

# Make sure PATH update is preserved for sub-shells
export PATH="$HOME/.local/bin:$PATH"

# 2. Sync project dependencies from pyproject.toml
if [ -f "pyproject.toml" ]; then
    echo "Synchronizing workspace dependencies..."
    uv sync
else
    echo "Warning: pyproject.toml not found in the current directory. Skipping uv sync."
fi

# 3. Register Jupyter kernel
echo "Registering custom environment as Jupyter Kernel..."
uv run python -m ipykernel install --user --name="dat204m-final-project" --display-name="Python (DAT204M)"

echo "=== SageMaker Bootstrapping Completed Successfully ==="
