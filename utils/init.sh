#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/work/BINARY-CLASSIFIER-MISSIONS"
export UV_CACHE_DIR="/work/.uv-cache"
export UV_PYTHON_INSTALL_DIR="/work/.uv-python"

# Install uv if it is not already on PATH.
command -v uv >/dev/null || {
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
}

cd "$REPO_DIR"

# Self-manage Python 3.13 (system Python is 3.12).
uv python install 3.13 || true

# Sync dependencies and create the virtual environment.
uv sync

# Optional: pre-pull open-weight annotator weights into /work/models for
# fast job start. Uncomment and set the desired model identifier.
# MODEL_ID="Qwen/Qwen3-235B-A22B-Instruct-2507"
# uv run huggingface-cli download "${MODEL_ID}" \
#     --local-dir "/work/models/${MODEL_ID}"
