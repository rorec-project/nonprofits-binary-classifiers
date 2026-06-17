#!/usr/bin/env bash
# utils/init.sh — lean RUNTIME init for the binary-classifier pipeline on UCloud.
#
# App-agnostic: works as the Initialization Bash script for a Terminal /
# PyTorch / JupyterLab job, and as a manual `bash utils/init.sh`. It wires the
# ephemeral job at the persistent storage spine (two UCloud Drives), exports
# runtime env, creates the data symlinks, and syncs the Python env. It does NOT
# install editors or agent harnesses — that is the optional `utils/devenv.sh`
# overlay (interactive SSH only), kept decoupled so GPU/batch jobs stay lean.
#
# Bootstrap order:
#   First-ever job : create .env on the project drive (pre-job) → start Terminal
#                    with this script as Initialization → it sources .env, seeds
#                    git, detects no checkout, exits → SSH in → clone the repo
#                    → `bash utils/init.sh`.
#   Every later job: attach this script as Initialization. It detects the
#                    existing checkout and only pulls + syncs + relinks.
#
set -euo pipefail

# Fallback defaults for the critical path variables — used when this script
# runs as the UCloud Initialization script (before the repo checkout is
# accessible). Duplicated in utils/config.sh for the other scripts; edit
# config.sh when changing drive names, then update this block to match.
DATA_DRIVE="CHANGE_ME_DATA_DRIVE"
PROJECT_DRIVE="CHANGE_ME_PROJECT_DRIVE"
GIT_NAME="YOUR_NAME"
GIT_EMAIL="YOUR_EMAIL"

# Pre-pull the open-weight annotator weights into the shared HF cache. Leave 0
# on every job EXCEPT the vLLM annotation job (stage 03 open-weight arm); the
# model id is read from config/religious_missions.yaml, never hardcoded here.
PREPULL_MODEL=0

# ─── Wait for the /work mount ─────────────────────────────────────────────────

echo "--- Waiting for /work mount ---"
TIMEOUT=60
ELAPSED=0
until [ -d "/work" ] && [ -n "$(ls -A /work 2>/dev/null)" ]; do
  sleep 2
  ELAPSED=$((ELAPSED + 2))
  if [ "${ELAPSED}" -ge "${TIMEOUT}" ]; then
    echo "ERROR: /work not mounted after ${TIMEOUT}s." >&2
    exit 1
  fi
done
echo "--- /work ready after ${ELAPSED}s ---"

# Now the mount is up — source config.sh from the project drive to override
# the fallback defaults above. For the Initialization script use case, this
# absolute-path source works because the drive name is already known; for
# manual `bash utils/init.sh` from the repo root, it resolves to the same
# location as "$(dirname "$0")/config.sh".
PROJECT="/work/${PROJECT_DRIVE}"
CONFIG_SH="${PROJECT}/utils/config.sh"
if [ -f "${CONFIG_SH}" ]; then
  . "${CONFIG_SH}"
else
  echo "WARNING: ${CONFIG_SH} not found — using fallback defaults." >&2
fi

# Derive all paths from (possibly overridden) config values.
DATA="/work/${DATA_DRIVE}"
PROJECT="/work/${PROJECT_DRIVE}"
REPO_DIR="${PROJECT}"
ENV_FILE="${PROJECT}/.env"
ENV_SH="${PROJECT}/env.sh"

# ─── Assert both Drives are mounted (fatal if absent) ─────────────────────────

for drive in "${DATA}" "${PROJECT}"; do
  if [ ! -d "${drive}" ]; then
    echo "ERROR: expected Drive not found at ${drive}" >&2
    echo "       Attach it to the job and set DATA_DRIVE / PROJECT_DRIVE to the" >&2
    echo "       names you actually see under /work:" >&2
    ls -la /work/ >&2 || true
    exit 1
  fi
done
echo "--- Drives OK: ${DATA} , ${PROJECT} ---"

# ─── Install uv if missing (UCloud usually ships it pre-installed) ────────────

if ! command -v uv >/dev/null 2>&1; then
  echo "--- Installing uv ---"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

# ─── Export runtime env for THIS shell ────────────────────────────────────────
# HF_HOME covers both the transformer encoders and the vLLM open-weight download
# so encoders and the Gemma annotator share one reusable cache across projects.
# The uv cache lives on the data drive but .venv lives on the project drive
# (separate mounts), so hardlinks fail — UV_LINK_MODE=copy makes the fallback
# explicit and quiet (the cache still avoids re-downloads).
export HF_HOME="${DATA}/hf-cache"
export UV_CACHE_DIR="${DATA}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${DATA}/uv-python"
export UV_LINK_MODE=copy

# Secrets live ONLY on the project drive (never the shared corpus drive). Source
# is guarded so a job missing .env warns instead of aborting under `set -e`.
if [ -f "${ENV_FILE}" ]; then
  echo "--- Sourcing secrets from ${ENV_FILE} ---"
  set -a && . "${ENV_FILE}" && set +a
else
  echo "WARNING: ${ENV_FILE} not found — OPENAI/HF/GITHUB tokens will be unset." >&2
fi

# ─── Write the lean runtime env.sh and auto-source it on every login ──────────
# Generated with the resolved absolute /work paths baked in; $HOME/$PATH stay
# literal so they evaluate at login time. Kept lean — the dev/agent overlay
# writes its OWN sourced file so batch jobs never pick up the harness env.
echo "--- Writing ${ENV_SH} ---"
cat >"${ENV_SH}" <<EOF
# env.sh — runtime environment for the binary-classifier pipeline on UCloud.
# Auto-generated by utils/init.sh. Sourced by ~/.bashrc, ~/.profile, utils/run.sh.
export HF_HOME="${DATA}/hf-cache"
export UV_CACHE_DIR="${DATA}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${DATA}/uv-python"
export UV_LINK_MODE=copy
export PATH="\$HOME/.local/bin:\$PATH"
[ -f "${ENV_FILE}" ] && set -a && . "${ENV_FILE}" && set +a
EOF

SOURCE_LINE="[ -f '${ENV_SH}' ] && source '${ENV_SH}'"
for rc in "${HOME}/.bashrc" "${HOME}/.profile" "${HOME}/.bash_profile"; do
  grep -qF "${ENV_SH}" "${rc}" 2>/dev/null || echo "${SOURCE_LINE}" >>"${rc}"
done

# ─── Git identity + HTTPS credential helper (no token in URLs or history) ─────

git config --global user.name "${GIT_NAME}"
git config --global user.email "${GIT_EMAIL}"
git config --global credential.helper store
if [ -n "${GITHUB_TOKEN:-}" ]; then
  printf 'https://%s:x-oauth-basic@github.com\n' "${GITHUB_TOKEN}" >"${HOME}/.git-credentials"
  chmod 600 "${HOME}/.git-credentials"
fi

# ─── First-job bootstrap: repo not cloned yet ─────────────────────────────────

if [ ! -d "${REPO_DIR}/.git" ]; then
  echo ""
  echo "=== Bootstrap: repository not found at ${REPO_DIR} ==="
  echo "Env, env.sh, and git credentials are now set up. Clone the repo, then"
  echo "re-run this script (it will pull + sync + relink from here on):"
  echo ""
  echo "  cd \"${REPO_DIR}\""
  echo "  git init"
  echo "  git remote add origin https://github.com/rorec-project/nonprofits-binary-classifiers.git"
  echo "  git fetch origin master"
  echo "  git checkout master"
  echo "  bash utils/init.sh"
  echo ""
  exit 0
fi

# ─── cwd is load-bearing: PathRegistry anchors at the current directory ───────

cd "${REPO_DIR}"
git pull --ff-only || echo "WARNING: 'git pull --ff-only' failed (network, auth, or local changes). Check stderr above." >&2

# ─── Pre-create drive-side target directories ─────────────────────────────────
# The symlinks below point here; PathRegistry.ensure_dirs() does
# mkdir(exist_ok=True) on these paths, which RE-RAISES on a symlink whose target
# is missing. Creating the top-level targets first makes the links valid; the
# nested dirs (manifests, shards, runs, checkpoints, embeddings) are created
# through the now-valid links by ensure_dirs().
mkdir -p \
  "${DATA}/raw" "${DATA}/hf-cache" "${DATA}/uv-cache" "${DATA}/uv-python" \
  "${PROJECT}/outputs/interim" \
  "${PROJECT}/outputs/models" \
  "${PROJECT}/outputs/processed/evaluation" \
  "${PROJECT}/outputs/processed/predictions" \
  "${PROJECT}/outputs/processed/prevalence" \
  "${PROJECT}/outputs/processed/figures"

# ─── Create the data symlinks (idempotent) ────────────────────────────────────
# `ln -sfn` safely replaces an existing SYMLINK, but nests inside a real dir, so
# clear any wrong-type path first. data/processed itself stays a REAL directory
# (it holds the git-tracked gold/ subtree); only its leaf entries are symlinked.
link() {
  local target="$1" linkpath="$2"
  if [ -e "${linkpath}" ] && [ ! -L "${linkpath}" ]; then
    rm -rf "${linkpath}"
  fi
  ln -sfn "${target}" "${linkpath}"
}

mkdir -p data data/processed

link "${DATA}/raw" data/raw
link "${PROJECT}/outputs/interim" data/interim
link "${PROJECT}/outputs/models" data/models
link "${PROJECT}/outputs/processed/silver_labels.csv" data/processed/silver_labels.csv
link "${PROJECT}/outputs/processed/evaluation" data/processed/evaluation
link "${PROJECT}/outputs/processed/predictions" data/processed/predictions
link "${PROJECT}/outputs/processed/prevalence" data/processed/prevalence
link "${PROJECT}/outputs/processed/figures" data/processed/figures

echo "--- Symlinks ---"
ls -l data data/processed

# ─── Python environment ───────────────────────────────────────────────────────
# System Python is 3.12; the project requires 3.13. `uv sync` includes the `dev`
# group by default (ruff + ty), so verification step 4 works out of the box.
# Add `--extra serve` ONLY on the vLLM annotation job.
if ! uv python install 3.13; then
  echo "WARNING: 'uv python install 3.13' failed. 'uv sync' may fail if 3.13 is unavailable." >&2
fi
uv sync

# ─── Optional: pre-pull the open-weight annotator into the shared HF cache ────
# Single switch point: the model id is derived from the YAML's vLLM candidate,
# never hardcoded. Skips cleanly if the vLLM arm is commented out.
if [ "${PREPULL_MODEL}" -eq 1 ]; then
  MODEL_ID="$(uv run python -c "from binary_classifier.config import load_config; print(next((c.id for c in load_config('config/religious_missions.yaml').model_slate.bakeoff_candidates if c.provider=='vllm'), ''))")"
  if [ -n "${MODEL_ID}" ]; then
    echo "--- Pre-pulling ${MODEL_ID} into ${HF_HOME} ---"
    uv run huggingface-cli download "${MODEL_ID}"
  else
    echo "--- No vLLM candidate in config; skipping model pre-pull. ---"
  fi
fi

echo "=== init.sh complete $(date) ==="
echo "Open a new shell (or 'source ${ENV_SH}') and the runtime env is loaded."
