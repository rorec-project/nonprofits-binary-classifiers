#!/usr/bin/env bash
# utils/run.sh — batch/segment runner for the binary-classifier pipeline.
#
# Runs on a UCloud job (Batch parameter or `bash utils/run.sh ...` over SSH).
# Stage-range aware because the human gates G1/G2/G3/G4 segment the pipeline:
#   run.sh "06,07,08"            # GPU segment (train → evaluate → infer)
#   run.sh "04,05,09"            # CPU segment
#   run.sh "03" --annotate-limit 20   # smoke-test stage 03 (extra args pass through)
#   run.sh "10"                  # standalone visualization (CPU)
#
# Node map (put a heavy GPU job only where it earns it):
#   CPU / local : 01, 02 (OpenAI arm), 04, 05, 09, 10, 11
#   GPU         : 03 (only the open-weight vLLM arm), 06, 07, 08
#
# Stages 01–09 go through scripts/run_pipeline.py (which enforces the gates).
# Stages 10 and 11 are NOT in that orchestrator's registry; they are standalone
# scripts, so this runner routes them directly.

set -euo pipefail

PROJECT_DRIVE="CHANGE_ME"
ENV_FILE="/work/${PROJECT_DRIVE}/.env"
[ -f "${ENV_FILE}" ] && set -a && . "${ENV_FILE}" && set +a

CONFIG="config/religious_missions.yaml"

REPO_DIR="/work/${PROJECT_DRIVE}"
ENV_SH="/work/${PROJECT_DRIVE}/env.sh"

if [ ! -f "${ENV_SH}" ]; then
  echo "ERROR: ${ENV_SH} not found — run 'bash utils/init.sh' first." >&2
  exit 1
fi
# shellcheck source=/dev/null
. "${ENV_SH}"

cd "${REPO_DIR}"
git pull --ff-only || echo "WARNING: 'git pull --ff-only' failed (network, auth, or local changes). Check stderr above." >&2

STAGES="${1:?usage: run.sh <stages> [extra run_pipeline.py args...]}"
shift || true

# Split the comma list: 10/11 → standalone scripts, everything else → orchestrator.
pipeline_stages=""
run_10=0
run_11=0
for s in ${STAGES//,/ }; do
  case "${s}" in
    10) run_10=1 ;;
    11) run_11=1 ;;
    *) pipeline_stages="${pipeline_stages:+${pipeline_stages},}${s}" ;;
  esac
done

echo "Pipeline start: $(date)"

if [ -n "${pipeline_stages}" ]; then
  uv run python scripts/run_pipeline.py --stages "${pipeline_stages}" --config "${CONFIG}" "$@"
fi
[ "${run_10}" -eq 1 ] && uv run python scripts/10_visualize.py --config "${CONFIG}"
[ "${run_11}" -eq 1 ] && uv run python scripts/11_aggregation_compare.py --config "${CONFIG}"

echo "Pipeline end:   $(date)"
