#!/usr/bin/env bash
# utils/serve-llm.sh — start the coding LLM stack on a UCloud interactive job.
#
# Starts vLLM with LLM_CODING_MODEL (defined in config.sh) and prints the
# endpoint URLs for both job-local and laptop access.
#
# Laptop access uses UCloud's native Public IP feature (Resources → IP addresses
# in the UCloud web UI). Allocate one static IP, open port LLM_CODING_PORT TCP,
# set UCLOUD_PUBLIC_IP in config.sh, and attach it at job submission — the IP
# is static across jobs, so your laptop config never needs updating.
#
# Usage:
#   bash utils/serve-llm.sh           # start (idempotent)
#   bash utils/serve-llm.sh --stop    # kill vLLM
#   bash utils/serve-llm.sh --status  # print running state + endpoint URLs
#
# Prerequisites:
#   - init.sh has run (drives mounted, uv env synced, env.sh written)
#   - devenv.sh has run (Opencode installed, ~/.config/opencode/ written)
#   - uv sync --extra serve has been run (vllm in .venv)
#   - At job submission: Public IP resource attached with LLM_CODING_PORT open
#
# Logs: $PROJECT/outputs/interim/vllm-coding.log
#
set -euo pipefail

# ── Source config + secrets ───────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/config.sh"

PROJECT="/work/${PROJECT_DRIVE}"
ENV_SH="${PROJECT}/env.sh"
ENV_FILE="${PROJECT}/.env"

[ -f "${ENV_SH}" ] && source "${ENV_SH}"
[ -f "${ENV_FILE}" ] && set -a && . "${ENV_FILE}" && set +a

VLLM_LOG="${PROJECT}/outputs/interim/vllm-coding.log"

# ── Helpers ───────────────────────────────────────────────────────────────────

vllm_running() {
  pgrep -f "vllm serve ${LLM_CODING_MODEL}" >/dev/null 2>&1
}

vllm_ready() {
  curl -sf "http://localhost:${LLM_CODING_PORT}/health" >/dev/null 2>&1
}

print_endpoints() {
  local model_short
  model_short="$(basename "${LLM_CODING_MODEL}")"

  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  Model:  ${model_short}  (port :${LLM_CODING_PORT})"
  echo ""
  echo "  Within this job (Opencode in SSH session):"
  echo "    http://localhost:${LLM_CODING_PORT}/v1"
  echo ""
  if [ -n "${UCLOUD_PUBLIC_IP:-}" ]; then
    echo "  From your laptop (UCloud Public IP — static):"
    echo "    http://${UCLOUD_PUBLIC_IP}:${LLM_CODING_PORT}/v1"
    echo ""
    echo "  Set once in your laptop shell init (~/.bashrc / ~/.zshrc):"
    echo "    export UCLOUD_LLM_BASE_URL=\"http://${UCLOUD_PUBLIC_IP}:${LLM_CODING_PORT}/v1\""
    echo "    export UCLOUD_LLM_KEY=\"${LLM_API_KEY}\""
  else
    echo "  Laptop access: set UCLOUD_PUBLIC_IP in config.sh."
    echo "  Allocate a static IP in UCloud → Resources → IP addresses,"
    echo "  open port ${LLM_CODING_PORT} TCP, attach at job submission."
  fi
  echo "═══════════════════════════════════════════════════════════════"
  echo ""
}

# ── --stop ────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--stop" ]]; then
  echo "--- Stopping vLLM coding server ---"
  if vllm_running; then
    pkill -f "vllm serve ${LLM_CODING_MODEL}" && echo "  Stopped."
  else
    echo "  Not running."
  fi
  exit 0
fi

# ── --status ──────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--status" ]]; then
  echo "vLLM:  $(vllm_running && echo "running (PID $(pgrep -f "vllm serve ${LLM_CODING_MODEL}"))" || echo "stopped")"
  echo "ready: $(vllm_ready && echo "yes" || echo "no")"
  print_endpoints
  exit 0
fi

# ── Validate drives are mounted ───────────────────────────────────────────────

if [ ! -d "${PROJECT}" ]; then
  echo "ERROR: project drive not found at ${PROJECT}." >&2
  echo "       Attach both drives and run init.sh first." >&2
  exit 1
fi
mkdir -p "$(dirname "${VLLM_LOG}")"

# ── Ensure vllm is in the env (uv sync --extra serve) ────────────────────────

if ! uv run python -c "import vllm" 2>/dev/null; then
  echo "--- vllm not in .venv; running: uv sync --extra serve ---"
  cd "${PROJECT}"
  uv sync --extra serve
fi

# ── Start vLLM ────────────────────────────────────────────────────────────────

if vllm_running; then
  echo "--- vLLM already running (PID $(pgrep -f "vllm serve ${LLM_CODING_MODEL}")) ---"
else
  echo "--- Starting vLLM: ${LLM_CODING_MODEL} ---"
  echo "    port=${LLM_CODING_PORT}  tp=${LLM_TENSOR_PARALLEL}  max-len=${LLM_MAX_MODEL_LEN}"

  cd "${PROJECT}"
  nohup uv run vllm serve "${LLM_CODING_MODEL}" \
    --host 0.0.0.0 \
    --port "${LLM_CODING_PORT}" \
    --tensor-parallel-size "${LLM_TENSOR_PARALLEL}" \
    --max-model-len "${LLM_MAX_MODEL_LEN}" \
    --dtype auto \
    --api-key "${LLM_API_KEY}" \
    &>>"${VLLM_LOG}" &
  echo "    PID $! — logs: ${VLLM_LOG}"

  echo -n "    Waiting for /health"
  ATTEMPTS=0
  until vllm_ready; do
    sleep 5
    ATTEMPTS=$((ATTEMPTS + 1))
    echo -n "."
    if [ "${ATTEMPTS}" -ge 72 ]; then
      echo ""
      echo "ERROR: vLLM did not become healthy after 6 min." >&2
      echo "       tail -f ${VLLM_LOG}" >&2
      exit 1
    fi
  done
  echo " ready (${ATTEMPTS}×5s)"
fi

print_endpoints
