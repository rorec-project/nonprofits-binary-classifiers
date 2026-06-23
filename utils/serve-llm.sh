#!/usr/bin/env bash
# utils/serve-llm.sh — serve vLLM model(s) on a UCloud interactive job.
#
# Two roles, each on its own port, normally share one GPU (a B200's 183 GB
# fits both at once with small/medium models):
#   annotate — the open-weight production annotation arm for stages 02/03. The
#              model id prefers the confirmed production_slate.json and falls
#              back to the first vLLM bake-off candidate before G2 exists, so the
#              served model matches the model the pipeline asks for.
#   coding   — interactive dev assistant (Opencode); LLM_CODING_MODEL from .env.
#              If set to deepseek-ai/DeepSeek-V4-Flash, this role needs a full
#              B200 to itself (~146 GiB weights alone) -- `both` mode will not
#              fit it alongside annotate on the same GPU; use separate GPUs
#              (CUDA_VISIBLE_DEVICES) or run coding solo. See
#              docs/UCLOUD_CODING_LLM.md for sizing math and tradeoffs.
#
# Usage:
#   bash utils/serve-llm.sh annotate          # serve the annotation model
#   bash utils/serve-llm.sh coding            # serve the coding model
#   bash utils/serve-llm.sh both              # serve both concurrently (split VRAM)
#   bash utils/serve-llm.sh --status [role]   # running state + endpoint URLs
#   bash utils/serve-llm.sh --stop   [role]   # stop one role (default: all)
#
# Laptop access uses UCloud's native Public IP feature (Resources → IP addresses
# in the UCloud web UI). Allocate one static IP, open the relevant port(s) TCP,
# set UCLOUD_PUBLIC_IP in .env, and attach it at job submission — the IP is
# static across jobs, so your laptop config never needs updating.
#
# Prerequisites:
#   - init.sh has run (drives mounted, env.sh written, CUDA module loaded)
#   - uv sync --extra serve (vllm in .venv) — auto-run here if missing
#   - On Blackwell (B200), env.sh loads CUDA/12.8.0 so FlashInfer's first-launch
#     JIT compile of the sm_100 sampling kernels can find nvcc.
#
# Logs: $PROJECT/outputs/interim/vllm-<role>.log
#
set -euo pipefail
shopt -s nullglob

ALL_ROLES="annotate coding"
USAGE="Usage: serve-llm.sh {annotate|coding|both} | --status [role] | --stop [role]"

# ── Config + secrets ──────────────────────────────────────────────────────────

load_env() {
  local required="${1:-1}"

  ENV_FILE=""
  for d in /work/*/; do
    if [ -f "${d}.env" ]; then
      ENV_FILE="${d}.env"
      PROJECT_DRIVE="$(basename "${d}")"
      echo "--- Found .env on /work/${PROJECT_DRIVE} ---"
      break
    fi
  done

  if [ -z "${ENV_FILE}" ]; then
    if [ "${required}" -eq 1 ]; then
      echo "ERROR: .env not found under /work/." >&2
      echo "       Create it on the project drive before starting the job." >&2
      exit 1
    fi
    return 1
  fi

  set -a && . "${ENV_FILE}" && set +a

  PROJECT="/work/${PROJECT_DRIVE}"
  ENV_SH="${PROJECT}/env.sh"

  # env.sh loads the CUDA module + GPU cache redirects — load-bearing for vLLM.
  # shellcheck disable=SC1090
  [ -f "${ENV_SH}" ] && source "${ENV_SH}"
}

# ── Role → (model, port, max-model-len) resolution ────────────────────────────

annotate_model_id() {
  # Stage 03's source of truth is the confirmed production slate. Before G2
  # exists, fall back to the first configured vLLM bake-off candidate so the same
  # launcher still works for stage-02 smoke tests.
  ( cd "${PROJECT}" && uv run python - <<'PY'
import sys
from pathlib import Path

from binary_classifier.config import load_config, load_slate
from binary_classifier.paths import PathRegistry

config_path = Path("config/religious_missions.yaml")
cfg = load_config(config_path)
registry = PathRegistry(config_path)
slate_path = registry.production_slate
if slate_path.exists():
    slate = load_slate(slate_path)
    if slate.confirmed:
        models = [model.id for model in slate.models if model.provider == "vllm"]
        if len(models) == 1:
            print(models[0])
            raise SystemExit(0)
        if len(models) > 1:
            print(
                "ERROR: production_slate.json lists multiple vLLM production "
                "models; serve them manually on separate ports.",
                file=sys.stderr,
            )
            raise SystemExit(1)

print(
    next(
        (
            candidate.id
            for candidate in cfg.model_slate.bakeoff_candidates
            if candidate.provider == "vllm"
        ),
        "",
    )
)
PY
  )
}

role_model() {
  case "$1" in
    annotate) annotate_model_id ;;
    coding)   printf '%s' "${LLM_CODING_MODEL:-}" ;;
  esac
}

role_port() {
  case "$1" in
    annotate) printf '%s' "${LLM_ANNOTATE_PORT:-8000}" ;;
    coding)   printf '%s' "${LLM_CODING_PORT:-8001}" ;;
  esac
}

role_max_len() {
  case "$1" in
    annotate) printf '%s' "${LLM_ANNOTATE_MAX_MODEL_LEN:-8192}" ;;
    coding)   printf '%s' "${LLM_CODING_MAX_MODEL_LEN:-8192}" ;;
  esac
}

# ── Process / health helpers ──────────────────────────────────────────────────

vllm_pids() {  # $1 = model id
  [ -n "${1:-}" ] || return 1
  ps aux | grep -F "vllm serve $1" | grep -v grep | awk '{print $2}'
}

port_ready() {  # $1 = port
  [ -n "${1:-}" ] || return 1
  curl -sf "http://localhost:$1/health" >/dev/null 2>&1
}

print_endpoint() {  # $1 role  $2 model  $3 port
  local model_short
  model_short="$(basename "$2")"
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  [$1]  ${model_short}   (port :$3)"
  echo "    in-job:  http://localhost:$3/v1"
  if [ -n "${UCLOUD_PUBLIC_IP:-}" ]; then
    echo "    laptop:  http://${UCLOUD_PUBLIC_IP}:$3/v1   (open port $3 TCP on the IP)"
  else
    echo "    laptop:  set UCLOUD_PUBLIC_IP in .env to expose this port."
  fi
  echo "═══════════════════════════════════════════════════════════════"
}

ensure_vllm() {
  if ! ( cd "${PROJECT}" && uv run python -c "import vllm" ) 2>/dev/null; then
    echo "--- vllm not in .venv; running: uv sync --extra serve ---"
    ( cd "${PROJECT}" && uv sync --extra serve )
  fi
}

# ── Start one role ────────────────────────────────────────────────────────────

serve_one() {  # $1 role  $2 gpu_mem_util
  local role="$1" util="$2" model port max_len log attempts
  model="$(role_model "${role}")"
  port="$(role_port "${role}")"
  max_len="$(role_max_len "${role}")"
  log="${PROJECT}/outputs/interim/vllm-${role}.log"

  if [ -z "${model}" ]; then
    echo "ERROR: no model resolved for role '${role}'." >&2
    if [ "${role}" = "annotate" ]; then
      echo "       Ensure config/religious_missions.yaml has a vLLM bake-off candidate." >&2
    else
      echo "       Set LLM_CODING_MODEL in .env." >&2
    fi
    exit 1
  fi

  mkdir -p "$(dirname "${log}")"

  if [ -n "$(vllm_pids "${model}")" ]; then
    echo "--- [${role}] already running: ${model} (PID $(vllm_pids "${model}" | tr '\n' ' ')) ---"
    print_endpoint "${role}" "${model}" "${port}"
    return 0
  fi

  echo "--- [${role}] starting: ${model} ---"
  echo "    port=${port}  tp=${LLM_TENSOR_PARALLEL:-1}  max-len=${max_len}  gpu-mem=${util}"
  # On Blackwell (SM100) vLLM auto-selects FlashAttention v4 for the model's
  # vision tower (Gemma-3 is multimodal). FA4 lazily imports the CuTe DSL
  # (`import cutlass`), and the resolved nvidia-cutlass-dsl 4.5.2 wheel is
  # internally broken — `cutlass/cute/core.py` does `from .tuple import ... unwrap`
  # but that symbol no longer exists, so engine warmup's profile_run dies with
  # `ImportError: cannot import name 'unwrap' from 'cutlass.cute.tuple'`. We never
  # send images (text-only annotation), so pin the FA version to 2 — a prebuilt
  # kernel that needs no CuTe/cutlass import — to bypass the broken FA4 path.
  # (Harmless no-op for DeepSeek-V4-Flash: its MLA-sparse backend never
  # consults flash_attn_version — verified by inspection of the selected
  # DEEPSEEK_SPARSE_SWA backend.)
  #
  # DeepSeek-V4-Flash hits the SAME broken cutlass-dsl wheel a second and
  # third way (Lightning Indexer + KV compressor kernels), with no CLI flag
  # to dodge it. utils/vllm_compat/sitecustomize.py works around both --
  # see its header comment and docs/RUNNING_ON_UCLOUD.md's "DeepSeek-V4-Flash
  # setup" section for the full diagnosis. PYTHONPATH-loading it is harmless
  # for every other model (it only intercepts deepseek_v4-specific imports),
  # so it's applied unconditionally here rather than gated per-model.
  local extra_flags=()
  if [ "${model}" = "deepseek-ai/DeepSeek-V4-Flash" ]; then
    extra_flags=(
      --kv-cache-dtype fp8
      --max-num-batched-tokens "${max_len}"
      --trust-remote-code
      --tokenizer-mode deepseek_v4
      --reasoning-parser deepseek_v4
      --tool-call-parser deepseek_v4
      --enable-auto-tool-choice
    )
  fi
  ( cd "${PROJECT}" && PYTHONPATH="${PROJECT}/utils/vllm_compat:${PYTHONPATH:-}" \
      nohup uv run vllm serve "${model}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --served-model-name "${model}" \
      --tensor-parallel-size "${LLM_TENSOR_PARALLEL:-1}" \
      --max-model-len "${max_len}" \
      --gpu-memory-utilization "${util}" \
      --dtype auto \
      --attention-config '{"flash_attn_version": 2}' \
      --api-key "${LLM_API_KEY:-sk-ucloud}" \
      "${extra_flags[@]}" \
      &>>"${log}" & )
  echo "    logs: ${log}"

  # The first launch on Blackwell JIT-compiles FlashInfer kernels (~5 min) before
  # /health responds; allow a generous window but fail fast if the proc dies.
  echo -n "    waiting for /health (first Blackwell launch compiles kernels)"
  attempts=0
  until port_ready "${port}"; do
    sleep 5
    attempts=$((attempts + 1))
    echo -n "."
    if [ -z "$(vllm_pids "${model}")" ]; then
      echo ""
      echo "ERROR: [${role}] vLLM process exited before becoming healthy." >&2
      echo "       Last 50 log lines (${log}):" >&2
      tail -n 50 "${log}" >&2 || true
      exit 1
    fi
    if [ "${attempts}" -ge 180 ]; then
      echo ""
      echo "ERROR: [${role}] not healthy after 15 min — tail -f ${log}" >&2
      exit 1
    fi
  done
  echo " ready (${attempts}×5s)"
  print_endpoint "${role}" "${model}" "${port}"
}

# ── --stop ────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--stop" ]]; then
  echo "--- Stopping vLLM ---"
  if ! load_env 0; then
    echo "  .env not found; cannot identify models safely. Nothing stopped."
    exit 0
  fi
  roles="${2:-${ALL_ROLES}}"
  for r in ${roles}; do
    m="$(role_model "${r}")"
    if [ -n "${m}" ] && [ -n "$(vllm_pids "${m}")" ]; then
      kill $(vllm_pids "${m}") 2>/dev/null && echo "  [${r}] stopped (${m})."
    else
      echo "  [${r}] not running."
    fi
  done
  exit 0
fi

# ── --status ──────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--status" ]]; then
  if ! load_env 0; then
    echo "status: unknown (.env not found)"
    exit 0
  fi
  roles="${2:-${ALL_ROLES}}"
  for r in ${roles}; do
    m="$(role_model "${r}")"
    p="$(role_port "${r}")"
    state="stopped"
    [ -n "${m}" ] && [ -n "$(vllm_pids "${m}")" ] && state="running"
    ready="no"
    port_ready "${p}" && ready="yes"
    echo "[${r}] ${state}  ready=${ready}  model=${m:-<unset>}  port=${p}"
    [ "${state}" = "running" ] && print_endpoint "${r}" "${m}" "${p}"
  done
  exit 0
fi

# ── Start (annotate | coding | both) ──────────────────────────────────────────

MODE="${1:-}"
case "${MODE}" in
  annotate | coding | both) ;;
  "" | -h | --help)
    echo "${USAGE}" >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown role '${MODE}'." >&2
    echo "${USAGE}" >&2
    exit 2
    ;;
esac

load_env 1

if [ ! -d "${PROJECT}" ]; then
  echo "ERROR: project drive not found at ${PROJECT}." >&2
  echo "       Attach the drives and run init.sh first." >&2
  exit 1
fi

if [ -z "${CUDA_HOME:-}" ]; then
  echo "WARNING: CUDA_HOME is empty — on a B200, FlashInfer JIT will fail at warmup." >&2
  echo "         Run utils/init.sh (loads CUDA/12.8.0) or set CUDA_MODULE in .env." >&2
fi

ensure_vllm

if [ "${MODE}" = "both" ]; then
  # Split VRAM so both servers coexist on one GPU.
  serve_one annotate "${LLM_GPU_MEM_UTIL:-0.45}"
  serve_one coding "${LLM_GPU_MEM_UTIL:-0.45}"
else
  # Sole tenant: give the one model most of the GPU. 0.95 (not 0.90) because
  # DeepSeek-V4-Flash's ~146 GiB weights leave only ~37 GiB of headroom on a
  # 183 GiB B200 at 0.90 -- too little for KV cache + CUDA graph memory once
  # other overhead is accounted for. 0.95 was verified to leave a working
  # ~14 GiB KV cache (~40K tokens) for it; smaller models have ample room at
  # either setting.
  serve_one "${MODE}" "${LLM_GPU_MEM_UTIL_SOLO:-0.95}"
fi
