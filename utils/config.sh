#!/usr/bin/env bash
# utils/config.sh — shared UCloud drive configuration for the binary-classifier pipeline.
#
# Edit this file on first use. Sourced by utils/init.sh, utils/run.sh, utils/devenv.sh,
# utils/sync-results.sh, utils/tmux-session.sh, and utils/serve-llm.sh — one place to
# set drive names and git identity, instead of duplicating them across six scripts.
#
# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this block on first use
# ════════════════════════════════════════════════════════════════════════════

# Mounted directory NAMES under /work for the two Drives. These are LOAD-BEARING
# and must match what you observe in `ls /work` on a live job:
#   - a member-files drive often mounts with a #NNNN suffix
#     (e.g. "AlessandroPizzigolotto#6144")
#   - a user-created drive usually mounts as its (possibly sanitized) Title.
# The values below are PLACEHOLDERS — confirm them on the first job and edit.
DATA_DRIVE="CHANGE_ME_DATA_DRIVE"       # Drive A: shared corpus + reusable caches
PROJECT_DRIVE="CHANGE_ME_PROJECT_DRIVE" # Drive B: this repo + heavy outputs

# Git identity for commits made from UCloud. Change to your own name/email
# or use your GitHub noreply address for privacy.
GIT_NAME="YOUR_NAME"
GIT_EMAIL="YOUR_EMAIL"

# ── LLM serving (coding assistant / Opencode) ────────────────────────────────
# Model for interactive coding assistance.  Qwen3.6-35B-A3B fits any UCloud GPU
# node (A40/H100). Bump to 32B on a B200 for stronger completions. Must be a
# HuggingFace model id — weights cache under HF_HOME on the DATA_DRIVE.
LLM_CODING_MODEL="Qwen3.6-35B-A3B"

# Port for the coding vLLM instance. Uses 8000 by default on an interactive-only
# job. Change to 8001 to avoid clashing with the annotation vLLM if co-locating
# on the same B200 job.
LLM_CODING_PORT=8000

# Tensor-parallel size: 1 = single GPU. Raise for larger models.
LLM_TENSOR_PARALLEL=1

# Maximum context length (tokens). 8192 fits most coding tasks.
LLM_MAX_MODEL_LEN=8192

# API key accepted by vLLM (--api-key). Any non-empty string; Opencode sends
# it as Bearer token. Not a secret — it is only an access control for the
# local network/public IP. Set a real secret here if your Public IP is exposed.
LLM_API_KEY="sk-ucloud"

# Static public IP allocated in UCloud (Resources → IP addresses).
# One-time allocation: create the IP in UCloud, specify port LLM_CODING_PORT
# TCP, then paste the assigned address here. It does NOT change between jobs —
# attach the same IP resource at every job submission.
# Leave empty if using only job-local access (no laptop access needed).
UCLOUD_PUBLIC_IP="" # e.g. "130.225.164.42"

# ════════════════════════════════════════════════════════════════════════════
# END CONFIGURATION — do not edit below unless you know what you are doing
# ════════════════════════════════════════════════════════════════════════════
