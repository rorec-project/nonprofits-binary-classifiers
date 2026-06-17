#!/usr/bin/env bash
# utils/tmux-session.sh — tmux session manager for the binary-classifier pipeline
# on UCloud.
#
# Creates a named tmux session (derived from the repo directory name) that
# survives SSH disconnects. In a git worktree setup each checkout gets its own
# session name, so worktrees never collide.
# Usage:
#   bash utils/tmux-session.sh "06,07,08"    # create + start pipeline
#   bash utils/tmux-session.sh --kill          # kill existing session
#
# Inside the session: sources config.sh, sources env.sh, cds to the repo, and
# runs utils/run.sh.

set -euo pipefail

# ─── Guard: tmux must be installed ──────────────────────────────────────────
if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found. Install it or use batch mode." >&2
  exit 1
fi

. "$(dirname "$0")/config.sh"

REPO_DIR="/work/${PROJECT_DRIVE}"
ENV_SH="/work/${PROJECT_DRIVE}/env.sh"
SESSION="$(basename "${REPO_DIR}")"

# ─── Kill flag ──────────────────────────────────────────────────────────────
if [ "${1:-}" = "--kill" ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Killed session ${SESSION}."
  else
    echo "Session ${SESSION} does not exist."
  fi
  exit 0
fi

# ─── Check for existing session ──────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session ${SESSION} already exists. Attach with: tmux attach -t ${SESSION}"
  exit 0
fi

# ─── Stages argument ──────────────────────────────────────────────────────────
STAGES="${1:?usage: tmux-session.sh <stages>}"

# ─── Create session and send commands ───────────────────────────────────────
tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" ". '${REPO_DIR}/utils/config.sh'" Enter
tmux send-keys -t "$SESSION" ". '${ENV_SH}'" Enter
tmux send-keys -t "$SESSION" "cd '${REPO_DIR}'" Enter
tmux send-keys -t "$SESSION" "bash utils/run.sh ${STAGES}" Enter

echo "Started pipeline (stages: ${STAGES}) in tmux session '${SESSION}'."
echo "Re-attach with: tmux attach -t ${SESSION}"
