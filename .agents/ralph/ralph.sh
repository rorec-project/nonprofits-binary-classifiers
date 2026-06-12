#!/usr/bin/env bash
# Human-in-the-loop Ralph Wiggum runner for the pipeline roadmap PRs (stages 05-11).
# One invocation = one PR campaign: fresh-context opencode iterations loop until the
# PR acceptance gate passes (DONE), the orchestrator needs the human (BLOCKED/FAILED),
# or the iteration cap is hit. State lives in .agents/ralph/state/ + git history.
#
# Usage:   .agents/ralph/ralph.sh <pr-0|pr-1|...|pr-7> [max-iterations]
#
# Env:     RALPH_MODEL      opencode provider/model   (default: openai/gpt-5.5;
#                           reasoning effort pinned to high in opencode.json)
#          RALPH_AGENT      opencode primary agent    (default: ralph-orchestrator)
#          RALPH_MAX_ITERS  iteration cap when arg 2 is absent (default: 12)
#          RALPH_STEP=1     pause for Enter between iterations (per-iteration HITL)
#          RALPH_YES=1      skip the launch confirmation
#
# Exit:    0 = PR gate passed (DONE)   2 = max iterations exhausted (re-run to resume)
#          3 = BLOCKED on human input  4 = FAILED, needs a human decision
#          1 = usage / configuration error
set -euo pipefail

PR_ID="${1:-}"
case "$PR_ID" in
  pr-[0-7]) ;;
  *) echo "usage: $0 <pr-0..pr-7> [max-iterations]" >&2; exit 1 ;;
esac
MAX_ITERS="${2:-${RALPH_MAX_ITERS:-12}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PLAN_DIR=".agents/plans/eager-catmull-prs"
STATE_DIR=".agents/ralph/state"
LOG_DIR=".agents/ralph/logs/$PR_ID"
STATUS_FILE="$STATE_DIR/$PR_ID.status"
STATE_FILE="$STATE_DIR/$PR_ID.md"
MODEL="${RALPH_MODEL:-openai/gpt-5.5}"
AGENT="${RALPH_AGENT:-ralph-orchestrator}"

for f in "$PLAN_DIR/CONTEXT.md" "$PLAN_DIR/ORCHESTRATOR.md" "$PLAN_DIR/$PR_ID.md"; do
  [[ -f "$f" ]] || { echo "[ralph] missing required plan file: $f" >&2; exit 1; }
done
command -v opencode >/dev/null 2>&1 || { echo "[ralph] opencode CLI not on PATH" >&2; exit 1; }
mkdir -p "$STATE_DIR" "$LOG_DIR"

# Seed the state journal so the orchestrator's unconditional Read never hits a
# missing file (opencode aborts the whole run on a failed Read). The orchestrator
# replaces this stub with the ORCHESTRATOR.md A.3 template on the first iteration.
if [[ ! -f "$STATE_FILE" ]]; then
  cat > "$STATE_FILE" <<EOF
# Ralph state — ${PR_ID}

NOT YET SEEDED — no iteration has run yet. Orchestrator: this is the FIRST
iteration. Run the ${PR_ID}.md pre-flight (dependency sentinels, branch), then
replace this file with the ORCHESTRATOR.md A.3 template (one task-board row per
T-task), per A.2 step 2.
EOF
fi

# Seed DEVIATIONS.md for the same reason (it is in the unconditional read list).
if [[ ! -f "$STATE_DIR/DEVIATIONS.md" ]]; then
  cat > "$STATE_DIR/DEVIATIONS.md" <<'EOF'
# DEVIATIONS — living overlay over CONTEXT.md

Read by every orchestrator and subagent together with
`.agents/plans/eager-catmull-prs/CONTEXT.md`; on conflict, this file wins.
One table row per deviation, appended by the PR orchestrator that produced it
(ORCHESTRATOR.md A.4/A.2 step 5). Never rewrite or delete rows — append only.

| Date | PR | Task | Fact changed (CONTEXT.md § / work-order point) | What was done instead | Why | Downstream impact |
|---|---|---|---|---|---|---|
EOF
fi

echo "=== Ralph Wiggum — $PR_ID ==="
echo "work order : $PLAN_DIR/$PR_ID.md"
echo "agent/model: $AGENT / $MODEL"
echo "max iters  : $MAX_ITERS"
echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
echo "git status :"
git status --short | head -20 || true
if [[ "${RALPH_STEP:-0}" == "1" && ! -t 0 ]]; then
  echo "[ralph] RALPH_STEP=1 needs an interactive terminal" >&2
  exit 1
fi
if [[ "${RALPH_YES:-0}" != "1" ]]; then
  if [[ -t 0 ]]; then
    echo "The working tree must already contain this PR's dependencies (see the"
    echo "'Depends on' row in $PLAN_DIR/$PR_ID.md)."
    read -r -p "[ralph] Launch? [Enter to start / Ctrl-C to abort] "
  else
    echo "[ralph] no terminal for the launch confirmation — re-run with RALPH_YES=1" >&2
    exit 1
  fi
fi

# Stall guard: a crashed opencode run that changed nothing (no journal update, no
# working-tree or HEAD movement) is counted as a strike; three consecutive strikes
# abort the campaign instead of burning the remaining iteration budget.
fingerprint() {
  { cat "$STATE_FILE" 2>/dev/null
    git status --porcelain 2>/dev/null
    git rev-parse HEAD 2>/dev/null
  } | shasum | cut -d' ' -f1
}
STRIKES=0

for ((i = 1; i <= MAX_ITERS; i++)); do
  printf 'RUNNING\n' > "$STATUS_FILE"
  PROMPT="You are the ralph-orchestrator for ${PR_ID} (iteration ${i} of ${MAX_ITERS}; fresh context — your only memory is the plan documents, the Ralph state files, and git history).

Read fully, in this order:
1. ${PLAN_DIR}/CONTEXT.md          (shared context pack; original section numbering)
2. ${PLAN_DIR}/ORCHESTRATOR.md     (Part 2 defines this iteration's protocol)
3. ${PLAN_DIR}/${PR_ID}.md         (your work order)
4. ${STATE_DIR}/DEVIATIONS.md and ${STATE_DIR}/${PR_ID}.md (the state journal; if it
   still says NOT YET SEEDED, this is the first iteration — see A.2 step 2)

Then execute exactly ONE Ralph iteration for ${PR_ID} per ORCHESTRATOR.md Part 2
(A.2): advance one task or one [parallel-ok] group via ralph-implementer subagents,
verify their work yourself, update ${STATE_DIR}/${PR_ID}.md, and finish by writing
the status file ${STATUS_FILE} (DONE / BLOCKED: reason / FAILED: reason — or leave
RUNNING for another iteration)."

  echo "--- iteration $i/$MAX_ITERS — $(date '+%Y-%m-%d %H:%M:%S') ---"
  LOG_FILE="$LOG_DIR/iter-$(printf '%02d' "$i").log"
  FP_BEFORE="$(fingerprint)"
  OC_EXIT=0
  opencode run --agent "$AGENT" --model "$MODEL" "$PROMPT" 2>&1 | tee "$LOG_FILE" || OC_EXIT=$?
  if (( OC_EXIT != 0 )); then
    echo "[ralph] opencode exited $OC_EXIT; the status file decides what happens next" >&2
  fi

  STATUS="$(head -n 1 "$STATUS_FILE" 2>/dev/null || echo RUNNING)"
  case "$STATUS" in
    DONE*)
      echo "[ralph] $PR_ID DONE after $i iteration(s)."
      echo "[ralph] Review $STATE_DIR/$PR_ID.md, $STATE_DIR/DEVIATIONS.md and the commits, then merge."
      exit 0 ;;
    BLOCKED*)
      echo "[ralph] $PR_ID blocked:${STATUS#BLOCKED:}"
      echo "[ralph] Resolve it, then re-run: $0 $PR_ID"
      exit 3 ;;
    FAILED*)
      echo "[ralph] $PR_ID failed:${STATUS#FAILED:}"
      echo "[ralph] See $STATE_DIR/$PR_ID.md and $LOG_DIR/"
      exit 4 ;;
  esac

  if (( OC_EXIT != 0 )) && [[ "$(fingerprint)" == "$FP_BEFORE" ]]; then
    STRIKES=$((STRIKES + 1))
    echo "[ralph] iteration $i crashed with no progress (strike $STRIKES/3)" >&2
    if (( STRIKES >= 3 )); then
      echo "[ralph] 3 consecutive crashed iterations without progress — aborting." >&2
      echo "[ralph] See $LOG_DIR/ for transcripts; fix the cause, then re-run." >&2
      exit 4
    fi
  else
    STRIKES=0
  fi

  if [[ "${RALPH_STEP:-0}" == "1" && $i -lt $MAX_ITERS ]]; then
    read -r -p "[ralph] iteration $i finished (status: $STATUS). Enter = next iteration, Ctrl-C = stop "
  fi
done

echo "[ralph] max iterations ($MAX_ITERS) exhausted without DONE."
echo "[ralph] Review $STATE_DIR/$PR_ID.md and $LOG_DIR/, then re-run to resume (state-based resume is safe)."
exit 2
