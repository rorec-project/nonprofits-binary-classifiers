# Orchestrator Protocol — Pipeline Roadmap Completion (Stages 05–11)

> Part 1 is §2 of the source plan (`we-are-still-working-eager-catmull.md`),
> reproduced verbatim. Part 2 contains the BINDING execution adaptations for the
> opencode harness and the human-in-the-loop Ralph runner (`.agents/ralph/ralph.sh`).
> On conflict about harness mechanics (how to spawn, commit trailers, iteration
> pacing, state/status files), Part 2 wins; on conflict about engineering content,
> Part 1 and `CONTEXT.md` win. §-references resolve in `CONTEXT.md`; T-numbers in
> the `pr-*.md` work orders.

## Part 1 — §2 of the source plan (verbatim)

## 2. Orchestrator protocol and subagent contract

**Orchestrator (one per PR):**
1. Create the PR branch from the current mainline (`git switch -c <branch>`).
2. Read §3 (context pack), §4 (decision record), §5 (target architecture), §6
   (config/registry spec), and your PR's work order in §7 — fully, before spawning.
3. Spawn one subagent per task (T-numbers). Tasks within a PR may run in parallel
   ONLY where the work order marks them `[parallel-ok]`; otherwise sequential.
4. After all tasks report, run the PR acceptance gate (§7, per PR) yourself:
   `uv run pytest -m "not slow and not network"` + `uv run ruff check . &&
   uv run ruff format --check . && uv run ty check` must be green, plus the
   PR-specific checks.
5. Commit with a conventional message (`feat:`/`chore:`/`fix:`), with no
   attribution trailers (Part 2 A.1). Do not push or open a PR unless the
   human asks.

**Subagent input contract** (the orchestrator includes in each spawn prompt):
(a) the task's full text from §7; (b) the §3 code-map rows it references; (c) the §6
spec blocks it implements; (d) the repo conventions block (§3.1).

**Subagent report-back contract** (every subagent's final message MUST contain):
1. `FILES`: every file created/modified, with absolute path and a one-line summary each.
2. `TESTS`: exact commands run and their pass/fail counts (paste the summary line).
3. `DEVIATIONS`: any departure from the task spec, with justification — or "none".
4. `RISKS`: anything discovered that affects other tasks/PRs — or "none".
5. `ARTIFACTS`: any new artifact schema actually produced (if it differs from §5.4, that is a DEVIATION).

A task is done only when its listed acceptance checks pass locally. Subagents must
not modify files owned by another task in the same PR (ownership is listed per task);
shared files (`config.py`, `paths.py`, `run_pipeline.py`, `preflight.py`,
`pyproject.toml`, the YAMLs) are owned by exactly one task per PR.

---

## Part 2 — Execution adaptations (binding)

### A.1 Harness mapping (opencode)

- You are the `ralph-orchestrator` primary agent (`.opencode/agents/
  ralph-orchestrator.md`), launched non-interactively by `.agents/ralph/ralph.sh`
  via `opencode run`. Each launch = ONE fresh-context Ralph iteration: your only
  memory is the plan documents, the Ralph state files, and git history.
- "Spawn one subagent per task" (§2 step 3) = invoke the **Task tool** with the
  `ralph-implementer` subagent, one Task invocation per T-task. `[parallel-ok]`
  groups MAY be dispatched as multiple Task invocations in a single message when the
  harness supports it; otherwise run them back-to-back within the iteration — they
  are parallel-OK, not parallel-required. Tasks not marked `[parallel-ok]` are
  strictly sequential.
- The subagent input contract (§2, items a–d) is fulfilled by PASTING into the Task
  prompt: (a) the task's full text from your `pr-N.md`; (b) the `CONTEXT.md` §3.2
  code-map rows it references; (c) the §6 spec blocks it implements; (d) the §3.1
  repo-conventions block; PLUS (e) the task's file-ownership list and the files
  owned by OTHER tasks in the PR (which it must not touch); (f) any
  `.agents/ralph/state/DEVIATIONS.md` rows touching the task's facts. Subagents
  start fresh — paste content, do not merely reference files.
- No automatic attribution trailers. Do not add `Co-Authored-By`,
  `Signed-off-by`, or any other automatic attribution footer. The human
  author controls all footers.
- Never `git push`, never open or merge PRs (also denied by agent permissions). The
  human merges at PR boundaries.

### A.2 Ralph iteration protocol (one launch = one iteration)

Each iteration, in this order:

1. **Read fully**: `CONTEXT.md` → this file → your `pr-N.md` →
   `.agents/ralph/state/DEVIATIONS.md` → `.agents/ralph/state/pr-N.md` (always
   present — the runner seeds a `NOT YET SEEDED` stub before the first iteration).
   Then inspect reality: `git status`, `git log --oneline -15`, current branch.
   If the working tree contains changes you cannot attribute to the state journal or
   git history, STOP and write `BLOCKED: unexplained working-tree state` (A.2 #7) —
   never build on unexplained state.
2. **First iteration only** (`state/pr-N.md` still contains the runner's
   `NOT YET SEEDED` stub): run your `pr-N.md` pre-flight
   (dependency sentinels). Switch to the PR branch, creating it from the HEAD the
   human prepared if absent (`git switch <branch>` / `git switch -c <branch>`).
   Seed `state/pr-N.md` from the A.3 template (one task-board row per T-task).
3. **Pick ONE unit of work**: the first task (in work-order sequence) whose status is
   not `done` — either one sequential task or one `[parallel-ok]` group. Do not start
   more than that per iteration: bounded iterations beat context exhaustion. If all
   tasks are `done`, go to step 6.
4. **Spawn and verify**: dispatch `ralph-implementer` per A.1. On report-back,
   VERIFY — re-run the task's listed acceptance checks yourself; do not trust pass
   claims. Then record the condensed report
   (FILES / TESTS / DEVIATIONS / RISKS / ARTIFACTS) under "Task reports" in
   `state/pr-N.md` and update the task board.
5. **Propagate deviations**: any DEVIATION that changes a `CONTEXT.md` fact or
   affects other tasks/PRs gets a row appended to `state/DEVIATIONS.md` (format A.4)
   in the same iteration.
6. **PR gate**: when ALL tasks are `done`, run the full acceptance gate yourself —
   the Part-1 step-4 commands (`uv run pytest -m "not slow and not network"` +
   `uv run ruff check . && uv run ruff format --check . && uv run ty check`) plus the
   PR-specific gate at the bottom of `pr-N.md`. Green → commit per Part-1 step 5
   (no attribution trailers, per A.1), INCLUDING the updated state files, then write `DONE` to the
   status file. Not green → record the failure precisely in the journal; fix within
   this iteration only if small, otherwise leave an exact next-action note and end
   the iteration.
7. **Status file** (`.agents/ralph/state/pr-N.status`; the runner parses its FIRST
   line after every iteration). Write it as your LAST action, only for terminal
   states — the runner pre-sets `RUNNING`:
   - `DONE` — PR gate green and committed; the loop stops for human review.
   - `BLOCKED: <one-line reason>` — you need something only the human can provide
     (credentials, a decision, a dependency PR missing from the tree).
   - `FAILED: <one-line reason>` — unrecoverable without a human decision (e.g. a
     CONTEXT.md fact is wrong in a way that invalidates the work order).
   - anything else (incl. `RUNNING`) — the runner launches the next fresh iteration.
8. **Journal**: append an `## Iteration <n> — <ISO date>` entry to `state/pr-N.md`:
   what was attempted, verified results (paste test summary lines), and the exact
   next action. The next iteration's orchestrator is fresh-context — write for it.

### A.3 State journal template (`state/pr-N.md` — on the first iteration the orchestrator replaces the runner's stub with this)

```markdown
# Ralph state — pr-N (<branch>)

## Task board
| Task | Status | Iteration | Notes |
|---|---|---|---|
| TN.1 | todo | — | |
<one row per T-task; status ∈ todo | in-progress | done | blocked>

## Task reports
<### TN.x — condensed FILES / TESTS / DEVIATIONS / RISKS / ARTIFACTS per task>

## Iteration journal
<## Iteration <n> — <date> entries, append-only>
```

### A.4 DEVIATIONS.md row format

`| date | PR | task | fact changed (CONTEXT.md § or work-order point) | what was done instead | why | downstream impact |`

Deviations are a living overlay. Rows are kept for the historical record
even after the deviation is resolved — resolved rows have a follow-up
entry documenting the fix.

### A.5 Honesty and safety rules (binding)

- **Never probe existence with Read**: opencode aborts the whole session when a
  `Read` targets a missing file. For pre-flight dependency sentinels, resume
  checks, and any artifact that may legitimately be absent, probe with bash
  (`test -f`, `ls`) or the glob tool first, and `Read` only paths the probe
  confirmed. A missing dependency sentinel is a `BLOCKED: <what is missing>`
  status, not a crash. Include this rule in every subagent package.
- Never mark a task `done` without having re-run its acceptance checks in THIS
  iteration and pasted the summary lines into the journal.
- Never weaken, skip, or xfail a test to make a gate pass — that is a `FAILED`
  status, not a workaround.
- Trust `CONTEXT.md` facts; re-verify only on conflict (§3 preamble). A confirmed
  conflict goes to `DEVIATIONS.md`.
- Respect file ownership: never touch files owned by another task in the same PR;
  shared files (`config.py`, `paths.py`, `run_pipeline.py`, `preflight.py`,
  `pyproject.toml`, the YAMLs) have exactly one owner per PR (Part 1).
- **Smoke vs production artifacts**: during Tier-2 verification (§8) you MAY
  programmatically fill smoke coding templates and write a smoke
  `selected_model.json` / confirmed `test_unlock.json`. You must NEVER create or
  modify the production human-only artifacts — anything §5.4 lists with producer
  "human" under `data/processed/gold/` for the production config
  (`config/religious_missions.yaml`). Smoke runs that would write into the real
  `data/` tree follow T1.3's acceptance pattern: use a scratch checkout/worktree,
  and never force past a clobber protection on production paths.
- **Tier-3 is human-gated**: the §8 Tier-3 real-data subset (pre-merge bar for
  PR-2/3/4) needs real data and HF downloads and ~1 h — do NOT run it unattended.
  When your PR lists it, note it as pending in the journal; the human runs or
  explicitly waives it at the PR boundary.
