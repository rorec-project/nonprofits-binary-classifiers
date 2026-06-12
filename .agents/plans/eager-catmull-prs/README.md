# Pipeline Roadmap Completion (Stages 05–11) — PR Execution Pack

Operator guide for executing the frozen plan
`.agents/plans/we-are-still-working-eager-catmull.md` as 8 separately-run PRs, each
driven by a human-in-the-loop Ralph Wiggum loop on the **opencode CLI**
(<https://opencode.ai/docs>). The source plan stays untouched and authoritative;
this pack is a 1:1 mechanical split of it (verbatim by line range) plus the
execution machinery. Nothing was omitted — see "Adaptations" below for the complete
list of what was added or relocated.

## Document map

| File | Audience | Content |
|---|---|---|
| `CONTEXT.md` | agents | §1, §3–§6, §8–§10 of the source plan, **verbatim**, original numbering kept so all §-references resolve |
| `ORCHESTRATOR.md` | agents | Part 1 = §2 of the source plan verbatim; Part 2 = binding opencode/Ralph adaptations (A.1–A.5) |
| `pr-0.md` … `pr-7.md` | agents | one standalone work order per PR: §7 split **verbatim**, fronted by branch / dependency sentinels / human checkpoints / the §7 task-conventions preamble |
| `README.md` | human | this file |
| `../../ralph/ralph.sh` | human | the runner — one invocation = one PR campaign |
| `../../ralph/state/` | both | per-PR journals (`pr-N.md`), status files (`pr-N.status`, gitignored), `DEVIATIONS.md` (living overlay over CONTEXT.md) |
| `../../ralph/logs/` | human | per-iteration opencode transcripts (gitignored) |
| `/.opencode/agents/ralph-orchestrator.md` | opencode | primary agent: per-PR orchestrator (gpt-5.5, reasoning effort high via `opencode.json`) |
| `/.opencode/agents/ralph-implementer.md` | opencode | subagent: implements exactly one T-task, returns the 5-point report |

## PR overview and order

| PR | Branch | Delivers (stage) | Depends on |
|---|---|---|---|
| PR-0 | `chore/deps-transformers-v5` | dependency modernization: transformers v5, torch 2.7, vllm→extra, sentencepiece | — |
| PR-1 | `feature/05-anchor-sample` | stage 05 anchor sample (n=500, full frame incl. LOW) + shared `metrics.py` + all registry/config/gate plumbing | PR-0 |
| PR-2 | `feature/06-training` | stage 06: soft-label fine-tuning, baselines, OOF cross-fit, gated arms, selection report | PR-1 |
| PR-3 | `feature/07-evaluation` | stage 07: calibrator + threshold + rule validation + G3 + one-shot frozen-test eval | PR-2 |
| PR-4 | `feature/08-inference` | stage 08: calibrated predictions over 560k (classifier + rule router, sharded/resumable) | PR-3 |
| PR-5 | `feature/09-prevalence` | stage 09: weighted PPI++ composite prevalence, overall + per-NTEE | PR-4 |
| PR-6 | `feature/10-viz` | stage 10: figures from stage artifacts | PR-2 (full value after PR-5) |
| PR-7 | `feature/11-aggregation-compare` | stage 11: CROWDLAB/Dawid–Skene unlock + comparison vs majority | PR-2 + PR-4 conventions (e2e extension expects PR-6) |

Run them in numeric order unless you deliberately parallelize PR-6.

## Running one PR

Prerequisites (once):

- `opencode` installed and authenticated for the OpenAI provider; the model default
  is `openai/gpt-5.5` with `reasoningEffort: high` (pinned in `opencode.json`).
- A native arm64 toolchain — CONTEXT.md §9.9 documents the observed Rosetta x86_64
  uv/python failure; fixing it is PR-0 T0.1 step (2).
- The working tree contains the PR's dependencies (see its "Depends on" row) and
  `git status` is clean (the orchestrator refuses to build on unexplained state).
- `OPENAI_API_KEY` in the environment is only needed if you want the live Tier-2
  route (§8 step 2); the offline route does not need it.

Launch:

```bash
.agents/ralph/ralph.sh pr-0                 # autonomous, cap 12 iterations
.agents/ralph/ralph.sh pr-2 20              # custom iteration cap
RALPH_STEP=1 .agents/ralph/ralph.sh pr-3    # confirm between every iteration
RALPH_MODEL=openai/gpt-5.5 RALPH_YES=1 .agents/ralph/ralph.sh pr-1   # env overrides
```

Each iteration is one fresh-context `opencode run`: the orchestrator re-reads
CONTEXT.md → ORCHESTRATOR.md → its `pr-N.md` → the state files, advances exactly one
task (or one `[parallel-ok]` group) via `ralph-implementer` subagents, verifies the
work itself, journals to `state/pr-N.md`, and ends by writing the status file. When
all tasks are done it runs the full PR acceptance gate, commits (never pushes), and
writes `DONE`.

Exit codes: `0` DONE (gate passed, committed) · `2` iteration cap exhausted — review
the journal, re-run the same command to resume (resume is state-based and safe) ·
`3` BLOCKED on something only you can provide (the status line says what) ·
`4` FAILED, needs your decision · `1` usage/config error.

## Between PRs (the human gate)

1. Read `state/pr-N.md` (task board + condensed FILES/TESTS/DEVIATIONS/RISKS/
   ARTIFACTS reports + iteration journal) and `state/DEVIATIONS.md`.
2. Review the commits (`git log`, `git diff`). Optionally re-run the gate yourself:
   `uv run pytest -m "not slow and not network"` +
   `uv run ruff check . && uv run ruff format --check . && uv run ty check` + the
   PR-specific gate at the bottom of `pr-N.md`.
3. **PR-2/3/4 only**: the §8 Tier-3 real-data subset (~1 h, network + real data) is
   part of the pre-merge bar. The loop never runs it unattended — run it yourself or
   explicitly waive it.
4. Merge into your mainline (agents never push or merge), prepare the tree for the
   next PR, launch it.

## Human-only artifacts (production)

Agents may create the SMOKE analogues of these during Tier-2 verification only
(ORCHESTRATOR.md A.5); the production versions are yours alone (§5.4, §8 Tier-4):

- coding `data/processed/gold/anchor_to_code.csv` after the production stage-05 run;
- `data/processed/gold/selected_model.json` after reviewing `selection_report.json`;
- the acceptance criteria + `data/processed/gold/test_unlock.json` before stage 07;
- the stage-11 adoption decision (majority retained unless beaten per the adoption
  rule — adopting an arm invalidates `silver_labels.csv` ⇒ re-run stages 04→06).

## Adaptations from the source plan (everything else is verbatim)

1. §2 and §7 relocated into `ORCHESTRATOR.md` and `pr-*.md`; numbering preserved,
   stub pointers left in `CONTEXT.md` so every cross-reference still resolves.
2. Subagent spawning mapped to the opencode **Task tool** + `ralph-implementer`
   subagent; the §2 input/report contracts are unchanged (A.1).
3. Commit trailer adapted to `Co-Authored-By: opencode/gpt-5.5 <noreply@opencode.ai>`
   (A.1) — the source plan named a Claude trailer.
4. Ralph mechanics added (A.2–A.4): one task/group per fresh-context iteration,
   per-PR state journal, status-file protocol, append-only `DEVIATIONS.md` as the
   living overlay over CONTEXT.md.
5. Guardrails made explicit (A.5): smoke-vs-production artifact rule; Tier-3 is
   human-gated; never build on unexplained working-tree state.
6. Per-PR doc headers add branch, dependency sentinels, cross-PR notes, and human
   checkpoints — derived from §1, §7 task text, and §8; not present as such in §7.
