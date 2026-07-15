---
created: 2026-06-10
---

# Plan-parity verification: glimmering-knuth (ORIGINAL vs CURRENT)

Read-only check. ORIGINAL = `~/.claude/plans/...-glimmering-knuth.md`;
CURRENT = `.agents/plans/...-glimmering-knuth.md`. Neither file modified.

## Verdict

**Unintended drift found — 3 items lost that are NOT among the 7 known-intentional changes.**

1. `python-standards.md` doc-sync target DROPPED (silent).
2. `run_annotation_matrix` wiring DROPPED from Workstream B.
3. Silver-symlink re-point step LOST from the A3 setup note (partial).

## Mapping table

| Original item | Present in current? | Intentional (1-7)? / FLAG |
|---|---|---|
| WS A1 gitignore gold-tracking fix | Yes (A1, current:68-79) | n/a (kept; +parquet guard = #2) |
| WS A2 docs: layout + symlink-as-DVC + DVC migration note | Yes (A2/A3, current:81-90) | n/a |
| WS A2 fix "gold is committed" doc claim | Revised | #2 (AGENTS.md:43 rewrite, README reconcile) |
| WS A3 on-disk move = user action (documented) | Yes (current:87-88) | n/a |
| WS A3 "re-point the silver symlink to data/interim/" | **Partial loss** | **FLAG — not in 7. Current only says user "moves gold" (current:88); re-point step gone** |
| WS B canary loader replaces CANARY_EIN2 | Revised | #3 (now from monitor slice) |
| WS B source = held-out validation | Revised | #3 (now monitor slice carved from gold) |
| WS B wire through run_annotation **and run_annotation_matrix** | **Lost (matrix)** | **FLAG — not in 7. Current B names only run_annotation.py; run_annotation_matrix absent** |
| WS B monitoring-only docstring + --canary help | Yes (current:111-113) | n/a |
| WS C quarantine D-S + CROWDLAB (NotImplementedError) | Yes (current:118-123) | n/a |
| WS C keep both in dispatch; majority default | Yes (current:121) | n/a |
| WS D thread thresholds into build_silver/build_gold_set | Revised | #7 (build_silver_pool:51, build_gold_set:169) |
| WS D remove fresh QThresholdsConfig() at :91/:193 | Yes (current:128-130) | n/a |
| WS D quality.py fallback handling | Revised | #7 (keep quality.py:593 fallback) |
| WS D fix compute_quality_score docstring (MEDIUM 3.0-5.0) | Yes (current:132-133) | n/a |
| WS E remove dead _read_prompt_file | Yes (current:138-139, +orphaned Path import) | n/a |
| WS E load_missions parquet de-dup | Dropped | #5 (explicitly dropped, current:146-147) |
| WS E guided_json honesty (wire flag) | Yes (current:140-141) | n/a |
| WS E doc-sync: README:~83 | Yes (current:96-97 reconcile) | #2 |
| WS E doc-sync: configuration.md silver_dir | Revised | #2 (fix BOTH gold_dir + silver_dir) |
| WS E doc-sync: **python-standards.md → data/interim/manifests/** | **DROPPED** | **FLAG — not in 7. Absent from current (grep: 0 hits)** |
| WS E doc-sync: pipeline.md results/ → data/ | Revised | #2 (edit dropped — no results/ string) |
| WS E rename test_foundation.py:77 | Yes (current:144-145) | n/a |
| Canary on held-out (not prompt_dev) intent | Revised | #3 (monitor slice) |
| Out-of-scope: "dedicated monitoring slice" deferred | Revised (now in-scope) | #1 (monitor split added in W0) |
| Out-of-scope: fine-tuning, dvc init, rubric magic numbers, D-S real arm | Yes (current:177-178) | n/a |
| Verification: gitignore / tests / aggregators / canary / Q-threshold / smoke | Yes (current:182-195) | n/a, extended |
| — (new) Workstream 0: annotate silver∪gold, freeze-gate fix, monitor split, silver-only freeze scope | Added | #1 |
| — (new) Workstream F: abstain/evidence parity in run_quality_check | Added | #4 |
| — (new) "Tests to ADD" section | Added | #6 |

## Evidence

- python-standards DROP: ORIGINAL:134 `docs/agents/conventions/python-standards.md`
  example path → `data/interim/manifests/...`. CURRENT grep for
  `python-standards|conventions|interim/manifests` → NONE FOUND. Change #2 only lists
  *corrections* to pipeline.md / configuration.md / AGENTS.md / README — it does not
  authorize removing python-standards.md from the doc-sync set.
  (Note: file shows `M` in git status, so it may already be edited on the branch —
  but plan-parity ≠ working-tree state; the plan no longer tracks this target.)

- run_annotation_matrix DROP: ORIGINAL:92 "Wire it through `run_annotation` /
  `run_annotation_matrix` (both already receive registry/cfg)." CURRENT grep for
  `run_annotation_matrix|matrix` → NONE FOUND. Current B references only
  `run_annotation.py` (current:108, 114).

- silver re-point LOSS: ORIGINAL:83-84 user should "move `train_test_datasets/gold/*` →
  `data/processed/gold/`, **and re-point the silver symlink** to `data/interim/`"
  (also ORIGINAL:19). CURRENT:88 setup note says only the user "moves gold to
  `data/processed/gold/`"; no re-point instruction. "silver pool" at current:84 is
  descriptive (what stays symlinked), not the action.

## Items correctly accounted for by the 7 intentional changes
#1 Workstream 0 (silver∪gold, freeze-gate bug, monitor split, silver-only freeze scope) — present.
#2 reviewer doc-sync corrections (pipeline.md no results/, configuration.md both keys,
   AGENTS.md:43 rewrite, README whole-file reconcile) — present.
#3 canary from monitor slice — present.
#4 Workstream F abstain parity — present.
#5 load_missions de-dup dropped — present (explicit).
#6 "Tests to ADD" section — present.
#7 WS D function names/lines + quality.py:593 fallback — present.
