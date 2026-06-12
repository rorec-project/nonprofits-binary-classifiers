# PR-6 — `feature/10-viz`

> Standalone work order for the per-PR orchestrating agent, extracted VERBATIM from
> §7 of `.agents/plans/we-are-still-working-eager-catmull.md` (2026-06-12). All
> §-references resolve in `CONTEXT.md`; T-numbers of other PRs resolve in their
> `pr-N.md`. The binding orchestration protocol — subagent input/report contracts,
> Ralph iteration rules, state/status files, smoke-vs-production guardrails — is
> `ORCHESTRATOR.md`. Read `CONTEXT.md` §1–§5 (plus the §6 blocks listed below) and
> `ORCHESTRATOR.md` in full before spawning any subagent.

| | |
|---|---|
| Branch | `feature/10-viz` |
| Depends on | PR-2 — sentinel: `train/sweep.py` produces `results.jsonl`. Full value after PR-5 (`prevalence_by_ntee.csv` feeds the forest plot); the script skips missing inputs by design, so PR-6 may run before PR-5 lands |
| Blocks | nothing (PR-7's T7.2 e2e extension expects this PR's viz code in the tree) |
| Spec blocks implemented | §5.4 `figures_dir` |
| Ralph state | `.agents/ralph/state/pr-6.md` + `pr-6.status` |

**Pre-flight (first iteration):** verify the PR-2 sentinel above; switch to / create
the branch. Stage 10 is script-only — NOT wired into `_STAGE_MODULES` (§5.2).

**Human checkpoints:** none.

**Task conventions (§7 preamble, verbatim):**

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

---

### PR-6 — `feature/10-viz`

**T6.1 — figures** (owns `viz/__init__.py`, `viz/ngrams.py`, `viz/curves.py`,
`viz/prevalence_plots.py`, `tests/test_viz.py`)
Operations: pure functions, each `(data, ax: matplotlib.axes.Axes) -> None`, reading
ONLY stage artifacts (§5.4) — no model loading, no torch import:
- `ngram_log_odds(silver_df_with_text, top_k=30)`: CountVectorizer (1–2 grams,
  min_df=5) on religious vs non silver rows; log-odds with +1 smoothing; horizontal
  bar chart (substitutes the roadmap's word clouds — see §4.4 docs note).
- `documentation_curve(results_jsonl_rows)`: validation PR-AUC vs train_fraction per
  encoder, seed bands where >1 seed.
- `pr_curve(points)`, `reliability_diagram(points, ece)`: from
  `test_evaluation.json` / `calibrator.json` serialized points.
- `prevalence_forest(prevalence_by_ntee_df)`: estimates + CIs per NTEE group,
  suppressed groups greyed.
Tests: each renders to a tmp PNG under the `Agg` backend from fabricated inputs.
Acceptance: Tier-1 green.

**T6.2 — script + docs** (owns `scripts/10_visualize.py`,
`.agents/architecture/pipeline.md` (viz note), README viz note) [parallel-ok]
Operations: script renders every figure whose input artifact exists (skip + log
otherwise) into `figures_dir` as PNG and SVG; `--config` flag only. Docs: record the
word-cloud → log-odds substitution rationale.
Acceptance: Tier-1 green; smoke run renders ≥1 figure after Tier-2.

**PR-6 gate**: all green.

