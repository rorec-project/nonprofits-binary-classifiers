# PR-7 — `feature/11-aggregation-compare`

> Standalone work order for the per-PR orchestrating agent, extracted VERBATIM from
> §7 of `.agents/plans/we-are-still-working-eager-catmull.md` (2026-06-12). All
> §-references resolve in `CONTEXT.md`; T-numbers of other PRs resolve in their
> `pr-N.md`. The binding orchestration protocol — subagent input/report contracts,
> Ralph iteration rules, state/status files, smoke-vs-production guardrails — is
> `ORCHESTRATOR.md`. Read `CONTEXT.md` §1–§5 (plus the §6 blocks listed below) and
> `ORCHESTRATOR.md` in full before spawning any subagent.

| | |
|---|---|
| Branch | `feature/11-aggregation-compare` |
| Depends on | PR-2 (the `oof_pred_probs` artifact producer `train/crossfit.py`) and PR-4 conventions (§1). T7.2 extends the e2e through stages 10–11 incl. viz renders ⇒ practically requires PR-6 in the tree as well |
| Blocks | nothing (final PR) |
| Spec blocks implemented | §6.1 `AggregationConfig`; §5.4 `aggregation_compare` schema |
| Ralph state | `.agents/ralph/state/pr-7.md` + `pr-7.status` |

**Pre-flight (first iteration):** verify `train/crossfit.py` and (for the full e2e
extension) `src/binary_classifier/viz/` exist; switch to / create the branch.
Stage 11 is script-only — NOT wired into `_STAGE_MODULES` (§5.2).

**Human checkpoints:** the adoption-rule verdict in the comparison report is
decision-support only — adopting a non-majority arm invalidates the frozen
`silver_labels.csv` ⇒ re-run stages 04→06 (stated in T7.2); that adoption decision
is the human's, and the code must never auto-switch `cfg.aggregation.method`.

**Task conventions (§7 preamble, verbatim):**

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

---

### PR-7 — `feature/11-aggregation-compare`

**T7.1 — config + unlock implementations** (owns `config.py` (AggregationConfig +
root + YAMLs), `src/binary_classifier/annotate/aggregate.py`,
`tests/test_aggregate_unlock.py`)
Operations: implement the quarantined arms (replacing the `NotImplementedError`
bodies at lines 85–135; PRESERVE the quarantine behavior when prerequisites are
missing):
- `aggregate_crowdlab(df, pred_probs: pd.DataFrame | None = None)`: if
  `pred_probs is None` → raise the existing quarantine error. Else: pivot the long
  store to an (EIN2 × source_id) label DataFrame (NaN = abstain/missing — cleanlab
  handles it, §4.5); align `pred_probs[["p0","p1"]]` by EIN2 (drop + warn on rows
  missing predictions); call `get_label_quality_multiannotator(labels, probs,
  quality_method="crowdlab")`; map the returned consensus + quality back to the
  EXACT majority_vote wide schema (§3.2: `EIN2, silver_label, silver_confidence
  (=consensus quality), num_votes, num_abstain, agreement, tie(=False)`).
- `aggregate_dawid_skene(df)`: long → `(task=EIN2, worker=source_id, label)` frame →
  `crowdkit DawidSkene(n_iter=100).fit_predict_proba`; same wide mapping; KEEP the
  correlated-annotators caveat in the docstring.
- `aggregate_labels(df, method, pred_probs=None)` dispatch grows the optional kwarg.
Tests: pivot shape/NaN handling on a fabricated store; drop-in schema equality vs
`majority_vote` columns; quarantine preserved (`crowdlab` without pred_probs raises);
small-N end-to-end for both arms (cleanlab/crowdkit run on ~20 rows).
Acceptance: Tier-1 green.

**T7.2 — comparison report** (owns `qc/aggregation_compare.py`,
`scripts/11_aggregation_compare.py`, `tests/test_aggregation_compare.py`)
Operations: `run_aggregation_compare(cfg, registry) -> None`: load the long store +
`oof_pred_probs` (PR-2 artifact — the out-of-sample probs cleanlab requires); for
majority + each arm in `cfg.aggregation.comparison_arms`: aggregate, join human
**validation** labels (template, `split=="validation"`), score with
`metrics.compute_metric_bundle` + bootstrap CIs; write `aggregation_compare`
(§5.4) including the **adoption rule verdict**: an arm may replace majority ONLY if
its minority-F1 CI lower bound > majority's point estimate (standing rule,
configuration.md), and the report MUST state that adoption invalidates the frozen
`silver_labels.csv` ⇒ re-run stages 04→06. Never auto-switch
`cfg.aggregation.method`.
Tests: fabricated store + OOF probs + coded validation → report schema, verdict
logic both ways. Extend `tests/test_e2e_stages_05_11.py` through stages 10–11
(viz renders + comparison report — completing the offline Tier-2 route).
Acceptance: Tier-1 green; smoke run after Tier-2 produces the report.

**T7.3 — docs** (owns configuration.md weak-supervision hooks, README) [parallel-ok]
Operations: update the "future weak-supervision arms" hook: CROWDLAB/Dawid-Skene now
implemented as gated comparison arms; adoption rule unchanged.
Acceptance: docs only.

**PR-7 gate**: all green.

