# PR-3 — `feature/07-evaluation`

> Standalone work order for the per-PR orchestrating agent, extracted VERBATIM from
> §7 of `.agents/plans/we-are-still-working-eager-catmull.md` (2026-06-12). All
> §-references resolve in `CONTEXT.md`; T-numbers of other PRs resolve in their
> `pr-N.md`. The binding orchestration protocol — subagent input/report contracts,
> Ralph iteration rules, state/status files, smoke-vs-production guardrails — is
> `ORCHESTRATOR.md`. Read `CONTEXT.md` §1–§5 (plus the §6 blocks listed below) and
> `ORCHESTRATOR.md` in full before spawning any subagent.

| | |
|---|---|
| Branch | `feature/07-evaluation` |
| Depends on | PR-2 — sentinels: `train/trainer.py` + `train/crossfit.py` exist; `_STAGE_MODULES["06"]` wired in `scripts/run_pipeline.py`; `TrainingConfig` has `targets`/`crossfit_folds`; `tests/test_e2e_stages_05_11.py` exists |
| Blocks | PR-4, PR-5 |
| Spec blocks implemented | §6.1 `EvaluationConfig`, `AcceptanceCriteria`, `TestUnlock`, `load_test_unlock`; §5.3 G3; §5.4 `calibrator`, `anchor_oof_scores`, `test_evaluation`, `rule_validation` schemas |
| Ralph state | `.agents/ralph/state/pr-3.md` + `pr-3.status` |

**Pre-flight (first iteration):** verify the PR-2 sentinels above; switch to / create
the branch. T3.1 wires G3 at the orchestrator TODO hook left by PR-1's T1.4.

**Human checkpoints:** the confirmed `test_unlock.json` written for Tier-2 step 5 is
SMOKE-ONLY (ORCHESTRATOR.md A.5) — the production unlock is human-authored
(§5.4: producer "human before 07"). §8 Tier-3 is part of the pre-merge bar for this
PR — human runs or waives it at the boundary (A.5).

**Task conventions (§7 preamble, verbatim):**

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

---

### PR-3 — `feature/07-evaluation`

**Objective**: calibrator + threshold from the anchor (cross-fit), rule validation,
G3, one-shot frozen-test evaluation with acceptance verdict.

**T3.1 — config + G3 plumbing** (owns `config.py`, `qc/preflight.py`,
`scripts/run_pipeline.py`, both YAMLs, `tests/test_evaluation_gate.py`)
Operations: add `EvaluationConfig`, `AcceptanceCriteria`, `TestUnlock`,
`load_test_unlock` (§6.1) + root field + YAML blocks. Preflight: `_validate_test_unlock
(cfg, registry) -> list[str]` (file exists; `confirmed`; acceptance snapshot ==
current config acceptance — field-by-field; sha matches `selected_model.json`'s);
wire as G3 for stage 07 at the orchestrator TODO hook from T1.4; add `"07"` to
`_STAGE_MODULES`. Tests: G3 variants (missing / unconfirmed / acceptance drift /
sha mismatch / happy path).
Acceptance: Tier-1 green.

**T3.2 — calibration** (owns `evaluation/__init__.py`, `evaluation/calibration.py`,
`tests/test_calibration.py`)
Operations: implement
- `fit_platt(scores, labels) -> {a, b}` (logistic on logit(score); use
  sklearn LogisticRegression on the 1-D logit feature) and
  `fit_temperature(scores, labels) -> {T}` (scalar NLL minimization via
  `scipy.optimize.minimize_scalar` — scipy ships with sklearn; if unavailable, golden
  section by hand);
- `crossfit_calibrate(scores, labels, folds, methods, seed) -> (oof_calibrated,
  per_method_metrics)` — OOF discipline per §4.2;
- metrics: Brier, log-loss, ECE (equal-width bins, `cfg.evaluation.ece_bins`),
  reliability-curve points (serializable);
- winner = best mean OOF Brier (log-loss tiebreak); refit winner on ALL anchor rows
  → deployed params.
Tests: **Platt absorbs a synthetic prior shift, temperature does not** (construct
scores calibrated under 30% prior, evaluate against 13%-prior labels — assert Platt
OOF Brier < temperature OOF Brier); ECE/Brier hand-checked values; OOF discipline
(no in-fold fitting); round-trip serialize/deserialize.
Acceptance: Tier-1 green.

**T3.3 — thresholds + subgroups + decision curve** (owns `evaluation/thresholds.py`,
`evaluation/subgroups.py`, `evaluation/decision_curve.py`, `tests/test_thresholds.py`,
`tests/test_subgroups.py`, `tests/test_decision_curve.py`) [parallel-ok with T3.2]
Operations:
- `pick_threshold(probs, labels, policy, precision_floor) -> {threshold, policy,
  achieved_precision, achieved_recall, max_f1_threshold, pr_curve_points}` —
  precision floor: sweep PR curve, among thresholds with precision ≥ floor take max
  recall; documented fallback if floor unattainable: threshold at max precision, flag
  `floor_unattainable: true`.
- `subgroup_report(df, y_true, y_pred, y_prob, *, by, length_bins, min_n)` — per
  NTEE major group and word-count bin: n, minority-F1, FPR, FNR; suppress (report
  `suppressed: true`) below `min_n`.
- `net_benefit(y_true, y_prob, thresholds) -> points` — NB(t) = TP/n − FP/n·t/(1−t),
  plus treat-all/treat-none reference lines (Vickers & Elkin 2006). Report-only.
Tests: hand-computed PR points / floor policy / fallback; subgroup suppression;
net-benefit values vs manual computation.
Acceptance: Tier-1 green.

**T3.4 — stage entrypoint** (owns `evaluation/evaluate.py`, `scripts/07_evaluate.py`,
`tests/test_evaluate_stage.py`)
Operations: `run_evaluation(cfg, registry, *, predictor=None) -> None` in this exact
order:
1. Load + verify `selected_model.json` (sha256 of checkpoint file matches; hard error
   with guidance if absent/mismatched). `predictor` kwarg (default: load the
   checkpoint; tests inject a stub with `predict_proba(texts) -> (n,2)`).
2. G4 re-check (anchor coded).
3. Predict raw probs on all anchor rows (join text via `load_missions`); run T3.2
   cross-fit; write `calibrator_path` + `anchor_oof_scores` (§5.4).
4. Threshold via T3.3 on the OOF-calibrated anchor scores; store in calibrator.json.
5. Rule validation: anchor LOW cells → `apply_rule_label` vs human labels →
   sens/spec/precision/recall + Wilson CIs → `rule_validation` (§5.4).
6. G3 re-check. **One-shot guard**: if `test_evaluation.json` exists → raise with
   "delete it explicitly to re-run" (loud, auditable; single-researcher acceptable).
7. Frozen-test eval (test split read HERE only, via an internal reader — not the
   T2.2 loader): discrimination bundle from `metrics.compute_metric_bundle` with
   `bootstrap_resamples`; subgroups; net-benefit points; calibration metrics on the
   anchor OOF (NOT on test — test is boundary-enriched, §3.3).
8. Acceptance verdict (min_pr_auc + min_minority_f1_ci_lower on test; max_ece on
   anchor OOF) — on failure raise like stage-04's freeze gate. Write
   `test_evaluation.json` (§5.4 incl. metadata).
Tests: full happy path with injected predictor stub + fabricated anchor/gold
artifacts on `tiny_registry`; one-shot refusal; acceptance failure raises; ordering
(calibrator written before any test read — assert via file mtimes or call recording).
Extend `tests/test_e2e_stages_05_11.py` through stage 07 (writes a confirmed
`test_unlock.json` fixture).
Acceptance: Tier-1 green; Tier-2 step 5 (§8) passes.

**PR-3 gate**: all green; smoke E2E through stage 07.

