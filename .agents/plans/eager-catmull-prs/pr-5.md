# PR-5 — `feature/09-prevalence`

> Standalone work order for the per-PR orchestrating agent, extracted VERBATIM from
> §7 of `.agents/plans/we-are-still-working-eager-catmull.md` (2026-06-12). All
> §-references resolve in `CONTEXT.md`; T-numbers of other PRs resolve in their
> `pr-N.md`. The binding orchestration protocol — subagent input/report contracts,
> Ralph iteration rules, state/status files, smoke-vs-production guardrails — is
> `ORCHESTRATOR.md`. Read `CONTEXT.md` §1–§5 (plus the §6 blocks listed below) and
> `ORCHESTRATOR.md` in full before spawning any subagent.

| | |
|---|---|
| Branch | `feature/09-prevalence` |
| Depends on | PR-4 — sentinels: `inference/predict.py` exists; predictions schema (§5.4) incl. `decision_source` produced by the smoke route |
| Blocks | nothing hard (PR-6 reaches full value once this lands) |
| Spec blocks implemented | §6.1 `PrevalenceConfig`; §6.4 (PR-5 deps: `ppi-py`, `quapy`, deptry map); §5.4 `prevalence_report` + `prevalence_by_ntee` schemas |
| Ralph state | `.agents/ralph/state/pr-5.md` + `pr-5.status` |

**Pre-flight (first iteration):** verify the PR-4 sentinels above; switch to / create
the branch. T5.5 owns this PR's `run_pipeline.py` edit (wires `"09"`).

**Human checkpoints:** §9.1 import risks — if T5.1's `ppi_py`/`quapy` smoke imports
fail, the documented fallbacks engage (vendored EMQ, `importorskip` KDEy) and the
DEVIATION must be reviewed by the human at the boundary.

**Task conventions (§7 preamble, verbatim):**

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

---

### PR-5 — `feature/09-prevalence`

**T5.1 — deps** (owns `pyproject.toml`, `uv.lock`)
Operations: add `ppi-py>=0.2.3`, `quapy>=0.2.0,<0.3`; extend
`[tool.deptry.package_module_name_map]` with `ppi-py = ["ppi_py"]` (quapy imports as
`quapy`, no entry needed); `uv lock && uv sync`; smoke imports:
`uv run python -c "import ppi_py, quapy"` (numba and abstention are the known risks,
§4.5). If quapy fails: record DEVIATION, skip the dep, and rely on T5.4's vendored
EMQ (KDEy becomes `importorskip`-optional).
Acceptance: lock clean; imports OK (or documented fallback engaged); Tier-1 green.

**T5.2 — config + weights** (owns `config.py` (PrevalenceConfig + root + YAMLs),
`prevalence/__init__.py`, `prevalence/weights.py`, `tests/test_prevalence_weights.py`)
Operations: `design_weights(anchor_manifest_df, *, normalize=True) -> pd.Series`
(`w = 1/sample_prob`, normalized to mean 1 over the labeled set); join helper to align
weights/labels/predictions by EIN2 (string-normalized). Tests: weight math, alignment,
missing-EIN2 handling (drop + warn count).
Acceptance: Tier-1 green.

**T5.3 — PPI wrapper** (owns `prevalence/ppi.py`, `tests/test_ppi.py`)
Operations: `ppi_prevalence(y_labeled, yhat_labeled, yhat_unlabeled, *, alpha, w=None)
-> {estimate, ci_lower, ci_upper, lam, n_labeled, n_unlabeled, weighted}` calling
`ppi_py.ppi_mean_pointestimate` / `ppi_mean_ci` with `alpha=cfg.prevalence.alpha`
**explicitly** (library default is 0.1, §4.5) and `w=` when provided; `yhat` are
calibrated probabilities (floats — PPI on the mean of Y via prob predictions is the
standard prevalence use). The wrapper's returned dict MUST include the `alpha`
actually used. Tests: synthetic population with known prevalence — CI covers truth
across seeds; weighted vs unweighted differ when weights are skewed;
**assert `result["alpha"] == cfg.prevalence.alpha`** (guards the library's 0.1
default silently producing 90% CIs); `importorskip("ppi_py")`.
Acceptance: Tier-1 green.

**T5.4 — quantification cross-checks** (owns `prevalence/quantify.py`,
`tests/test_quantify.py`) [parallel-ok]
Operations: `emq_prevalence(val_posteriors, val_labels, corpus_posteriors,
max_iter=1000, tol=1e-6) -> float` — **vendor the SLD/EMQ EM loop directly**
(~40 lines: iterate prior re-weighting of posteriors until convergence; Saerens et
al. 2002) so the cross-check never depends on quapy resolving; ALSO provide
`quapy_emq_prevalence(...)` and `kdey_prevalence(...)` behind
`importorskip`-style guards using the 0.2.0 API (`EMQ(clf, fit_classifier=False)`,
`aggregate(posteriors)`; wrap our precomputed posteriors in a minimal sklearn-style
shim). Tests: vendored EMQ recovers a known prior shift on synthetic posteriors;
quapy parity test guarded by importorskip.
Acceptance: Tier-1 green.

**T5.5 — composite + entrypoint** (owns `prevalence/composite.py`,
`prevalence/estimate.py`, `scripts/09_prevalence.py`, `tests/test_prevalence_stage.py`)
Operations:
- `rogan_gladen(p_obs, sens, spec) -> float` + variance propagation (delta method) +
  clipping to [0,1];
- `composite(prev_by_stratum: dict[str, (est, var, share)]) -> (est, var)`
  (§4.2 formulas);
- `run_prevalence(cfg, registry) -> None`: G4 assumed (orchestrator); inputs =
  `predictions_parquet`, `anchor_oof_scores`, `rule_validation`, `anchor_manifest`;
  compute: HM stratum via T5.3 (labeled = anchor HM rows with OOF-calibrated scores;
  unlabeled = corpus HM calibrated probs; weighted AND unweighted variants); LOW
  stratum via rule labels + Rogan–Gladen with T3.4's sens/spec (+ sensitivity band
  over their Wilson CIs when `low_tier_sensitivity`); composite with corpus tier
  shares; cross-checks (T5.4) on the same inputs; per-NTEE loop with
  `ntee_min_n` suppression (EMQ point-estimate fallback for suppressed groups,
  flagged). Write `prevalence_report` + `prevalence_by_ntee` (§5.4) with explicit
  estimand statements.
Tests: Rogan–Gladen vs hand-computed; composite variance; full stage on fabricated
artifacts (known truth within CI); suppression behavior; report schema. Extend
`tests/test_e2e_stages_05_11.py` through stage 09.
Acceptance: Tier-1 green; Tier-2 step 6 second command. Wire `"09"` into
`_STAGE_MODULES` (this task owns the run_pipeline.py edit for PR-5).

**PR-5 gate**: all green; smoke E2E through stage 09.

