# PR-4 — `feature/08-inference`

> Standalone work order for the per-PR orchestrating agent, extracted VERBATIM from
> §7 of `.agents/plans/we-are-still-working-eager-catmull.md` (2026-06-12). All
> §-references resolve in `CONTEXT.md`; T-numbers of other PRs resolve in their
> `pr-N.md`. The binding orchestration protocol — subagent input/report contracts,
> Ralph iteration rules, state/status files, smoke-vs-production guardrails — is
> `ORCHESTRATOR.md`. Read `CONTEXT.md` §1–§5 (plus the §6 blocks listed below) and
> `ORCHESTRATOR.md` in full before spawning any subagent.

| | |
|---|---|
| Branch | `feature/08-inference` |
| Depends on | PR-3 — sentinels: `evaluation/evaluate.py` exists; `registry.calibrator_path` producer code present; G3 wired for stage 07 |
| Blocks | PR-5 (and PR-7 relies on this PR's conventions) |
| Spec blocks implemented | §6.1 `InferenceConfig`; §5.4 `predictions_parquet` + `monitor_scores` schemas |
| Ralph state | `.agents/ralph/state/pr-4.md` + `pr-4.status` |

**Pre-flight (first iteration):** verify the PR-3 sentinels above; switch to / create
the branch. T4.2 owns this PR's `run_pipeline.py` edit (wires `"08"`).

**Human checkpoints:** §8 Tier-3 is part of the pre-merge bar for this PR
(`scripts/08_infer.py --limit 5000` with the tier-3 checkpoint) — human runs or
waives it at the boundary (ORCHESTRATOR.md A.5).

**Task conventions (§7 preamble, verbatim):**

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

---

### PR-4 — `feature/08-inference`

**T4.1 — config + router** (owns `config.py` (InferenceConfig + root + YAMLs),
`inference/__init__.py`, `inference/router.py`, `tests/test_inference_router.py`)
Operations: `route(text, tier, cfg) -> tuple[str, int | None]` pure function
implementing: tier in {HIGH, MEDIUM} → `("classifier", None)`; tier LOW and
`route_low_to_rules`: `apply_rule_label(text)` → 1 ⇒ `("rule_strong_positive", 1)`,
0 ⇒ `("rule_short_negative", 0)`, None ⇒ `("low_via_classifier", None)` if
`rule_ambiguous_to_classifier` else `("rule_abstain", None)`. Truth-table tests for
every branch.
Acceptance: Tier-1 green.

**T4.2 — predictor + sharding** (owns `inference/predict.py`, `scripts/08_infer.py`,
`tests/test_predict_stage.py`)
Operations: `run_inference(cfg, registry, *, predictor=None, limit=None) -> None`:
1. `load_missions(cfg)` (FULL corpus); compute Q/tier; apply T4.1 router.
2. Load `calibrator.json`; resolve device per §4.5 policy.
3. Shard into `cfg.inference.shard_size` chunks (stable order by EIN2). Per shard:
   classifier-routed rows → tokenize/batch (`cfg.inference.batch_size`),
   `torch.inference_mode()`, autocast bf16 only when resolved device is cuda+bf16;
   apply calibrator + threshold; rule rows get `prob_raw/prob_calibrated = NaN` and
   the router's label. Write `predictions_dir/shards/shard_{i:05d}.parquet`
   (§5.4 schema, all metadata columns stamped). **Resume = skip existing shards.**
4. Merge shards → `predictions_parquet`.
5. **Monitor scoring**: predict the monitor split (EIN2s from `monitor_manifest`),
   write `monitor_scores.json` (per-row calibrated probs + run metadata) for
   run-over-run drift diffing (§4.4).
Also wire `"08"` into `_STAGE_MODULES` (this task owns the run_pipeline.py edit for
PR-4; pass `limit=` through like stage 03 does).
Script flags: `--config --limit N`. Tests (predictor stub): schema incl. NaN rule
rows + decision_source mix; shard resume (pre-write a shard, assert skipped); EIN2
completeness (merged == input); monitor scores written; metadata stamped. Extend
`tests/test_e2e_stages_05_11.py` through stage 08.
Acceptance: Tier-1 green; Tier-2 step 6 first command.

**PR-4 gate**: all green; smoke inference produces rule-routed LOW rows + monitor file.

