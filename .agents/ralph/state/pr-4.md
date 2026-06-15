# Ralph state — pr-4 (feature/08-inference)

## Task board
| Task | Status | Iteration | Notes |
|---|---|---|---|
| T4.1 | done | 1 | Router/config implemented; Tier-1 green in iteration 1. |
| T4.2 | done | 2 | Predictor/sharding implemented; Tier-1 and focused smoke/E2E checks green in iteration 2. |

## Task reports

### T4.1 — config + router

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/config.py` — added `InferenceConfig` and root `inference` field.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/religious_missions.yaml` — added explicit production `inference:` defaults.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/inference/__init__.py` — created inference package export for `route`.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/inference/router.py` — implemented pure LOW-tier routing function.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_inference_router.py` — added truth-table router tests.

TESTS
- Subagent: `uv run pytest tests/test_inference_router.py` — `7 passed in 0.21s`.
- Subagent: `uv run pytest -m "not slow and not network"` — `160 passed, 4 deselected, 24 warnings in 302.01s (0:05:02)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `47 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `160 passed, 4 deselected, 24 warnings in 321.06s (0:05:21)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `47 files already formatted`; `All checks passed!`.

DEVIATIONS
- none

RISKS
- none

ARTIFACTS
- none

### T4.2 — predictor + sharding

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/inference/predict.py` — added stage-08 `run_inference`, sharding/resume, calibration application, device/precision policy, and monitor scoring.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/08_infer.py` — added thin CLI wrapper with `--config` and `--limit`.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/run_pipeline.py` — wired stage `"08"` and `--infer-limit`.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_predict_stage.py` — added predictor-stub tests for schema, rule rows, shard resume, EIN2 completeness, monitor output, and metadata.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_e2e_stages_05_11.py` — extended the slow synthetic E2E through stage 08.

TESTS
- Subagent: `uv run pytest tests/test_predict_stage.py tests/test_inference_router.py` — final `9 passed in 1.47s`.
- Subagent: `uv run pytest tests/test_e2e_stages_05_11.py::test_e2e_stages_05_to_08_with_finetune_stub` — `1 passed, 2 warnings in 6.40s`.
- Subagent: `uv run pytest -m "not slow and not network"` — `162 passed, 4 deselected, 24 warnings in 298.80s`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `49 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest tests/test_predict_stage.py tests/test_inference_router.py` — `9 passed in 3.33s`.
- Orchestrator verification: `uv run pytest tests/test_e2e_stages_05_11.py::test_e2e_stages_05_to_08_with_finetune_stub` — `1 passed, 2 warnings in 4.47s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `162 passed, 4 deselected, 24 warnings in 301.90s (0:05:01)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `49 files already formatted`; `All checks passed!`.

DEVIATIONS
- The literal Tier-2 CLI command `scripts/08_infer.py --config config/smoke.yaml --limit 500` was not run in the repo checkout because prerequisite smoke `calibrator.json` and reviewed smoke `selected_model.json` artifacts are absent. I did not create production human-only artifacts. The predictor-stub tests and slow synthetic E2E verify the stage-08 smoke path, including rule-routed LOW rows and `monitor_scores.json`.

RISKS
- If `cfg.inference.rule_ambiguous_to_classifier=false`, T4.1 can emit `rule_abstain`, which is outside the §5.4 `decision_source` set; stage 08 now raises instead of writing a non-schema artifact.

ARTIFACTS
- `predictions_parquet` and shard parquet schema implemented as specified: `EIN2, pred_label, prob_raw, prob_calibrated, decision_source, tier, Q, ntee_major_group, model_id, checkpoint_sha256, calibrator_method, threshold, inference_date, pipeline_version, config_hash`.
- `monitor_scores.json` implemented as `{"metadata": {...}, "rows": [{"EIN2", "pred_label", "prob_raw", "prob_calibrated", "decision_source", "tier", "Q"}]}`; no production artifacts were written.

## Iteration journal

## Iteration 1 — 2026-06-16

Attempted first PR-4 iteration. Read the shared context, orchestration protocol,
PR-4 work order, deviations overlay, and seeded state. Inspected reality:
current branch was `refactor/harmonize-pipeline`; working tree only had the
runner-seeded untracked `.agents/ralph/state/pr-4.md`, which is attributable to
this Ralph launch. Pre-flight dependency sentinels passed: `evaluation/evaluate.py`
exists, `registry.calibrator_path` is written by `evaluation/evaluate.py`, and G3
is wired for stage 07 in `qc/preflight.py`. Created/switched to
`feature/08-inference` and seeded the state board.

Selected exactly one unit of work: T4.1 (config + router). Spawned one
`ralph-implementer` subagent with the full task package. The subagent implemented
the inference config block, production YAML defaults, the pure router, package
export, and truth-table tests. I independently verified Tier-1:

- `uv run pytest -m "not slow and not network"` — `160 passed, 4 deselected, 24 warnings in 321.06s (0:05:21)`.
- `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `47 files already formatted`; `All checks passed!`.

No deviations were introduced. Next action for iteration 2: run T4.2 (predictor +
sharding), including `run_pipeline.py` stage-08 wiring and the T4.2 tests; do not
start any other task in that iteration. PR gate is not yet due because T4.2 remains
todo. Tier-3 real-data inference remains human-gated at the PR boundary.

## Iteration 2 — 2026-06-16

Started from branch `feature/08-inference`. Working-tree changes were attributable
to T4.1 and the iteration-1 state journal. Selected exactly one remaining unit of
work: T4.2 (predictor + sharding). Spawned one `ralph-implementer` subagent with
the full task package. The subagent implemented stage-08 inference, the CLI, the
run-pipeline stage wiring, focused tests, and the stage-08 extension of the slow
synthetic E2E.

I independently verified the task and PR gate inputs:

- `uv run pytest tests/test_predict_stage.py tests/test_inference_router.py` — `9 passed in 3.33s`.
- `uv run pytest tests/test_e2e_stages_05_11.py::test_e2e_stages_05_to_08_with_finetune_stub` — `1 passed, 2 warnings in 4.47s`.
- `uv run pytest -m "not slow and not network"` — `162 passed, 4 deselected, 24 warnings in 301.90s (0:05:01)`.
- `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `49 files already formatted`; `All checks passed!`.

The literal Tier-2 smoke CLI command could not be run without first creating the
smoke `calibrator.json` and reviewed `selected_model.json`; per ORCHESTRATOR.md
A.5 I did not create production human-only artifacts in the repo checkout. The
focused predictor-stub test verifies rule-routed LOW rows plus `monitor_scores`,
and the slow synthetic E2E verifies stages 05→08 in-process with a stubbed
predictor. No CONTEXT.md fact changed, so `DEVIATIONS.md` was not appended.

All PR-4 tasks are now done, Tier-1 is green, and the PR-specific smoke behavior
is covered by local tests. Next action: commit the PR-4 changes and mark the Ralph
status `DONE`. Tier-3 real-data inference (`scripts/08_infer.py --limit 5000`
with the tier-3 checkpoint) remains human-gated at the PR boundary.
