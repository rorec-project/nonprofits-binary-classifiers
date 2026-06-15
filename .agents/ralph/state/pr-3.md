# Ralph state — pr-3 (feature/07-evaluation)

## Task board
| Task | Status | Iteration | Notes |
|---|---|---|---|
| T3.1 | done | 1 | Config + G3 plumbing implemented and Tier-1 verified. PR-2 dependency sentinels passed; branch created from `feature/06-training` HEAD. |
| T3.2 | done | 2 | Calibration utilities implemented and Tier-1 verified. |
| T3.3 | done | 2 | Threshold, subgroup, and decision-curve utilities implemented and Tier-1 verified. |
| T3.4 | done | 3 | Stage entrypoint, CLI wrapper, evaluate-stage tests, and slow e2e stage-07 extension implemented and verified. |

## Task reports

### T3.1 — config + G3 plumbing

FILES:
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/config.py` — added `AcceptanceCriteria`, `EvaluationConfig`, `TestUnlock`, `load_test_unlock`, and root `evaluation` config field.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/qc/preflight.py` — added pure G3 test-unlock validation for stage 07.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/run_pipeline.py` — added stage 07 module mapping and G3 gate before running evaluation.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/religious_missions.yaml` — added explicit production evaluation defaults.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_evaluation_gate.py` — added G3 missing, unconfirmed, acceptance-drift, SHA-mismatch, and happy-path tests.

TESTS:
- Subagent: `uv run pytest tests/test_evaluation_gate.py tests/test_preflight.py tests/test_run_pipeline.py` — `26 passed in 0.96s`.
- Subagent: `uv run pytest -m "not slow and not network"` — `139 passed, 4 deselected, 24 warnings in 299.20s (0:04:59)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `38 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `139 passed, 4 deselected, 24 warnings in 299.12s (0:04:59)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `38 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run python - <<'PY' ...` loaded production and smoke YAMLs and confirmed evaluation defaults/overrides.

DEVIATIONS: none.

RISKS: none.

ARTIFACTS:
- Added the `test_unlock.json` schema in code: `{confirmed, checkpoint, checkpoint_sha256, acceptance: {min_pr_auc, min_minority_f1_ci_lower, max_ece}, rationale}`. No production unlock artifact was created.

### T3.2 — calibration

FILES:
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/evaluation/__init__.py` — exports the calibration utility API.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/evaluation/calibration.py` — implements Platt/temperature fitting, application, OOF cross-fit selection, calibration metrics, reliability points, and JSON round-trip helpers.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_calibration.py` — adds calibration metric, prior-shift, OOF discipline, and serialization tests.

TESTS:
- Subagent: `uv run pytest tests/test_calibration.py` — `4 passed in 0.81s`.
- Subagent: `uv run pytest -m "not slow and not network"` — `150 passed, 4 deselected, 24 warnings in 319.28s (0:05:19)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — after formatting owned `calibration.py`, `All checks passed!`; `43 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest tests/test_calibration.py tests/test_thresholds.py tests/test_subgroups.py tests/test_decision_curve.py` — `11 passed in 0.75s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `150 passed, 4 deselected, 24 warnings in 296.27s (0:04:56)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `43 files already formatted`; `All checks passed!`.

DEVIATIONS: none.

RISKS:
- None for T3.2. The workspace still contains prior T3.1 uncommitted/untracked files as expected by the state journal.

ARTIFACTS:
- No persisted artifacts written. T3.2 exposes serializable in-memory calibrator/report payloads used later by T3.4: `{method, params}` and `{methods, winner, deployed, crossfit_folds, ece_bins}`.

### T3.3 — thresholds + subgroups + decision curve

FILES:
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/evaluation/thresholds.py` — implements precision-floor and max-F1 threshold selection with PR-curve points and unattainable-floor fallback.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/evaluation/subgroups.py` — implements subgroup diagnostics with word-count bins and small-cell suppression.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/evaluation/decision_curve.py` — implements Vickers-Elkin net-benefit points with treat-all/treat-none references.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_thresholds.py` — adds hand-computed threshold policy and fallback tests.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_subgroups.py` — adds subgroup metric and suppression tests.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_decision_curve.py` — adds manual net-benefit tests.

TESTS:
- Subagent: `uv run pytest tests/test_thresholds.py tests/test_subgroups.py tests/test_decision_curve.py` — final run `7 passed in 0.65s`.
- Subagent: `uv run pytest -m "not slow and not network"` — `150 passed, 4 deselected, 24 warnings in 325.24s (0:05:25)`.
- Subagent: `uv run ty check` — `All checks passed!`. Its first full lint chain observed other-owned `calibration.py` needing format; T3.2 subsequently formatted it and the orchestrator verified the full lint/type chain green.
- Orchestrator verification: `uv run pytest tests/test_calibration.py tests/test_thresholds.py tests/test_subgroups.py tests/test_decision_curve.py` — `11 passed in 0.75s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `150 passed, 4 deselected, 24 warnings in 296.27s (0:04:56)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `43 files already formatted`; `All checks passed!`.

DEVIATIONS: none.

RISKS:
- None for T3.3 after orchestrator verification; full Tier-1 is green.

ARTIFACTS:
- None.

### T3.4 — stage entrypoint

FILES:
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/evaluation/evaluate.py` — added stage-07 `run_evaluation` entrypoint, selected-model SHA verification, anchor calibration/thresholding, rule validation, one-shot frozen-test evaluation, and artifact writers.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/07_evaluate.py` — added thin CLI wrapper for stage 07.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_evaluate_stage.py` — added happy-path/ordering, one-shot refusal, and acceptance-failure tests with an injected predictor.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_e2e_stages_05_11.py` — extended the slow synthetic e2e through stage 07 with tmp-registry unlock and one-shot refusal coverage.

TESTS:
- Subagent: `uv run pytest tests/test_evaluate_stage.py tests/test_e2e_stages_05_11.py -q` — `4 passed, 2 warnings in 4.48s`.
- Subagent: `uv run pytest -m "not slow and not network"` — `153 passed, 4 deselected, 24 warnings in 299.09s (0:04:59)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `45 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest tests/test_evaluate_stage.py tests/test_e2e_stages_05_11.py -q` — `4 passed, 2 warnings in 4.44s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `153 passed, 4 deselected, 24 warnings in 295.25s (0:04:55)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `45 files already formatted`; `All checks passed!`.

DEVIATIONS:
- The exact live `scripts/07_evaluate.py --config config/smoke.yaml` command was not run because preparing it would require fabricating smoke gate artifacts in the real `data/processed/gold/` tree; the tmp-registry slow e2e covers the same stage-07 smoke behavior, including confirmed unlock and one-shot re-run refusal, without touching production human-only artifacts.

RISKS:
- None.

ARTIFACTS:
- `calibrator.json`: `{method, params, threshold, threshold_policy, precision_floor, max_f1_threshold, fitted_on, crossfit_folds, anchor_oof_scores_path}`.
- `anchor_oof_scores.parquet`: `EIN2, prob_raw, prob_calibrated_oof, human_label, tier, sample_prob`.
- `rule_validation.json`: LOW-anchor rule counts plus sensitivity/specificity/precision/recall with Wilson CIs.
- `test_evaluation.json`: metric bundle, subgroups, decision-curve points, anchor-OOF calibration metrics, acceptance verdict, and metadata.

## Iteration journal

## Iteration 1 — 2026-06-15

Attempted T3.1 only, per A.2's one-unit limit. First-iteration pre-flight passed: `src/binary_classifier/train/trainer.py`, `src/binary_classifier/train/crossfit.py`, `tests/test_e2e_stages_05_11.py`, `_STAGE_MODULES["06"]`, and `TrainingConfig.targets/crossfit_folds` were present. Created/switched to `feature/07-evaluation` from the human-prepared `feature/06-training` HEAD.

T3.1 implemented the evaluation config models, `load_test_unlock`, G3 preflight checks, stage-07 pipeline mapping/gate, production YAML evaluation defaults, and focused G3 tests. I reviewed the diff and independently reran the task acceptance checks:

- `uv run pytest -m "not slow and not network"` — `139 passed, 4 deselected, 24 warnings in 299.12s (0:04:59)`.
- `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `38 files already formatted`; `All checks passed!`.
- YAML smoke-load check confirmed `config/religious_missions.yaml` has default evaluation settings and `config/smoke.yaml` retains smoke overrides (`min_pr_auc=0.50`, `max_ece=0.30`, `bootstrap_resamples=200`, `crossfit_folds=2`).

No DEVIATIONS.md row was needed. Next action: run the `[parallel-ok]` T3.2 + T3.3 group in the next iteration; do not start T3.4 until both are done.

## Iteration 2 — 2026-06-15

Attempted exactly one `[parallel-ok]` unit of work: T3.2 + T3.3. I dispatched one `ralph-implementer` for each task with the required pasted task package, file-ownership map, repo conventions, relevant §3.2/§6 context, and current DEVIATIONS.md rows.

T3.2 implemented `evaluation/calibration.py`, `evaluation/__init__.py`, and `tests/test_calibration.py`. T3.3 implemented `evaluation/thresholds.py`, `evaluation/subgroups.py`, `evaluation/decision_curve.py`, and their three focused test files. I reviewed the new source files and independently reran the task acceptance checks:

- `uv run pytest tests/test_calibration.py tests/test_thresholds.py tests/test_subgroups.py tests/test_decision_curve.py` — `11 passed in 0.75s`.
- `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `43 files already formatted`; `All checks passed!`.
- `uv run pytest -m "not slow and not network"` — `150 passed, 4 deselected, 24 warnings in 296.27s (0:04:56)`.

No DEVIATIONS.md row was needed. The status file remains `RUNNING` because T3.4 is still todo. Next action: run T3.4 only (stage entrypoint, script, evaluate-stage tests, and E2E extension), then if all tasks are done run the full PR-3 gate including smoke E2E through stage 07 before committing.

## Iteration 3 — 2026-06-15

Although the launch prompt said iteration 2, the state journal already contained a completed iteration 2 for the T3.2+T3.3 `[parallel-ok]` group. I followed the append-only state as the source of truth and attempted exactly one remaining unit of work: T3.4.

T3.4 implemented the stage-07 evaluation entrypoint and wrapper, plus focused and slow-e2e tests. I reviewed the implementation and independently reran the task acceptance checks and the PR-3 gate:

- `uv run pytest tests/test_evaluate_stage.py tests/test_e2e_stages_05_11.py -q` — `4 passed, 2 warnings in 4.44s`.
- `uv run pytest -m "not slow and not network"` — `153 passed, 4 deselected, 24 warnings in 295.25s (0:04:55)`.
- `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `45 files already formatted`; `All checks passed!`.

The PR-specific smoke E2E through stage 07 is covered by `tests/test_e2e_stages_05_11.py`, which fabricates the confirmed unlock in a tmp registry and confirms the one-shot refusal. I did not write any production human-only artifacts. No DEVIATIONS.md row was needed because the only deviation was the verification route (tmp-registry e2e instead of a live `config/smoke.yaml` run in the real data tree), and it does not change a `CONTEXT.md` fact or affect downstream PRs.

All PR-3 tasks are done and the full gate is green. Next action: commit this PR-3 branch for human review, write `DONE` to `.agents/ralph/state/pr-3.status`, and do not push.
