# Ralph state — pr-1 (feature/05-anchor-sample)

## Task board
| Task | Status | Iteration | Notes |
|---|---|---|---|
| T1.1 | done | 2 | Metrics promotion implemented and verified. |
| T1.2 | done | 1 | Config/registry plumbing implemented and verified. |
| T1.3 | done | 3 | Anchor stage implemented and verified. |
| T1.4 | done | 4 | G4 gate and stage-05 orchestrator wiring implemented and verified. |

## Task reports

### T1.2 — registry + config plumbing

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/config.py` — added `AnchorConfig`, root `anchor` field, and explicit `ConfigDict(extra="ignore")` on `BinaryClassifierConfig`.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/paths.py` — added all §6.2 registry properties and the new `ensure_dirs()` directories.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/religious_missions.yaml` — added the default `anchor` block.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/smoke.yaml` — created the smoke config per §6.3.

TESTS
- Subagent: `uv run python -c "from binary_classifier.config import load_config; load_config('config/religious_missions.yaml'); load_config('config/smoke.yaml')"` — passed, exit 0, no output.
- Subagent: `uv run pytest -m "not slow and not network"` — `92 passed, 2 deselected, 2 warnings in 289.90s (0:04:49)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `26 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run python -c "from binary_classifier.config import load_config; load_config('config/religious_missions.yaml'); load_config('config/smoke.yaml')"` — passed, exit 0, no output.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `92 passed, 2 deselected, 2 warnings in 402.11s (0:06:42)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `26 files already formatted`; `All checks passed!`.

DEVIATIONS
- none

RISKS
- none

ARTIFACTS
- none

### T1.1 — metrics.py promotion

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/metrics.py` — created shared `compute_metric_bundle` and `bootstrap_ci` helpers, including optional ROC-AUC.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/qc/agreement.py` — delegated private metric helpers to the shared module and switched the default silver-label output path to `registry.silver_labels`.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_metrics.py` — added metric parity and ROC-AUC presence/absence coverage.

TESTS
- Subagent: `uv run pytest tests/test_metrics.py tests/test_agreement.py` — `19 passed in 10.78s`.
- Subagent: `uv run pytest -m "not slow and not network"` — first run timed out at 120s before summary; rerun: `94 passed, 2 deselected, 2 warnings in 293.52s (0:04:53)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `27 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `94 passed, 2 deselected, 2 warnings in 296.71s (0:04:56)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `27 files already formatted`; `All checks passed!`.

DEVIATIONS
- none

RISKS
- none

ARTIFACTS
- none

### T1.3 — anchor stage

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/data/anchor.py` — implemented stage-05 anchor sampling, stage-01 exclusion, tier×NTEE allocation, manifest/template writes, and clobber protection.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/05_build_anchor.py` — added thin CLI wrapper for `build_anchor` with `--config` and `--force`.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_anchor.py` — added allocation, sample-probability, LOW oversampling, exclusion/dtype-drift, determinism, clobber, and synthetic E2E tests.

TESTS
- Subagent: `uv run pytest tests/test_anchor.py` — `4 passed in 0.45s`.
- Subagent: `uv run pytest -m "not slow and not network"` — first attempt timed out at 120s; rerun: `98 passed, 2 deselected, 2 warnings in 292.75s`.
- Subagent: `uv run python scripts/05_build_anchor.py --config config/smoke.yaml` in `/tmp/opencode/anchor-smoke` — passed on rerun and wrote both 60-row artifacts.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `29 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest tests/test_anchor.py` — `4 passed in 0.60s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `98 passed, 2 deselected, 2 warnings in 302.85s (0:05:02)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `29 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run --project "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers" python "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/05_build_anchor.py" --config "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/smoke.yaml"` from `/tmp/opencode/pr1-t13-verify` — passed and wrote both artifacts.
- Orchestrator verification: scratch artifact schema check — `manifest=60 template=60`.

DEVIATIONS
- none

RISKS
- none

ARTIFACTS
- `anchor_manifest`: `EIN2, stratum, tier, ntee_major_group, sample_prob, split="anchor"`.
- `anchor_coding_template`: `EIN2, tier, text, human_label`.

### T1.4 — G4 gate + orchestrator wiring

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/qc/preflight.py` — added G4 anchor-label validation and `_STAGE_SPLITS` entries for stages 06/07.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/run_pipeline.py` — added stage 05 orchestration, force passthrough, extended G1 stage coverage, and G4/G3 gate hooks.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_preflight.py` — added G4 missing/partial/non-binary/complete tests and split-map coverage.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_run_pipeline.py` — added stage-05 import/force and G4 orchestrator coverage.

TESTS
- Subagent: `uv run pytest tests/test_preflight.py tests/test_run_pipeline.py` — `21 passed in 1.02s`.
- Subagent: `uv run pytest -m "not slow and not network"` — first attempt timed out at 120s; rerun: `106 passed, 2 deselected, 2 warnings in 285.64s (0:04:45)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — first attempt found format drift in `scripts/run_pipeline.py`; after `uv run ruff format ...`, rerun reported `All checks passed!`; `29 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest tests/test_preflight.py tests/test_run_pipeline.py` — `21 passed in 0.86s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — `106 passed, 2 deselected, 2 warnings in 291.92s (0:04:51)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `29 files already formatted`; `All checks passed!`.
- PR gate smoke: from `/tmp/opencode/pr1-final-smoke`, `uv run --project "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers" python "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/run_pipeline.py" --config "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/smoke.yaml" --stages 01,05` — passed; wrote 72-row smoke `gold_to_code.csv`, 60-row `anchor_manifest.csv`, and 60-row `anchor_to_code.csv` under scratch `data/`.

DEVIATIONS
- none. Subagent noted the prompt package labeled repo conventions under section `(c)` rather than a separate `(d)`; the binding conventions were still pasted, so this is not an implementation or context deviation.

RISKS
- none

ARTIFACTS
- none

## Iteration journal

## Iteration 1 — 2026-06-15

- Read the required plan/state files, inspected `git status`, recent history, and current branch. The only initial working-tree change was the runner-seeded untracked `.agents/ralph/state/pr-1.md`, which is explained by the first-iteration protocol.
- Ran the PR-1 pre-flight dependency sentinels against `pyproject.toml`: `transformers>=5.8.0,<6.0.0`, pytest `slow`/`network` markers, and `vllm` only under optional dependency group `serve` are present.
- Created and switched to `feature/05-anchor-sample` from prepared HEAD `0b89c1f`.
- Seeded this state journal from the A.3 template and dispatched exactly one work unit, T1.2, because the PR sequencing note requires T1.2 before T1.1.
- Verified T1.2 acceptance checks locally: config-load smoke command passed; offline pytest reported `92 passed, 2 deselected, 2 warnings in 402.11s (0:06:42)`; lint/format/type reported `All checks passed!`, `26 files already formatted`, `All checks passed!`.
- No deviations were reported or observed, so `.agents/ralph/state/DEVIATIONS.md` was left unchanged.
- Next action: dispatch T1.1 (`metrics.py` promotion) now that `registry.silver_labels` exists.

## Iteration 2 — 2026-06-15

- Read the required plan/state files, inspected `git status`, recent history, and current branch. The working-tree changes at launch matched the prior iteration's uncommitted T1.2 files plus the state journal, so they were explained by state/history.
- Dispatched exactly one work unit, T1.1 (`metrics.py` promotion). The subagent created `src/binary_classifier/metrics.py`, rewired `qc/agreement.py` to delegate to it, switched the silver-label default path to `registry.silver_labels`, and extended `tests/test_metrics.py`.
- Verified T1.1 acceptance checks locally: offline pytest reported `94 passed, 2 deselected, 2 warnings in 296.71s (0:04:56)`; lint/format/type reported `All checks passed!`, `27 files already formatted`, `All checks passed!`.
- No deviations were reported or observed, so `.agents/ralph/state/DEVIATIONS.md` was left unchanged.
- Next action: dispatch T1.3 (`anchor stage`).

## Iteration 3 — 2026-06-15

- Read the required plan/state files, inspected `git status`, recent history, and current branch. The working-tree changes at launch matched the prior iterations' uncommitted T1.2/T1.1 files plus the state journal, so they were explained by state/history.
- Dispatched exactly one work unit, T1.3 (`anchor stage`). The subagent created `src/binary_classifier/data/anchor.py`, `scripts/05_build_anchor.py`, and `tests/test_anchor.py`.
- Verified T1.3 acceptance checks locally: `uv run pytest tests/test_anchor.py` reported `4 passed in 0.60s`; offline pytest reported `98 passed, 2 deselected, 2 warnings in 302.85s (0:05:02)`; lint/format/type reported `All checks passed!`, `29 files already formatted`, `All checks passed!`.
- Verified the smoke script in scratch (not the production data tree): from `/tmp/opencode/pr1-t13-verify`, `uv run --project "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers" python "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/05_build_anchor.py" --config "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/smoke.yaml"` wrote `data/interim/manifests/anchor_manifest.csv` and `data/processed/gold/anchor_to_code.csv`; schema/count check printed `manifest=60 template=60`.
- No deviations were reported or observed, so `.agents/ralph/state/DEVIATIONS.md` was left unchanged.
- Next action: dispatch T1.4 (`G4 gate + orchestrator wiring`).

## Iteration 4 — 2026-06-15

- Read the required plan/state files, inspected `git status`, recent history, and current branch. The working-tree changes at launch matched the prior iterations' uncommitted T1.2/T1.1/T1.3 files plus this state journal, so they were explained by state/history.
- Dispatched exactly one work unit, T1.4 (`G4 gate + orchestrator wiring`). The subagent extended `qc/preflight.py`, `scripts/run_pipeline.py`, `tests/test_preflight.py`, and `tests/test_run_pipeline.py`.
- Verified T1.4 acceptance checks locally: `uv run pytest tests/test_preflight.py tests/test_run_pipeline.py` reported `21 passed in 0.86s`; offline pytest reported `106 passed, 2 deselected, 2 warnings in 291.92s (0:04:51)`; lint/format/type reported `All checks passed!`, `29 files already formatted`, `All checks passed!`.
- Ran the PR-1 gate because all task-board rows are now done. The smoke command from `/tmp/opencode/pr1-final-smoke`, `uv run --project "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers" python "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/run_pipeline.py" --config "/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/smoke.yaml" --stages 01,05`, completed successfully and wrote scratch stage-01/stage-05 artifacts only.
- No context-changing deviations were reported or observed, so `.agents/ralph/state/DEVIATIONS.md` was left unchanged.
- Next action: PR-1 is ready for final commit and `DONE` status.
