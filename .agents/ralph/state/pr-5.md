# Ralph state — pr-5 (`feature/09-prevalence`)

## Task board
| Task | Status | Iteration | Notes |
|---|---|---|---|
| T5.1 | done | 1 | Added prevalence deps; used published `ppi-python` package name (see DEVIATIONS). |
| T5.2 | done | 2 | Added PrevalenceConfig/root field and design-weight/EIN2 alignment helpers. |
| T5.3 | done | 3 | Added PPI++ prevalence wrapper and tests. |
| T5.4 | done | 4 | Added vendored EMQ plus optional QuaPy EMQ/KDEy cross-check wrappers. |
| T5.5 | done | 5 | Added composite estimators, stage 09 entrypoint/wiring, report writers, and tests. |

## Task reports

### T5.1 — deps

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/pyproject.toml` — added PR-5 prevalence deps and deptry import map entry.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/uv.lock` — regenerated lockfile with the new dependency closure.

TESTS
- Subagent: `uv lock` — initial FAIL: `No solution found ... ppi-py was not found in the package registry`; final PASS: `Resolved 259 packages`.
- Subagent: `uv sync` — PASS: `Installed 12 packages`.
- Subagent: `uv run python -c "import ppi_py, quapy"` — PASS: no output.
- Subagent: `uv run pytest -m "not slow and not network"` — PASS: `162 passed, 4 deselected, 24 warnings in 302.17s (0:05:02)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `49 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv lock && uv sync && uv run python -c "import ppi_py, quapy"` — PASS: `Resolved 259 packages in 0.62ms`; `Resolved 259 packages in 0.59ms`; `Audited 143 packages in 1ms`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — PASS: `162 passed, 4 deselected, 24 warnings in 299.83s (0:04:59)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `49 files already formatted`; `All checks passed!`.

DEVIATIONS
- Used `ppi-python>=0.2.3` with deptry map `ppi-python = ["ppi_py"]` instead of specified `ppi-py>=0.2.3` / `ppi-py = ["ppi_py"]` because `ppi-py` is not present in the package registry, while `ppi-python==0.2.3` provides the required `ppi_py` import and passed smoke import. Recorded in `.agents/ralph/state/DEVIATIONS.md`.

RISKS
- `quapy==0.2.0` pulls in `abstention==0.1.3.1`; smoke import passed locally.

ARTIFACTS
- none.

### T5.2 — config + weights

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/config.py` — added `PrevalenceConfig` and root `prevalence` field.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/config/religious_missions.yaml` — added explicit production `prevalence:` block.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/prevalence/__init__.py` — exported prevalence weight utilities.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/prevalence/weights.py` — added inverse-probability design weights and string-normalized EIN2 label/prediction alignment helper.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_prevalence_weights.py` — covered config defaults/YAML load, weight math, alignment, dropped-row warnings, and duplicate handling.

TESTS
- Subagent: `uv run pytest tests/test_prevalence_weights.py` — PASS: `10 passed in 0.22s`.
- Subagent: `uv run pytest -m "not slow and not network"` — PASS: `172 passed, 4 deselected, 24 warnings in 293.84s (0:04:53)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `51 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — PASS: `172 passed, 4 deselected, 24 warnings in 301.00s (0:05:01)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `51 files already formatted`; `All checks passed!`.

DEVIATIONS
- none.

RISKS
- none.

ARTIFACTS
- none.

### T5.3 — PPI wrapper

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/prevalence/ppi.py` — added PPI prevalence wrapper returning estimate, CI, lambda, alpha, row counts, and weighted flag.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_ppi.py` — added `pytest.importorskip("ppi_py")` coverage for known-prevalence CI coverage, alpha recording, and weighted-vs-unweighted behavior.

TESTS
- Subagent: `uv run pytest tests/test_ppi.py` — PASS: `7 passed in 0.73s`.
- Subagent: `uv run pytest -m "not slow and not network"` — PASS: `179 passed, 4 deselected, 24 warnings in 302.84s (0:05:02)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `52 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest tests/test_ppi.py` — PASS: `7 passed in 0.77s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — PASS: `179 passed, 4 deselected, 24 warnings in 305.09s (0:05:05)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `52 files already formatted`; `All checks passed!`.

DEVIATIONS
- none.

RISKS
- none.

ARTIFACTS
- none.

### T5.4 — quantification cross-checks

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/prevalence/quantify.py` — added vendored SLD/EMQ prior-shift prevalence estimation plus optional QuaPy EMQ/KDEy wrappers.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_quantify.py` — added synthetic prior-shift recovery, guarded QuaPy parity, and input-validation tests.

TESTS
- Subagent: `uv run pytest tests/test_quantify.py` — PASS: `6 passed in 1.94s`.
- Subagent: `uv run pytest -m "not slow and not network"` — PASS: `185 passed, 4 deselected, 24 warnings in 302.99s (0:05:02)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `53 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest tests/test_quantify.py` — PASS: `6 passed in 1.71s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — PASS: `185 passed, 4 deselected, 24 warnings in 303.70s (0:05:03)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `53 files already formatted`; `All checks passed!`.

DEVIATIONS
- none.

RISKS
- none.

ARTIFACTS
- none.

### T5.5 — composite + entrypoint

FILES
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/prevalence/composite.py` — added Rogan-Gladen point/variance helpers and stratified composite estimator.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/prevalence/estimate.py` — added stage 09 prevalence estimation, report/NTEE artifact writing, cross-checks, and schema validation.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/09_prevalence.py` — added thin CLI wrapper for stage 09.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/run_pipeline.py` — wired stage `"09"` into `_STAGE_MODULES` and execution order.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_prevalence_stage.py` — added T5.5 unit/stage tests.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_e2e_stages_05_11.py` — extended the slow synthetic E2E route through stage 09.

TESTS
- Subagent: `uv run pytest tests/test_prevalence_stage.py` — PASS: `3 passed in 0.85s`.
- Subagent: `uv run pytest tests/test_e2e_stages_05_11.py -m slow` — PASS: `1 passed, 2 warnings in 6.44s` (after an initial fixture adjustment).
- Subagent: `uv run pytest -m "not slow and not network"` — PASS: `188 passed, 4 deselected, 24 warnings in 314.99s (0:05:14)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `56 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest tests/test_prevalence_stage.py` — PASS: `3 passed in 0.86s`.
- Orchestrator verification: `uv run pytest tests/test_e2e_stages_05_11.py -m slow` — PASS: `1 passed, 2 warnings in 6.18s`.
- Orchestrator verification: `uv run pytest -m "not slow and not network"` — PASS: `188 passed, 4 deselected, 24 warnings in 295.18s (0:04:55)`.
- Orchestrator verification: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — PASS: `All checks passed!`; `56 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run python scripts/09_prevalence.py --config /tmp/opencode/pr5-stage09-smoke/config.yaml` on fabricated `/tmp/opencode` artifacts — PASS: wrote `prevalence_by_ntee.csv` and `prevalence_report.json`.

DEVIATIONS
- none.

RISKS
- none.

ARTIFACTS
- `prevalence_report` and `prevalence_by_ntee` are produced with the §5.4 schemas.

## Iteration journal

## Iteration 1 — 2026-06-16

- Read the required context, orchestrator protocol, PR-5 work order, DEVIATIONS overlay, and fresh `pr-5.md` state stub.
- Inspected reality before work: branch was `refactor/harmonize-pipeline`; `git status --short` showed only the runner-seeded untracked `.agents/ralph/state/pr-5.md`; recent history ended at `4ebed2e feat: add inference stage`.
- Ran first-iteration pre-flight: confirmed `src/binary_classifier/inference/predict.py` exists; found `decision_source` in inference code and smoke/integration tests; switched to new branch `feature/09-prevalence`; seeded this state journal.
- Advanced exactly one work unit: T5.1 deps.
- Verified T5.1 locally after subagent return:
  - `uv lock && uv sync && uv run python -c "import ppi_py, quapy"` passed with `Resolved 259 packages in 0.62ms`; `Resolved 259 packages in 0.59ms`; `Audited 143 packages in 1ms`.
  - `uv run pytest -m "not slow and not network"` passed with `162 passed, 4 deselected, 24 warnings in 299.83s (0:04:59)`.
  - `uv run ruff check . && uv run ruff format --check . && uv run ty check` passed with `All checks passed!`; `49 files already formatted`; `All checks passed!`.
- Appended a DEVIATIONS row for the package-name correction (`ppi-python` instead of nonexistent `ppi-py`).
- Next action: run iteration 2 and start T5.2 (`config.py` PrevalenceConfig/root + YAMLs, `prevalence/__init__.py`, `prevalence/weights.py`, `tests/test_prevalence_weights.py`).

## Iteration 2 — 2026-06-16

- Read the required context, orchestrator protocol, PR-5 work order, DEVIATIONS overlay, and existing `pr-5.md` state journal.
- Inspected reality before work: branch was `feature/09-prevalence`; `git status --short` showed only iteration-1 T5.1/state changes (`pyproject.toml`, `uv.lock`, `DEVIATIONS.md`, untracked `pr-5.md`); recent history still ended at `4ebed2e feat: add inference stage`.
- Advanced exactly one work unit: T5.2 config + weights.
- Verified the subagent touched only T5.2-owned files plus pre-existing T5.1/state changes; removed generated `__pycache__` files after test runs.
- Verified T5.2 locally after subagent return:
  - `uv run pytest -m "not slow and not network"` passed with `172 passed, 4 deselected, 24 warnings in 301.00s (0:05:01)`.
  - `uv run ruff check . && uv run ruff format --check . && uv run ty check` passed with `All checks passed!`; `51 files already formatted`; `All checks passed!`.
- No DEVIATIONS row was needed.
- Next action: run iteration 3 and start T5.3 (`prevalence/ppi.py`, `tests/test_ppi.py`) with the T5.1 package-name deviation in mind (`ppi-python` package, `ppi_py` import).

## Iteration 3 — 2026-06-16

- Read the required context, orchestrator protocol, PR-5 work order, DEVIATIONS overlay, and existing `pr-5.md` state journal.
- Inspected reality before work: branch was `feature/09-prevalence`; `git status --short` showed only explained iteration-1/2 PR-5 changes (`pyproject.toml`, `uv.lock`, `DEVIATIONS.md`, config/YAML/prevalence weights files/tests, and untracked state); recent history still ended at `4ebed2e feat: add inference stage`.
- Advanced exactly one work unit: T5.3 PPI wrapper.
- Verified the subagent touched only T5.3-owned files (`src/binary_classifier/prevalence/ppi.py`, `tests/test_ppi.py`) in addition to pre-existing T5.1/T5.2/state changes.
- Verified T5.3 locally after subagent return:
  - `uv run pytest tests/test_ppi.py` passed with `7 passed in 0.77s`.
  - `uv run pytest -m "not slow and not network"` passed with `179 passed, 4 deselected, 24 warnings in 305.09s (0:05:05)`.
  - `uv run ruff check . && uv run ruff format --check . && uv run ty check` passed with `All checks passed!`; `52 files already formatted`; `All checks passed!`.
- No DEVIATIONS row was needed.
- Next action: run iteration 4 and start T5.4 (`prevalence/quantify.py`, `tests/test_quantify.py`) for quantification cross-checks.

## Iteration 4 — 2026-06-16

- Read the required context, orchestrator protocol, PR-5 work order, DEVIATIONS overlay, and existing `pr-5.md` state journal.
- Inspected reality before work: branch was `feature/09-prevalence`; `git status --short` showed only explained iteration-1/2/3 PR-5 changes (`pyproject.toml`, `uv.lock`, `DEVIATIONS.md`, config/YAML/prevalence package files/tests, and untracked state); recent history still ended at `4ebed2e feat: add inference stage`.
- Advanced exactly one work unit: T5.4 quantification cross-checks.
- Verified the subagent touched only T5.4-owned files (`src/binary_classifier/prevalence/quantify.py`, `tests/test_quantify.py`) in addition to pre-existing T5.1/T5.2/T5.3/state changes.
- Verified T5.4 locally after subagent return:
  - `uv run pytest tests/test_quantify.py` passed with `6 passed in 1.71s`.
  - `uv run pytest -m "not slow and not network"` passed with `185 passed, 4 deselected, 24 warnings in 303.70s (0:05:03)`.
  - `uv run ruff check . && uv run ruff format --check . && uv run ty check` passed with `All checks passed!`; `53 files already formatted`; `All checks passed!`.
- No DEVIATIONS row was needed.
- Next action: run iteration 5 and start T5.5 (`prevalence/composite.py`, `prevalence/estimate.py`, `scripts/09_prevalence.py`, `tests/test_prevalence_stage.py`, stage-09 `run_pipeline.py` wiring, and the stage-09 E2E extension).

## Iteration 5 — 2026-06-16

- Read the required context, orchestrator protocol, PR-5 work order, DEVIATIONS overlay, and existing `pr-5.md` state journal.
- Inspected reality before work: branch was `feature/09-prevalence`; `git status --short` showed only explained iteration-1/2/3/4 PR-5 changes (`pyproject.toml`, `uv.lock`, `DEVIATIONS.md`, config/YAML/prevalence package files/tests, and untracked state); recent history still ended at `4ebed2e feat: add inference stage`.
- Advanced exactly one work unit: T5.5 composite + entrypoint.
- Verified the subagent touched T5.5-owned files (`src/binary_classifier/prevalence/composite.py`, `src/binary_classifier/prevalence/estimate.py`, `scripts/09_prevalence.py`, `scripts/run_pipeline.py`, `tests/test_prevalence_stage.py`, and `tests/test_e2e_stages_05_11.py`) in addition to pre-existing T5.1/T5.2/T5.3/T5.4/state changes.
- Verified T5.5 locally after subagent return:
  - `uv run pytest tests/test_prevalence_stage.py` passed with `3 passed in 0.86s`.
  - `uv run pytest tests/test_e2e_stages_05_11.py -m slow` passed with `1 passed, 2 warnings in 6.18s`.
  - `uv run pytest -m "not slow and not network"` passed with `188 passed, 4 deselected, 24 warnings in 295.18s (0:04:55)`.
  - `uv run ruff check . && uv run ruff format --check . && uv run ty check` passed with `All checks passed!`; `56 files already formatted`; `All checks passed!`.
  - `uv run python scripts/09_prevalence.py --config /tmp/opencode/pr5-stage09-smoke/config.yaml` on fabricated `/tmp/opencode` artifacts passed and wrote the stage-09 report/CSV.
- No DEVIATIONS row was needed.
- All PR-5 tasks are now done and the PR gate is green; committed the PR for human review.
- Next action: human review/merge boundary for PR-5.
