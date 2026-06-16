# Ralph state — pr-6 (`feature/10-viz`)

## Task board
| Task | Status | Iteration | Notes |
|---|---|---|---|
| T6.1 | done | 1 | Pure viz plotting helpers and rendering tests implemented; Tier-1 green |
| T6.2 | done | 2 | Script-only renderer, docs note, and smoke ≥1 figure verified |

## Task reports

### T6.1 — figures
FILES:
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/viz/__init__.py` — exports T6.1 plotting helpers.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/viz/ngrams.py` — adds silver-label n-gram log-odds bar plot.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/viz/curves.py` — adds documentation, PR, and reliability curve plots.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/src/binary_classifier/viz/prevalence_plots.py` — adds NTEE prevalence forest plot.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/tests/test_viz.py` — adds Agg backend tmp-PNG rendering tests.

TESTS:
- Subagent: `uv run pytest tests/test_viz.py -q` — `5 passed in 1.65s`.
- Subagent final: `uv run pytest tests/test_viz.py -q && uv run pytest -m "not slow and not network" && uv run ruff check . && uv run ruff format --check . && uv run ty check` — `5 passed`; `193 passed, 4 deselected, 24 warnings`; `All checks passed!`; `60 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run pytest -m "not slow and not network" && uv run ruff check . && uv run ruff format --check . && uv run ty check` — `193 passed, 4 deselected, 24 warnings in 304.79s`; `All checks passed!`; `60 files already formatted`; `All checks passed!`.
- Orchestrator verification: grep for `import torch|from torch` under `src/binary_classifier/viz` — no matches.

DEVIATIONS: none.

RISKS: none.

ARTIFACTS: none.

### T6.2 — script + docs
FILES:
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/scripts/10_visualize.py` — adds script-only stage-10 renderer with skip/log behavior and PNG+SVG output.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/.agents/architecture/pipeline.md` — notes visualization is script-only and not orchestrator-wired.
- `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers/README.md` — documents the word-cloud → n-gram log-odds substitution rationale.

TESTS:
- Subagent: `uv run python scripts/10_visualize.py --help` — passed; usage printed.
- Subagent: scratch smoke render with fabricated `results.jsonl` under `/tmp/opencode/viz-smoke-t62` — passed; `Rendered 1 figure(s)` and `figure_files=2`.
- Subagent: `uv run pytest -m "not slow and not network"` — `193 passed, 4 deselected, 24 warnings in 314.18s (0:05:14)`.
- Subagent: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `61 files already formatted`; `All checks passed!`.
- Orchestrator verification: `uv run python scripts/10_visualize.py --help` — passed; usage printed.
- Orchestrator verification: scratch smoke render with fabricated `results.jsonl` under `/tmp/opencode/pr6-viz-smoke` — `Rendered 1 figure(s)` and `figure_files=2` (`documentation_curve.png`, `documentation_curve.svg`).
- Orchestrator verification / PR gate: `uv run pytest -m "not slow and not network"` — `193 passed, 4 deselected, 24 warnings in 312.51s (0:05:12)`.
- Orchestrator verification / PR gate: `uv run ruff check . && uv run ruff format --check . && uv run ty check` — `All checks passed!`; `61 files already formatted`; `All checks passed!`.
- Orchestrator verification: grep for `import torch|from torch` under `src/binary_classifier/viz` and `scripts/10_visualize.py` — no matches.

DEVIATIONS: none.

RISKS: none.

ARTIFACTS:
- Stage 10 writes `processed/figures/{figure_name}.png` and `.svg`; scratch smoke produced `documentation_curve.png` and `documentation_curve.svg`, matching §5.4.

## Iteration journal

## Iteration 1 — 2026-06-16

Attempted: first-iteration pre-flight for PR-6, branch setup, state seeding, and T6.1.
The PR-2 sentinel was present: `src/binary_classifier/train/sweep.py` exists and references
`learning_curve_results` / `results.jsonl`. Created and switched to branch
`feature/10-viz` from `refactor/harmonize-pipeline` after `git switch feature/10-viz`
failed because the branch did not yet exist. Working-tree state before work contained only
the runner-seeded untracked `.agents/ralph/state/pr-6.md`, which is attributable to this
iteration.

Verified results: T6.1 implemented the pure plotting helpers in `src/binary_classifier/viz/`
and `tests/test_viz.py`. Orchestrator re-ran Tier-1 acceptance:
`uv run pytest -m "not slow and not network" && uv run ruff check . && uv run ruff format --check . && uv run ty check`
passed with `193 passed, 4 deselected, 24 warnings in 304.79s`; ruff check reported
`All checks passed!`; ruff format reported `60 files already formatted`; ty reported
`All checks passed!`. A grep for torch imports in `src/binary_classifier/viz` returned no
matches.

Next action: run T6.2 (`scripts/10_visualize.py`, `.agents/architecture/pipeline.md`,
README viz note), then run its acceptance including the smoke rendering check when the
needed Tier-2 artifacts are available/fabricated.

## Iteration 2 — 2026-06-16

Attempted: T6.2 only. The working tree at launch was on `feature/10-viz` with the
expected untracked T6.1 files and the seeded `pr-6.md` state journal from iteration 1;
no unexplained changes were present. Dispatched one `ralph-implementer` for T6.2.

Verified results: T6.2 added `scripts/10_visualize.py`, a pipeline architecture note,
and the README visualization note. The script renders any available figure inputs into
`figures_dir` as PNG and SVG, logs skips for missing artifacts, and is not wired into
`run_pipeline.py`. Orchestrator re-ran the T6.2 acceptance and PR gate:
`uv run python scripts/10_visualize.py --help` printed usage; a scratch smoke render
with fabricated `/tmp/opencode/pr6-viz-smoke/models/runs/results.jsonl` rendered one
figure and produced `figure_files=2`; `uv run pytest -m "not slow and not network"`
passed with `193 passed, 4 deselected, 24 warnings in 312.51s (0:05:12)`; and
`uv run ruff check . && uv run ruff format --check . && uv run ty check` reported
`All checks passed!`, `61 files already formatted`, and `All checks passed!`.
Torch-import checks under `src/binary_classifier/viz` and `scripts/10_visualize.py`
returned no matches.

Next action: all PR-6 tasks and the PR gate are green. Commit the PR-6 changes and
write `DONE` to `.agents/ralph/state/pr-6.status` for human review.
