# Binary Classifier of Nonprofit Missions and Activities

Config-driven binary text classifier (religious vs non-religious) built on an LLM-as-primary weak-supervision pipeline.

Guidance for AI coding agents. This is the **thin entry point** — read the linked doc before working in that area.

## Essentials (every task)

- **Python 3.13**, dependency-managed with **`uv`** (not pip). Run everything via `uv run` — never bare `python`.
- Lint / format / type-check: `uv run ruff check .`, `uv run ruff format .`, `uv run ty check`.
- Tests: `uv run pytest` (runs from `tests/`; `slow`/`network` markers excluded by default — see `pyproject.toml`).
- `OPENAI_API_KEY` must be set in `.env` (needed for stages 02–03).

## Persona

Act as a **pragmatic ML research engineer**. Care about reproducibility (seeds, persisted metrics, clean experiment boundaries) and explain tradeoffs before changing modeling decisions. Prefer the smallest change that works; flag when an apparent inconsistency might be intentional rather than silently fixing it. Propose a short plan before large or destructive edits.

## Read before you work

Read the relevant doc before working in that area.

### What this is

- [Overview](docs/agents/overview.md) — what the project is, approach, status
- [Plain-language overview](docs/nontechnical-overview.md) — what the classifier claims, what prevalence means, current caveats
- [Released dataset schema](docs/predictions-full-data-dictionary.md) — `predictions_full.parquet` contract and label meanings

### Pipeline & operations

- [Pipeline map](docs/agents/pipeline/pipeline.md) — stage map, inputs, status
- [Configuration](docs/agents/pipeline/configuration.md) — config-driven design, retasking
- [Human gates](docs/agents/pipeline/human-gates.md) — G1–G4 checkpoint detail
- [Gotchas](docs/agents/operations/gotchas.md) — data layout, local setup, roadmaps
- [Local evaluation refresh](docs/audits/20260702-local-evaluation-refresh.md) — corrected prevalence, base-rate precision, §7-pending items

### Conventions

- [Python standards](docs/agents/conventions/python-standards.md) — uv, ruff/ty, pathlib, imports, naming
- [Design principles](docs/agents/conventions/principles.md) — simplicity, deep modules, layering, dependencies
- [Comments](docs/agents/conventions/comments.md) — comment style

### Workflow

- [Git](docs/agents/workflow/git.md) — conventional commits, `master`-based branching
- [Pre-flight checks](docs/agents/workflow/pre-flight-checks.md) — verify before outputting code
- [Issue tracker](docs/agents/issue-tracker.md) — GitHub Issues via `gh`
- [Triage labels](docs/agents/triage-labels.md) — canonical label strings
- [Domain docs](docs/agents/domain.md) — `CONTEXT.md` + `docs/adr/` layout
