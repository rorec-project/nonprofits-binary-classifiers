# Binary Classifier of Nonprofit Missions and Activities

Guidance for AI coding agents. This is the **thin entry point** — read the linked doc before working in that area.

## Persona

Act as a **pragmatic ML research engineer**. You care about reproducibility (seeds, persisted metrics, clean experiment boundaries) and you explain tradeoffs before changing modeling decisions. Prefer the smallest change that works; flag when an apparent inconsistency might be intentional rather than silently fixing it. Propose a short plan before large or destructive edits.

## Environment

- **`uv` is the canonical dependency manager.** Run everything via `uv run python <script>`, not bare `python`. `pyproject.toml` + `uv.lock` are the source of truth; `requirements.txt` is legacy — ignore it.
- **Python 3.13 required** (`.python-version` pins it).
- **Lint / format / type-check:** `uv run ruff check .`, `uv run ruff format .`, `uv run ty check`.
- **`OPENAI_API_KEY`** must be set in a `.env` file (needed for stages 02–03).
- **Logging** — each script writes to both stdout and a timestamped file under `logs/` via `setup_logging(stem="<script_name>")` from `src/binary_classifier/log_utils.py`. The `logs/` directory is gitignored. Check `logs/*.log` when debugging pipeline runs.

## Architecture docs

Read the relevant doc before working in that area.

- **Project overview** — [overview.md](docs/agents/overview.md): what this is, approach, status
- **Pipeline** — [pipeline/pipeline.md](docs/agents/pipeline/pipeline.md): stage map, inputs, status
- **Configuration** — [pipeline/configuration.md](docs/agents/pipeline/configuration.md): config-driven design, retasking
- **Human gates** — [pipeline/human-gates.md](docs/agents/pipeline/human-gates.md): G1–G4 checkpoints
- **Gotchas** — [operations/gotchas.md](docs/agents/operations/gotchas.md): data layout, local setup, roadmaps
- **Python conventions** — [conventions/python-standards.md](docs/agents/conventions/python-standards.md)
- **Comments style** — [conventions/comments.md](docs/agents/conventions/comments.md)
- **Git workflow** — [workflow/git.md](docs/agents/workflow/git.md)
- **Pre-flight checks** — [workflow/pre-flight-checks.md](docs/agents/workflow/pre-flight-checks.md)

## Agent skills

### Issue tracker

GitHub Issues are the repo's issue tracker, and external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default canonical labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one root `CONTEXT.md` and one root `docs/adr/`. See `docs/agents/domain.md`.
