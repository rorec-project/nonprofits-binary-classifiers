# Python Code Standards

## Environment

- **Package Manager:** `uv` (strict) — use `uv add`, `uv sync`, `uv run`. `pyproject.toml` + `uv.lock` are the source of truth; `requirements.txt` is legacy.
- **Python:** pinned by `.python-version` (3.13) and provisioned by uv — don't rely on system Python.
- **Linter / Formatter:** Ruff, config in `[tool.ruff]` — 88-char lines, PEP 8. Not "default rules": some are deliberately deferred to ty (see below).
- **Type Checker:** ty (Astral), config in `[tool.ty.*]`.
- Run every tool through the project environment: `uv run ruff …`, `uv run ty …`, `uv run python …` — never bare `python`/`ruff`/`ty`.

## Linting & Type Checking

Ruff and ty are split deliberately so the two tools never double-report the same issue. **Change rules in `pyproject.toml`, not with inline `# noqa` / `# type: ignore`.**

| Tool | Owns                                              | Config block                                                |
| ---- | ------------------------------------------------- | ----------------------------------------------------------- |
| Ruff | Style, formatting, import sorting, lint rules     | `[tool.ruff]`, `[tool.ruff.lint]`                           |
| ty   | Type checking, name resolution, import resolution | `[tool.ty.environment]`, `[tool.ty.rules]`, `[tool.ty.src]` |

- **Division of labor — ruff defers two `F` rules, but only one is a true handoff:**
  - `F821` (undefined name) → genuinely covered by ty's `unresolved-reference`; re-enabling it in ruff would just double-report. Leave it to ty.
- **ty is tuned for gradual typing on an ML codebase:** `unresolved-import` and `possibly-unresolved-reference` are `warn` (not `error`); `division-by-zero` and `index-out-of-bounds` are `ignore` (too noisy on tensor/array code). Tighten rule by rule as typing coverage grows — flag the tradeoff before loosening further.
- **Scope:** ty checks `src/**`, `scripts/**`, and top-level `*.py`. Both ruff and ty exclude `.agents/`, the archived legacy code, `tests/`, and notebooks; ty respects `.gitignore`. The exact patterns live in `[tool.ruff]` and `[tool.ty.src]` — treat `pyproject.toml` as the source of truth (keep the two legacy-path excludes in sync with where the legacy code actually lives).
- **Stale-venv first:** if an installed package reports `unresolved-import`, run `uv sync` before touching config — the venv, not the type checker, is usually stale.

## Logging

Pipeline scripts write to both stdout and a timestamped file under `logs/` via `setup_logging(stem="<script_name>")` from `src/binary_classifier/log_utils.py`. The `logs/` directory is gitignored — check `logs/*.log` when debugging pipeline runs. Use `logging`, not `print()`/`breakpoint()`.

## Path Handling

**Use `pathlib.Path` exclusively.** Never string concatenation.

In pipeline code, prefer the resolved paths on `PathRegistry` (`binary_classifier/paths.py`) over building paths by hand.

```python
from pathlib import Path

# Correct
config_path = Path("config/religious_missions.yaml")
silver_manifest = Path("data/interim/manifests/silver_manifest.csv")

# Wrong
silver_manifest = "data/interim/manifests/" + "silver_manifest.csv"
```

## Documentation

- **Docstrings:** Google or NumPy style (the package uses Google style — match it)
- **Purpose:** Explain modeling/labeling intent, data provenance, and the role of each stage
- **Scope:** Document what and why, not syntax

## Imports

Order: Standard library → Third-party → Local

```python
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from binary_classifier.config import load_config
from binary_classifier.paths import PathRegistry
```

## Naming

| Type                | Convention            |
| ------------------- | --------------------- |
| Variables/Functions | `snake_case`          |
| Classes             | `PascalCase`          |
| Constants           | `UPPER_SNAKE_CASE`    |
| Private             | `_leading_underscore` |

## Commands

```bash
# Sync the environment from the lockfile
uv sync

# Add dependencies (updates pyproject.toml + uv.lock)
uv add <package>
uv add --dev <package>

# Run code in the project environment
uv run python scripts/run_pipeline.py

# Lint + format (rules live in pyproject.toml)
uv run ruff check .          # add --fix to apply safe auto-fixes
uv run ruff format .

# Type check
uv run ty check
```
