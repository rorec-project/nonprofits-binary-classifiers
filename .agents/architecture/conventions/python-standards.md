# Python Code Standards

## Environment

- **Package Manager:** `uv` (strict) — use `uv init`, `uv add`, `uv run`
- **Linter:** Ruff (standard rules), PEP8 style
- **Line Length:** 88 characters

## Path Handling

**Use `pathlib.Path` exclusively.** Never string concatenation.

```python
from pathlib import Path

# Correct
raw_data = Path("data/raw/mortality.csv")
processed_data = Path("data/processed/cleaned.csv")

# Wrong
raw_data = "data/raw/mortality.csv"
```

## Documentation

- **Docstrings:** Google or NumPy style
- **Purpose:** Explain business logic and economic intent
- **Scope:** Document what and why, not syntax

## Imports

Order: Standard library → Third-party → Local

```python
import os
from pathlib import Path

import pandas as pd
import numpy as np

from src.functions import my_function
```

## Naming

| Type | Convention |
|------|-----------|
| Variables/Functions | `snake_case` |
| Classes | `PascalCase` |
| Constants | `UPPER_SNAKE_CASE` |
| Private | `_leading_underscore` |

## Commands

```bash
# Initialize
uv init my_project

# Add dependencies
uv add pandas numpy
uv add --dev ruff pytest

# Run
uv run python src/script.py

# Sync environment
uv sync

# Lint
uv run ruff check src/
uv run ruff format src/
```


