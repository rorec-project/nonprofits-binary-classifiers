# LazyVim + Python (uv / ruff / ty) — Setup Memo

Personal editor setup for working on this project in LazyVim with the Astral
toolchain. **Optional and out of band:** the pipeline runs from any shell via
`uv run …` — none of this is required to build or run the classifier. Project
lint/type *rules* live in `pyproject.toml`; see
[`python-standards.md`](../.agents/architecture/conventions/python-standards.md).

## Tools

```bash
uv tool install ruff      # linter + formatter
uv tool install ty        # type checker (Astral, beta)
```

(Or let Mason install `ruff`/`ty`. uv-managed venvs have no `pip`, so Mason's
"pip not available" warning is cosmetic — ignore it.)

## LazyVim

1. Enable the Python extra: `:LazyExtras` → `lang.python`.
2. Pick the LSP + linter in `lua/config/options.lua`:

   ```lua
   vim.g.lazyvim_python_lsp = "ty"     -- fast; falls back to "pyright" if needed
   vim.g.lazyvim_python_ruff = "ruff"  -- native ruff LSP, not ruff_lsp
   ```

   LazyVim disables the other Python LSPs automatically once `lsp` is set.

## venv

`uv run …` always uses `.venv` without activation. For an interactive shell,
either `source .venv/bin/activate` or let mise auto-source it (local, gitignored
`mise.toml` with `python.uv_venv_auto = "source"`).

## Common fixes

- **`unresolved-import` for an installed package** → `uv sync`, then `:LspRestart`.
  The venv is stale far more often than the type checker is wrong.
- **Duplicate diagnostics (ruff + ty on the same line)** → in `lua/plugins/python.lua`
  set ruff's `init_options.settings.showSyntaxErrors = false` and let ty own them.

## Jupyter (optional)

Inline plots in notebooks need a terminal that speaks the Kitty graphics protocol
(Ghostty or Kitty) plus `molten-nvim` + `image.nvim`. `jupytext` converts
`.ipynb` ↔ `.py` for editing.
