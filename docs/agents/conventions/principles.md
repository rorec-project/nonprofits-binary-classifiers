# Design Principles

How to approach implementation and refactoring decisions. Read before designing or changing a module.

- **Choose the simplest implementation that fully meets the current requirements.** Avoid speculative abstractions, configuration, and indirection.
- **Favor strategic programming and deep modules** over tactical programming and shallow modules.
- **Grow the system in layers.** Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- **Use dependencies already in the project before writing your own implementation or adding packages.** Prefer established, well-maintained libraries and do not reimplement common functionality without a clear reason. Do not assume a library lacks a capability without checking its documentation and types.

## Markdown Prose

Markdown documents (`.md`) are **soft-wrapped**: write prose as one logical paragraph per line, and do **not** hard-wrap at 80 characters. The hard-wrap rules in the language conventions apply to _code only_, not to Markdown. Rationale: unwrapped prose keeps diffs to the actual content change, avoids reformatting churn when a sentence is edited, and the editor's view wraps text for display.

- Put each heading on its own line.
- Put each list item on its own line (long items are fine on one line).
- Keep tables, code blocks, and fenced blocks exactly as authored (tables need one row per line).
- Only join/split a paragraph when the text itself changes; do not reflow pre-existing wrapped Markdown files unless explicitly asked.
