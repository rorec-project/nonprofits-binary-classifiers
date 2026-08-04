# Design Principles

How to approach implementation and refactoring decisions. Read before designing or changing a module.

- **Choose the simplest implementation that fully meets the current requirements.** Avoid speculative abstractions, configuration, and indirection.
- **Favor strategic programming and deep modules** over tactical programming and shallow modules.
- **Grow the system in layers.** Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- **Use dependencies already in the project before writing your own implementation or adding packages.** Prefer established, well-maintained libraries and do not reimplement common functionality without a clear reason. Do not assume a library lacks a capability without checking its documentation and types.