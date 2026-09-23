---
created: 2026-07-05
---

# PRD: Simplify Stage-10 Visualization Architecture

## Problem

Stage 10 has accumulated too much implementation logic in `scripts/10_visualize.py`.
The script is no longer a thin pipeline entry point: it performs orchestration,
artifact loading, population-language transformations, keyness sensitivity setup,
silver-text fallback joins, and output writing. This makes the visualization stage
harder to test, harder to reuse from package code, and easier to break when adding
new diagnostics.

The visualization module also has several pockets of avoidable ceremony. Silver
wordclouds are exposed through many near-identical wrapper functions, population
diagnostic outputs are spread across call sites rather than declared centrally, and
generic JSON extraction is broader than the stage currently needs. At the same time,
some complexity is intentional and should be preserved: publication-grade vector and
selectable wordcloud output, probability-weighted population keyness, multi-format
plot output, and the current population diagnostics surface unless it is confirmed
to be non-contractual.

## Goals

- Restore `scripts/10_visualize.py` to a thin wrapper that parses config, sets up
  logging, builds `PathRegistry`, and calls package-level stage code.
- Move stage orchestration and helper logic into package modules under
  `src/binary_classifier/viz/` so tests can target stable public helpers instead of
  private script internals.
- Preserve all current artifact filenames and behavior unless explicitly called out
  as deferred or requiring confirmation.
- Reduce repeated silver-wordcloud wrapper functions by replacing them with a small
  declarative spec loop over weighting, n-gram ranges, and class labels.
- Centralize population diagnostic output definitions in a spec table, while keeping
  the current output count for now.
- Move population-language and population-keyness construction into package code,
  including probability-weighted keyness, sensitivity term selection, and min-DF
  policy.
- Narrow overly generic JSON extraction after schemas stabilize, deleting unused
  keys such as `_PR_POINT_KEYS` if confirmed unused.
- Keep tests focused on public package behavior and critical output guarantees.

## Non-Goals

- Do not simplify wordcloud SVG/PDF rendering back to raster-in-vector output.
- Do not remove or weaken probability-weighted keyness.
- Do not reduce the number of population diagnostics unless the artifact contract is
  reviewed and confirms those outputs are non-contractual.
- Do not weaken tests that protect vector wordcloud selectability and absence of
  embedded raster images.
- Do not redesign the whole visualization API, configuration model, or pipeline
  artifact contract in this refactor.
- Do not change modeling semantics, thresholds, labels, or weighting policies except
  to relocate existing logic.

## Proposed Design

### Stage Entrypoint

Create `src/binary_classifier/viz/stage10.py` with a public function such as
`run_visualization(cfg, registry, logger)` or the smallest signature that matches the
existing call pattern. Move `run_visualization` and script-local stage helpers there.
Keep `scripts/10_visualize.py` responsible only for CLI/config parsing, logging setup,
`PathRegistry` construction, and calling the package function.

### Wordcloud Outputs

Move `_save_wordcloud_outputs` out of the script into package code, preferably
`src/binary_classifier/viz/wordclouds.py` if it naturally belongs beside renderer
logic, or a small output helper if that keeps responsibilities cleaner. Preserve the
renderer's vector/selectable SVG/PDF behavior and current PNG/SVG/PDF output naming.

Replace the 12 silver-wordcloud wrapper functions with a spec table or generator loop
that declares:

- class label or target subset
- weighting mode
- n-gram range
- output filename stem

The spec must preserve existing filenames so downstream docs, audits, and users do
not need migration steps.

### Population Language

Create `src/binary_classifier/viz/population_language.py` or an equivalent package
module for population-language transformations. Move the following logic out of the
script:

- `_population_language_frame`
- `_population_keyness_frame`
- probability-weighted population keyness construction
- keyness sensitivity term selection
- min-DF policy

Keep `compute_keyness_frame` behavior for hard labels and probability weights. The
new module should expose a small public surface that the stage runner can call and
tests can exercise without importing script-private functions.

### Population Diagnostics

Keep the current population diagnostics output surface initially. Centralize output
definitions in a declarative population-diagnostic spec table that records each
diagnostic's name, primary input, output filename stem, and whether it is considered
primary or extended.

Document primary outputs in code or adjacent docs. Defer any reduction of extended
outputs until the artifact contract is reviewed. If later confirmed non-contractual,
extended diagnostics can become opt-in without affecting primary outputs.

### JSON Extraction

Delete unused JSON extraction constants such as `_PR_POINT_KEYS` when verified unused.
Avoid recursive arbitrary JSON search for known stage artifacts once schemas are
stable. Prefer explicit known paths to reduce the chance of accepting the wrong
payload shape. This should be a small hardening pass after the stage code has moved,
not a prerequisite for the main extraction.

### Tests

After moving stage logic into package modules, update tests to import package-level
public helpers instead of private symbols from `scripts/10_visualize.py`. Keep focused
smoke tests for vector wordcloud output that verify no raster image is embedded and
text remains selectable, without over-specifying renderer internals.

## Implementation Plan

### Phase 1: Extract Stage Runner

- Add `src/binary_classifier/viz/stage10.py`.
- Move `run_visualization` and direct helper logic from `scripts/10_visualize.py` into
  the new package module with minimal behavior changes.
- Leave `scripts/10_visualize.py` as a thin CLI wrapper.
- Update imports and the smallest necessary tests.
- Verify with `uv run pytest tests/test_viz.py -q`.

### Phase 2: Move Wordcloud Output Helpers

- Move `_save_wordcloud_outputs` into package code.
- Replace silver-wordcloud wrapper functions with a spec loop while preserving output
  filenames.
- Keep renderer-perfect SVG/PDF behavior unchanged.
- Keep or add focused vector-output smoke coverage.

### Phase 3: Extract Population Language Logic

- Add `population_language.py` or equivalent.
- Move population language frames, probability keyness, sensitivity selection, and
  min-DF policy out of the stage runner.
- Expose package-level public helpers for tests.
- Confirm hard-label and probability-weighted `compute_keyness_frame` paths remain
  covered.

### Phase 4: Centralize Diagnostic Specs

- Introduce a population-diagnostic spec table.
- Route existing diagnostic output generation through the spec table.
- Mark outputs as primary vs extended, but keep all existing outputs emitted by
  default.
- Document any candidate opt-in-only extended outputs for later artifact-contract
  review.

### Phase 5: Harden JSON Extraction and Tests

- Remove verified-unused extraction keys such as `_PR_POINT_KEYS`.
- Replace broad recursive JSON search with explicit known paths where schemas are
  stable.
- Update tests away from private script imports.
- Run full verification.

## Acceptance Criteria

- `scripts/10_visualize.py` is a thin wrapper that performs only CLI/config parsing,
  logging setup, `PathRegistry` construction, and package-stage invocation.
- Stage-10 orchestration lives in package code under `src/binary_classifier/viz/`.
- Wordcloud output saving lives in package code, not in `scripts/10_visualize.py`.
- Existing wordcloud output filenames are preserved.
- Silver-wordcloud generation is declared through a compact spec loop rather than 12
  near-identical wrapper functions.
- Population-language and population-keyness logic lives in package code and still
  supports probability-weighted keyness.
- Population diagnostic definitions are centralized, and existing diagnostics remain
  emitted by default.
- Tests import package-level public helpers rather than private script functions.
- Vector wordcloud tests still protect selectable text and no raster-in-vector
  regression.
- Verification commands pass:
  - `uv run pytest tests/test_viz.py -q`
  - `uv run ruff check .`
  - `uv run ty check`
- A stage-10 smoke run completes against the expected local fixture or current
  pipeline artifacts without changing the frozen test artifact. Use the repository's
  current stage invocation pattern, for example `uv run python scripts/10_visualize.py`
  with the appropriate config arguments for the active task.

## Risks

- Moving orchestration can accidentally change artifact paths or filenames. Mitigate
  by preserving stems in tests and reviewing generated output lists before and after.
- Population diagnostics may already be part of an implicit artifact contract.
  Mitigate by keeping all outputs by default and only labeling primary vs extended.
- Tightening JSON extraction too early may break legitimate historical payloads.
  Mitigate by deferring explicit-path extraction until schemas are confirmed stable.
- Tests that currently inspect SVG internals may become brittle during the move.
  Mitigate by testing user-visible guarantees: vector output, no embedded raster, and
  selectable text.
- Refactoring the script and population logic in one large patch could obscure
  behavior changes. Mitigate with the staged plan above and small follow-up issues.

## Open Questions

- Which population diagnostic outputs are contractual for released artifacts, docs,
  or downstream users?
- What exact stage-10 smoke command and config should be considered canonical for
  local verification?
- Are any historical JSON payload shapes still required, or can extraction target only
  current known schemas?
- Should the population-diagnostic primary vs extended classification be documented
  in user-facing docs or only in code until an artifact-contract review is complete?
