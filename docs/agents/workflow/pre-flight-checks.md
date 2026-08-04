# Pre-Flight Checks

\*Before outputting code, verify the following.

## Line Length

- [ ] Python: 88 characters (ruff-enforced)?

## Environment & tooling

- [ ] Run through `uv run` (never bare `python`/`ruff`/`ty`)?
- [ ] `uv run ruff check .` and `uv run ty check` clean?
- [ ] `uv run pytest` passes (no new failures)?

## Reproducibility

- [ ] Every stochastic step (sampling, splitting, annotation, training) seeded from the config `SEED`?
- [ ] No new hardcoded seed that bypasses the config?

## Configuration

- [ ] New knobs added to `config/*.yaml` **and** the pydantic model in `binary_classifier/config.py` — not hardcoded in a stage?
- [ ] `entity` / `field` / `label_name` read from config, never pinned to one entity?

## Path Handling

- [ ] Paths built with `pathlib.Path` — prefer `PathRegistry` (`binary_classifier/paths.py`) — not string concatenation?
- [ ] No absolute paths (`C:/Users/...`, `/Users/...`)?

## Data Flow

- [ ] Upstream parquet read via `PathRegistry` (resolves `../NonProfitData/...`), not a hardcoded filename?
- [ ] Outputs written under `data/` or `models/`?
- [ ] `EIN2` (the upstream join key) carried through every artifact?

## Validation

- [ ] `EIN2` uniqueness / disjoint splits / strata coverage checked before merges (mirror the assertions in `scripts/01_build_sample.py`)?
- [ ] Critical data-integrity checks in place; summary stats logged (not `print`)?

## Testing

- [ ] New logic covered by a pytest in `tests/`, run via `uv run pytest`?
- [ ] Critical data-integrity checks still asserted inline (log failures, don't `print`)?

## Code Comments

- [ ] Comments above code, not inline?
- [ ] Business/labeling logic explained, not syntax?

## Final Checks

- [ ] Code follows project conventions ([python-standards](../conventions/python-standards.md))?
- [ ] Ready for manual validation?

**Status:** ✅ READY FOR OUTPUT
