# Pre-Flight Checks

*Before outputting code, verify:*

## Line Length

- [ ] Python: 88 characters?
- [ ] R: 80 characters?

## Environment Manager

- [ ] Python: `uv` (strict)?
- [ ] R: `renv`?

## Random Operations

- [ ] Random seed set (if stochastic)?

## Path Handling

- [ ] All paths relative (`pathlib.Path` or `here()`)?
- [ ] No absolute paths like `C:/Users/...`?

## Data Flow

- [ ] Reading from `data/raw`?
- [ ] Writing to `data/processed` or `out/`?
- [ ] Symlinks preserved (not recreated)?

## Testing Structure

- [ ] No `tests/` folders created (unless requested)?

## Code Comments

- [ ] Comments above code, not inline?
- [ ] Business logic explained, not syntax?

## R-Specific

- [ ] Native pipe `|>` in R (not `%>%`)?
- [ ] Explicit namespace calls in R functions (e.g., `dplyr::`)?

## Validation

- [ ] Unique keys validated before merges?
- [ ] Critical data integrity checks in place?
- [ ] Summary statistics printed for verification?

## Final Checks

- [ ] All checklist items completed
- [ ] Code follows project conventions
- [ ] Code is ready for manual validation

**Status:** ✅ READY FOR OUTPUT
