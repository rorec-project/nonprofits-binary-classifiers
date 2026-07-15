# Gold–Silver Overlap Memo

## Symptom

In the missions run, 109 EIN2s (24.2 % of the 450 gold set) appeared in **both** `silver_manifest.csv` and `gold_manifest.csv`. The overlap wasted ~327 LLM inference calls during stage 03 on EIN2s that were later excluded from `silver_labels.csv` by the stage 04 gold-exclusion guard.

## Root cause

`build_sample()` in `src/binary_classifier/data/sample.py` calls `build_silver_pool()` and `build_gold_set()` on the **same** full dataframe without excluding the silver pool's EIN2s from the gold draw:

```python
silver = build_silver_pool(df, ...)
gold = build_gold_set(df, ...)       # <— re-draws from the same df
```

The two functions independently sample from the full population frame, so they can pick the same EIN2.

## Fix

Add an `exclude_ein2s` parameter to `build_gold_set()` so it can receive the silver pool's EIN2s and filter them out before sampling.

### `build_gold_set()` signature change

```python
def build_gold_set(
    df: pd.DataFrame,
    target_size: int,
    seed: int,
    thresholds: QThresholdsConfig,
    exclude_ein2s: set[str] | None = None,   # <— new
) -> pd.DataFrame:
```

Add a filter early in the function body (after tier assignment, before sampling logic):

```python
if exclude_ein2s:
    n_before = len(df)
    df = df[~df["EIN2"].isin(exclude_ein2s)].copy()
    logger.info("Excluded %d EIN2s already drawn into the silver pool", n_before - len(df))
```

### `build_sample()` caller change

```python
gold = build_gold_set(
    df,
    target_size=cfg.sample_sizes.gold,
    seed=cfg.SEED,
    thresholds=cfg.q_thresholds,
    exclude_ein2s=set(silver["EIN2"]),   # <— new
)
```

## Expected effect

- Zero overlap between silver and gold manifests.
- Gold EIN2s are not sent to the LLM annotator during stage 03 (saving ~target_size × n_prompts inference calls).
- No change to downstream artifacts (silver_labels, anchor, etc.).

## If the gold set target cannot be met

When the gold set's target size cannot be reached after excluding silver EIN2s, the function should raise a clear `ValueError` rather than silently returning fewer rows. This is already the existing behavior when individual strata cannot meet their targets; the same guard applies naturally after the exclusion filter reduces the available frame.

## Applicability

Fix before re-running stage 01 for a new task (e.g. pregnancy centers). Retroactively fixing the missions run is unnecessary — downstream artifacts (silver_labels, anchor) are already correct.
