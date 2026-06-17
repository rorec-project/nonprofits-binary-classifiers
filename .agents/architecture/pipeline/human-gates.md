# Human Gates (G1–G4)

Four human checkpoints gate the pipeline before it commits GPU work or irreversible operations.

## G1 — Labels gate

Before stages 02, 04, 06, and 07, the pipeline validates that `gold_to_code.csv` has complete `0/1` human labels for every row in the required split (`prompt_dev` for 02, `validation` for 04/06, `test` for 07). If labels are missing, blank, or non-`0/1`, the run exits gracefully (no GPU work wasted).

## G2 — Slate gate

Before stage 03 (full annotation), the pipeline requires a human-confirmed `data/processed/gold/production_slate.json`. If stages 02+03 are requested together and no confirmed slate exists, stage 02 runs (produces `proposed_slate.json`), then the pipeline exits gracefully before stage 03.

## G3 — Test-unlock gate

Before stage 07 (evaluation), the pipeline requires a human-confirmed `data/processed/gold/test_unlock.json` that matches the training checkpoint SHA. Protects the frozen test split from accidental leakage during model iteration.

## G4 — Anchor-labels gate

Before stages 07 and 09 (prevalence), the pipeline requires a fully coded `anchor_to_code.csv` for the anchor sample. Without it, prevalence estimates on LOW-quality rows cannot be validated.

## Related

- [pipeline.md](pipeline.md) — which stages each gate protects
- [../overview.md](../overview.md) — project status
