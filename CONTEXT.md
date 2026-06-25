# Context

This repo classifies nonprofit mission text as religious or non-religious.

## Terms

- **Mission**: the nonprofit text field being classified.
- **Silver**: LLM-labeled training data.
- **Gold**: human-coded labeled data used for selection, QC, and evaluation.
- **Anchor**: a representative labeled sample used to correct prevalence estimates.
- **G1**: human coding of `gold_to_code.csv`.
- **G2**: confirmation of `production_slate.json`.
- **G3**: confirmation of `test_unlock.json`.
- **G4**: human coding of `anchor_to_code.csv`.

## Naming rules

- Use `missions` for the current task name unless the config is retasked.
- Use `religious` for the positive class unless the glossary is updated.
- Keep `EIN2` as the stable record identifier in pipeline artifacts.
