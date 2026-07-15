# Context

This repo classifies nonprofit mission text as religious or non-religious.

## Terms

- **Mission**: the nonprofit text field being classified.
- **Silver**: LLM-labeled training data.
- **Gold**: human-coded labeled data used for selection, QC, and evaluation.
- **Anchor**: a representative labeled sample used to correct prevalence estimates.
- **Per-organization estimand**: prevalence is defined over raw `EIN2` organizations, not just deduplicated mission texts.
- **Duplicate-text convention**: identical values of the configured text field are treated as the same mission and receive the same classifier score when predictions are expanded back to raw `EIN2` rows.
- **Triple labels**: `pred_label` is the recall-first operating label used for prevalence; `pred_label_maxf1` uses the max-F1 threshold; `pred_label_baserate` uses the deployment/base-rate precision threshold.
- **LOW decomposition**: LOW-tier prevalence is split into classifier-routed rows (`low_via_classifier` → PPI) and pure rule rows (`rule_*` → Rogan-Gladen).
- **One-shot frozen test**: stage 07 may score the real frozen test only once per code-frozen evaluation cycle; any sanctioned reopen happens later under the controlled §7 UCloud procedure.
- **G1**: human coding of `gold_to_code.csv`.
- **G2**: confirmation of `production_slate.json`.
- **G3**: confirmation of `test_unlock.json`.
- **G4**: human coding of `anchor_to_code.csv`.

## Naming rules

- Use `missions` for the current task name unless the config is retasked.
- Use `religious` for the positive class unless the glossary is updated.
- Keep `EIN2` as the stable record identifier in pipeline artifacts.
