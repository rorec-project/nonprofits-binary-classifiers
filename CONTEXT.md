# Context

This repo classifies nonprofit mission text as religious or non-religious.

## Terms

- **Mission**: the nonprofit text field being classified.
- **Silver**: LLM-labeled training data.
- **Gold**: human-coded labeled data used for selection, QC, and evaluation.
- **Anchor**: a representative labeled sample used to correct prevalence estimates.
- **Per-organization estimand**: prevalence is defined over raw `EIN2` organizations in the **501C3-charity mission frame**, not just deduplicated mission texts. The qualifier matters: three wider `EIN2` universes exist (see **Panel frame**, **BMF registry**, **BMF-only population**) and prevalence is not defined over them.
- **Panel frame**: the 1.54M `EIN2` organizations in the upstream panel, spanning 501c3 charities, private foundations, and 501CX nonprofits.
- **BMF registry**: the 3.44M `EIN2` organizations in the unified Business Master File — the widest universe, and the only one carrying a name for every organization.
- **Name-only stratum**: organizations in the configured names-arm panel scope that have a name but no mission text. The shipped scope is `501C3 CHARITY`; `names.panel_scope_values` can select other panel classifications in a separate run.
- **BMF-only population**: organizations in the BMF registry that never enter the panel. Distinct from the **name-only stratum**, and the two must not be conflated — they differ sharply in composition.
- **Cross-field transfer**: scoring an existing fine-tuned checkpoint on a different text field of the same organization, with no target-field adaptation. Deliberately **not** called *zero-shot*: the label space is unchanged and the model was task-trained.
- **Names arm**: the cross-field transfer extension classifying organizations from their name. Secondary to the missions pipeline by construction — where the two conflict, missions are the first-class evidence.
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
- Never describe **cross-field transfer** as *zero-shot*; reserve *zero-shot* for prompted models with no task-specific training.
- Name an `EIN2` universe explicitly whenever coverage or prevalence is discussed — "the sample" is ambiguous across four of them.
