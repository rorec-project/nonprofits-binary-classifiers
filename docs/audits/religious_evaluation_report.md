# Religious Mission Classifier: Evaluation, Inference, Prevalence, and Visualization Audit

> **Superseded for current operator guidance.** Read [20260702-local-evaluation-refresh.md](20260702-local-evaluation-refresh.md) for the Wave-6 documentation refresh. This older report remains as the historical audit of the first UCloud run, but it predates the LOW-tier decomposition fix, base-rate precision label, `predictions_full.parquet`, stage-10 orchestration, and `run_manifest.json`.

Date: 2026-07-02

Scope: This report reviews the completed UCloud run of stages 07-09 and the script-only stage 10 visualization run for the religious mission binary classifier.

Primary run artifacts:

| Area | Artifact |
|---|---|
| Pipeline log | `logs/pipeline_20260702_115243.log` |
| Visualization log | `logs/10_visualize_20260702_120516.log` |
| Evaluation | `data/processed/evaluation/test_evaluation.json` |
| Calibration | `data/processed/evaluation/calibrator.json` |
| Rule validation | `data/processed/evaluation/rule_validation.json` |
| Anchor OOF scores | `data/processed/evaluation/anchor_oof_scores.parquet` |
| Predictions | `data/processed/predictions/predictions.parquet` |
| Prediction shards | `data/processed/predictions/shards/*.parquet` |
| Monitor scores | `data/processed/predictions/monitor_scores.json` |
| Prevalence report | `data/processed/prevalence/prevalence_report.json` |
| NTEE prevalence | `data/processed/prevalence/prevalence_by_ntee.csv` |
| Figures | `data/processed/figures/*.{png,svg,pdf}` |

## Executive Summary

The run completed successfully for stages 07, 08, 09, and 10, with two expected stage-10 skips and one important visualization defect.

The classifier passes the configured evaluation gates. Its frozen-test performance is strong overall: F1 `0.894`, recall `0.987`, precision `0.817`, MCC `0.809`, ROC-AUC `0.949`, PR-AUC `0.901`, and ECE `0.007`. The main caution is that PR-AUC only narrowly exceeds the configured `0.90` acceptance threshold, and the chosen operating threshold is low (`0.0576925`), intentionally trading false positives for very high recall.

Inference produced classifications for the deduplicated mission-text corpus: `531,660` prediction rows. It did not classify every raw `EIN2` row from the raw corpus because exact duplicate mission texts are dropped before inference. The raw corpus contains `560,354` rows, so `28,694` raw EINs are absent from `predictions.parquet` due to duplicate mission text removal.

The prevalence estimate over the deduplicated prediction corpus is `14.45%`, with 95% CI `12.62%` to `16.28%`. The estimate combines HIGH/MEDIUM weighted PPI (`13.43%`) with LOW-tier Rogan-Gladen correction (`18.36%`). LOW-tier uncertainty is material because rule sensitivity is estimated from a small validation sample.

Stage 10 rendered six figure groups successfully, but skipped the canary drift plot and precision-recall curve. The precision-recall curve skip is caused by a code contract/persistence gap: stage 07 computes PR curve points during threshold selection but does not persist them where stage 10 expects them. The `production_annotation_summary` figure was rendered but is effectively unusable because its height was derived from raw row count rather than plotted aggregate rows.

## Stage Completion Status

| Stage | Status | Evidence |
|---|---|---|
| 07 evaluation | Completed | `logs/pipeline_20260702_115243.log:2-16` |
| 08 inference | Completed | `logs/pipeline_20260702_115243.log:16-40` |
| 09 prevalence | Completed | `logs/pipeline_20260702_115243.log:41-82` |
| 10 visualization | Completed with skips | `logs/10_visualize_20260702_120516.log:348` |

Stage 10 rendered six figure groups:

| Figure | Status |
|---|---|
| `bakeoff_summary` | Rendered |
| `production_annotation_summary` | Rendered, but layout is defective |
| `documentation_curve` | Rendered |
| `reliability_diagram` | Rendered |
| `prevalence_forest` | Rendered |
| `ngram_log_odds` | Rendered |
| `precision_recall_curve` | Skipped, missing serialized PR points |
| `canary_drift` | Skipped, missing canary audit artifact |

## Question Group 1: Model Evaluation and Technical Visual Evidence

### Does the model have a good evaluation?

Yes. The model has a good evaluation for the configured pipeline gate. It passes all acceptance checks and has strong predictive performance on the frozen test split.

Acceptance gates from `config/religious_missions.yaml`:

| Gate | Observed | Threshold | Result |
|---|---:|---:|---|
| PR-AUC | `0.9013902009153695` | `>= 0.90` | Pass |
| Minority F1 CI lower | `0.839772184707728` | `>= 0.70` | Pass |
| ECE | `0.007083317363206847` | `<= 0.05` | Pass |

Frozen-test metrics from `data/processed/evaluation/test_evaluation.json`:

| Metric | Value |
|---|---:|
| Test rows | `175` |
| True positives | `76` |
| False positives | `17` |
| True negatives | `81` |
| False negatives | `1` |
| Precision | `0.8172043010752689` |
| Recall | `0.987012987012987` |
| F1 | `0.8941176470588236` |
| PR-AUC | `0.9013902009153695` |
| ROC-AUC | `0.9491121123774185` |
| MCC | `0.8092651388884946` |
| Balanced accuracy | `0.9067717996289425` |
| Cohen's kappa | `0.7958257713248639` |
| Krippendorff alpha | `0.7941176470588236` |

Bootstrap confidence intervals:

| Quantity | Lower | Upper |
|---|---:|---:|
| Accuracy | `0.8514285714285714` | `0.9371428571428572` |
| Minority F1 | `0.839772184707728` | `0.9381475117729413` |

Interpretation: The model is highly recall-oriented. It misses only one positive case in the frozen test split, but the operating point produces 17 false positives. This behavior is consistent with the configured precision-floor threshold policy.

### Main evaluation caveats

| Caveat | Why it matters |
|---|---|
| PR-AUC barely passes | Observed PR-AUC is `0.90139`, only `0.00139` above the `0.90` floor. |
| Low threshold | The selected calibrated threshold is `0.0576925`, far below the max-F1 threshold `0.6082767`. This explains high recall and nontrivial false positives. |
| Sparse calibration middle bins | Most calibration mass is in the lowest and highest bins; mid-probability reliability is weakly estimated. |
| Small subgroup cells | NTEE subgroup test cells are often `n=3-11`, useful for diagnostics but not stable subgroup claims. |
| LOW rule uncertainty | LOW-tier rule validation has strong point estimates but wide confidence intervals from small effective samples. |

Calibration summary:

| Quantity | Value |
|---|---:|
| Method | Platt |
| Params | `a=0.4167116424175239`, `b=0.8410947175108892` |
| Threshold policy | `precision_floor` |
| Precision floor | `0.8` |
| Selected threshold | `0.05769250483141822` |
| Max-F1 threshold | `0.6082766564370073` |
| Brier | `0.030050242073401165` |
| Log loss | `0.13866210791291844` |
| ECE | `0.007083317363206847` |

Rule validation summary for LOW cells:

| Quantity | Value |
|---|---:|
| LOW anchor rows | `149` |
| Rule applied | `67` |
| Abstained | `82` |
| True positives | `11` |
| False positives | `0` |
| True negatives | `54` |
| False negatives | `2` |
| Precision | `1.0`, CI `0.7411670330319683` to `1.0` |
| Sensitivity | `0.8461538461538461`, CI `0.5776536898051745` to `0.9567418216419186` |
| Specificity | `1.0`, CI `0.9335864119091036` to `1.0` |

Important interpretation: Rule metrics are conditional on covered LOW rows only. The `82` abstained LOW anchor rows are routed differently and are not counted in the rule-applied confusion matrix.

### Visualizations to add for a technical audience

Recommended high-value plots that can mostly be built from existing artifacts:

| Visualization | Source artifacts | Why it persuades |
|---|---|---|
| Confusion matrix with rates | `test_evaluation.json` | Makes the operating point legible: 76 TP, 17 FP, 81 TN, 1 FN. |
| Precision-recall curve with selected threshold | `anchor_oof_scores.parquet`, `calibrator.json`; or persisted test PR points | Shows the precision-recall frontier and justifies the low threshold. |
| Reliability diagram with bin counts | `test_evaluation.json`, `anchor_oof_scores.parquet` | Demonstrates calibration while showing sparse mid-probability bins. |
| Score histogram by label and tier | `anchor_oof_scores.parquet` | Shows separation, ambiguous cases, and tier-specific behavior. |
| Rule validation interval plot | `rule_validation.json` | Makes LOW-tier sensitivity, specificity, precision, and uncertainty auditable. |
| Prevalence decomposition plot | `prevalence_report.json` | Separates HIGH/MEDIUM PPI, LOW Rogan-Gladen, and composite prevalence. |
| Per-NTEE prevalence forest | `prevalence_by_ntee.csv` | Shows substantive heterogeneity and uncertainty across organizational domains. |
| Subgroup performance dot plot | `test_evaluation.json` | Shows robustness by NTEE and text length, while surfacing small-cell limits. |
| N-gram log-odds bars | Existing stage 10 output/input | Provides auditable language diagnostics instead of word clouds. |

Recommended SOTA-style computational social science visualizations that may require additional saved data:

| Visualization | Additional data needed | Why it matters |
|---|---|---|
| Frozen-test PR and ROC curves | Persist test `y_true`, calibrated probabilities, and threshold grid | PR-AUC barely passes; a curve shows whether performance is robust across thresholds. |
| Error audit panel | Save FP/FN predictions joined to mission text | Connects statistical performance to construct validity. |
| Calibration by subgroup/tier | Larger subgroup probability/label data | Checks whether global calibration hides local miscalibration. |
| Quantification sensitivity plot | Save naive classify-and-count, PPI, weighted PPI, EMQ, Rogan-Gladen estimates | Shows prevalence is not an artifact of one estimator. |
| Specification/tornado plot | Sensitivity scenarios for thresholds, LOW rule parameters, design weights | Shows robustness of the substantive prevalence estimate. |
| Model comparison by seed/encoder | Per-model/per-seed metrics from stage 06/07 | Shows the selected model is stable, not a lucky single run. |
| Label uncertainty visualization | Vote shares or annotator agreement joined to model predictions | Connects model errors to upstream label ambiguity. |

## Question Group 2: Inference Probabilities and Classification Coverage

### Do we preserve single probabilities per EIN2-mission from inference?

Not for every raw `EIN2` row. The inference artifact preserves one prediction row per deduplicated mission text, with one retained `EIN2` attached to that mission text.

The data loader drops exact duplicate mission texts before inference. The relevant behavior is in `src/binary_classifier/data/load.py`, where the frame is deduplicated by the configured text field before being renamed to `mission_text`.

Coverage summary:

| Frame | Rows | Unique EIN2 | Duplicate EIN2 |
|---|---:|---:|---:|
| Raw `missions_cross_section.parquet` | `560,354` | `560,354` | `0` |
| Unique raw mission texts | `531,660` | - | - |
| Prediction artifact | `531,660` | `531,660` | `0` |
| Raw EIN2 missing from predictions | `28,694` | - | - |

Interpretation: All deduplicated inference rows are classified. Not all raw EIN2 are classified, because exact duplicate mission texts are removed before inference. If the downstream estimand requires one classification per raw EIN2, the pipeline needs either to avoid deduplication for stage 08 or to map predictions back from deduplicated mission text to all raw EIN2 rows.

### Can deduplicated labels be merged back to the original EIN2 data?

Yes, but not by joining `predictions.parquet` directly to the raw data on `EIN2`. A direct `EIN2` join only recovers labels for the `531,660` retained deduplicated rows. The `28,694` raw EIN2 rows with duplicate mission texts remain unmatched because their EIN2 values were dropped before inference.

The correct merge-back strategy is to use mission text as the bridge:

1. Join `predictions.parquet` to the raw mission file on the retained prediction `EIN2` to recover the exact `LONGEST_MISSION` text for each prediction.
2. Build a mission-text-to-prediction mapping from that joined frame.
3. Join the mapping back to the full raw mission file on `LONGEST_MISSION`.
4. Preserve the raw `EIN2` as the final organization key.

Conceptual implementation:

```python
import pandas as pd

raw = pd.read_parquet("data/raw/missions_cross_section.parquet")
pred = pd.read_parquet("data/processed/predictions/predictions.parquet")

pred_with_text = pred.merge(
    raw[["EIN2", "LONGEST_MISSION"]],
    on="EIN2",
    how="left",
    validate="one_to_one",
)

prediction_by_text = pred_with_text.drop(columns=["EIN2"])

raw_with_labels = raw.merge(
    prediction_by_text,
    on="LONGEST_MISSION",
    how="left",
    validate="many_to_one",
)
```

This gives every raw EIN2 with an identical mission text the same label and probability as the deduplicated row that was actually scored.

Important caveats:

| Caveat | Implication |
|---|---|
| Identical text assumption | The merge-back assumes identical mission text should receive identical religious classification. This is reasonable for text-only classification, but it should be stated explicitly. |
| Probabilities are copied | Duplicate mission rows are not independently scored; they inherit the deduplicated row's probability and label. |
| Rule rows still have null probabilities | Deterministic `rule_short_negative` and `rule_strong_positive` rows have final labels but no model probabilities. |
| Organization-level prevalence changes | If the estimand is prevalence over all raw EIN2 organizations, stage 09 should use either expanded raw-EIN2 predictions or duplicate-count weights, not the deduplicated prediction corpus alone. |

Recommended engineering follow-up: Add a stage-08 output such as `data/processed/predictions/predictions_full.parquet` that expands deduplicated predictions back to every raw EIN2 row while preserving the existing deduplicated `predictions.parquet` for efficient model inference and reproducibility.

### Are all EIN2 classified as religious or not?

All EIN2 present in `predictions.parquet` receive a final binary `pred_label`. The current prediction artifact classifies the deduplicated corpus, not the full raw corpus.

Decision counts:

| Decision source | HIGH | MEDIUM | LOW | Total |
|---|---:|---:|---:|---:|
| `classifier` | `261,143` | `160,343` | `0` | `421,486` |
| `low_via_classifier` | `0` | `0` | `59,704` | `59,704` |
| `rule_short_negative` | `0` | `0` | `41,272` | `41,272` |
| `rule_strong_positive` | `0` | `0` | `9,198` | `9,198` |
| Total | `261,143` | `160,343` | `110,174` | `531,660` |

Probability availability:

| Decision source | Probability status |
|---|---|
| `classifier` | `prob_raw` and `prob_calibrated` present |
| `low_via_classifier` | `prob_raw` and `prob_calibrated` present |
| `rule_short_negative` | probabilities null |
| `rule_strong_positive` | probabilities null |

Null probability counts:

| Quantity | Count |
|---|---:|
| `prob_raw` null | `50,470` |
| `prob_calibrated` null | `50,470` |

Those nulls are expected for deterministic rule-routed LOW rows.

### What is the probability threshold for religious classification?

There is one fixed calibrated probability threshold for classifier-routed rows:

```text
threshold = 0.05769250483141822
```

This threshold appears in `data/processed/evaluation/calibrator.json`, `data/processed/predictions/monitor_scores.json`, and the prediction parquet metadata columns.

Classifier-routed rows are classified religious when:

```text
prob_calibrated >= 0.05769250483141822
```

There is no meaningful average threshold across all rows because deterministic rule rows bypass probabilities:

| Rule source | Behavior |
|---|---|
| `rule_strong_positive` | `pred_label = 1`, probabilities null |
| `rule_short_negative` | `pred_label = 0`, probabilities null |

## Question Group 3: Prevalence Results and Interpretation

Stage 09 completed successfully and produced the composite prevalence estimate over the deduplicated prediction corpus.

Population counts:

| Tier | Count | Share |
|---|---:|---:|
| HIGH | `261,143` | `49.12%` |
| MEDIUM | `160,343` | `30.16%` |
| HIGH/MEDIUM | `421,486` | `79.28%` |
| LOW | `110,174` | `20.72%` |
| Total | `531,660` | `100.00%` |

Main prevalence estimates:

| Scope | Estimate | 95% CI | Notes |
|---|---:|---:|---|
| Composite | `14.45%` | `12.62%` to `16.28%` | Full prediction corpus |
| HIGH/MEDIUM weighted PPI | `13.43%` | `11.57%` to `15.30%` | Primary HIGH/MEDIUM estimator |
| HIGH/MEDIUM unweighted PPI | `12.93%` | `10.99%` to `14.88%` | Sensitivity check |
| LOW Rogan-Gladen | `18.36%` | `13.14%` to `23.58%` | Corrected LOW estimate |
| LOW observed rule prevalence | `15.54%` | Not directly reported | Raw LOW rule-label rate |
| EMQ cross-check | `16.48%` | No CI reported | HIGH/MEDIUM only |

Important distinction: Model evaluation and prevalence estimation are separate claims. The classifier's frozen-test metrics answer how well the classifier predicts held-out labels. The prevalence report estimates a population share, combining model predictions, anchor labels, design weights, LOW rules, and uncertainty corrections.

### Per-NTEE prevalence caveats

The per-NTEE CSV has 27 groups, with 18 unsuppressed `ppi_rg_composite` rows and 9 suppressed `emq_fallback` rows. Suppression follows `prevalence.ntee_min_n: 10` based on total anchor rows.

Notable estimates:

| NTEE | Estimate | CI | Status |
|---|---:|---:|---|
| X | `83.92%` | `76.75%` to `91.09%` | Unsuppressed |
| P | `20.01%` | `14.00%` to `26.02%` | Unsuppressed |
| Q | `20.48%` | `2.90%` to `38.07%` | Unsuppressed, wide CI |
| S | `12.51%` | `0.00%` to `27.20%` | Unsuppressed, wide CI |

Suppressed fallback rows with missing CIs include `?`, `H`, `J`, `K`, `R`, `U`, `V`, `Y`, and `Z`. Several suppressed fallback point estimates around `77%` to `83%` look unstable and should be treated as diagnostics rather than publishable subgroup estimates.

Follow-up: Consider suppressing fallback point estimates entirely or labeling them as `not estimated` when anchor support is below threshold.

## Question Group 4: Stage 10 Figure Fixes and Missing Inputs

### How can we fix `production_annotation_summary`?

The bug is in `scripts/10_visualize.py::_maybe_render_production_summary`.

Current behavior sizes the figure using raw input row count:

```python
height=max(4.0, 0.35 * len(frame) + 1.5)
```

For the current artifact, `len(frame)` is about `20,341`, so the output PNG became approximately `2080 x 6,407,845` pixels and about `103 MB`. This is a layout failure, not a data failure.

Smallest recommended fix: size the figure by plotted aggregate rows, not raw annotation rows.

Recommended target:

```python
plot_rows = _production_summary_plot_rows(frame)
figsize=figure_size(width=PAGE_WIDTH, height=max(4.0, 0.35 * plot_rows + 1.5))
```

The helper should count the plotted source categories plus aggregate diagnostics. For the current figure, this would be about `3` sources plus `3` diagnostics, so the height would remain near the 4-inch floor.

Implementation location:

| File | Function |
|---|---|
| `scripts/10_visualize.py` | `_maybe_render_production_summary` |

### What is missing to produce `precision_recall_curve`? Is it a bug?

This is a code contract/persistence bug between stage 07 and stage 10.

What exists:

| Component | Status |
|---|---|
| Stage 07 threshold selection | Computes PR curve points during threshold selection |
| Stage 10 visualization | Looks for keys such as `pr_curve_points`, `precision_recall_curve`, `precision_recall_points`, `pr_points`, or `points` |
| Plotting function | Can render PR points with `precision` and `recall` fields |

What is missing:

| Artifact | Missing payload |
|---|---|
| `data/processed/evaluation/calibrator.json` | `pr_curve_points` |
| `data/processed/evaluation/test_evaluation.json` | Serialized PR curve points |

Stage 10 logged:

```text
No precision-recall curve points found in test_evaluation.json
No precision-recall curve points found in calibrator.json
Skipping precision-recall curve; no usable point payloads found.
```

Smallest fix: Persist `threshold_report["pr_curve_points"]` in the calibrator payload from stage 07.

Recommended target:

```python
"pr_curve_points": threshold_report["pr_curve_points"]
```

Implementation location:

| File | Function |
|---|---|
| `src/binary_classifier/evaluation/evaluate.py` | `_calibrator_payload` |

Caveat: This would produce the anchor OOF threshold-selection PR curve, not necessarily the frozen-test PR curve. If the desired figure is frozen-test PR, stage 07 should compute and persist test-set PR points from frozen-test labels and calibrated probabilities.

### Should we run canary drift in the pipeline?

Recommendation: Do not run canary drift as a default core stage. Run it as an explicit monitoring operation.

Rationale:

| Reason | Explanation |
|---|---|
| Purpose | Canary drift monitors LLM/provider snapshot drift, not classifier evaluation or prevalence estimation. |
| Cost | It may incur API or inference cost. |
| Human-gate semantics | It should not feed prompt tuning or production label freezing. |
| Timing | It is most useful before/after provider changes, scheduled monitoring, or before reusing LLM annotation. |

The missing artifact for stage 10 is:

```text
data/interim/canary_drift_audit.jsonl
```

Existing command to produce it:

```bash
uv run python scripts/03_annotate.py --config config/religious_missions.yaml --canary
```

If integrated into orchestration, it should be opt-in, for example a separate `--canary` mode or monitoring pseudo-stage, not part of the default `01`-`09` chain.

## Question Group 5: Rerun Reproducibility and Overwrite Behavior

### If we rerun stages 07-10, do we get the same results?

Mostly yes numerically, if all inputs, code, checkpoint, package versions, hardware, precision, and config remain unchanged. But a direct rerun is not a clean no-op and will overwrite several artifacts.

Stage behavior:

| Stage | Rerun behavior |
|---|---|
| 07 | Rewrites `anchor_oof_scores.parquet`, `calibrator.json`, and `rule_validation.json`, then refuses to overwrite existing `test_evaluation.json`. |
| 08 | Reuses matching shards when possible; overwrites `predictions.parquet` and `monitor_scores.json`. |
| 09 | Overwrites `prevalence_by_ntee.csv` and `prevalence_report.json`. |
| 10 | Overwrites fixed figure filenames in `data/processed/figures/`. |

Important stage-07 issue: The frozen-test one-shot guard happens after calibration-side artifacts are already rewritten. Therefore, a casual rerun with existing `test_evaluation.json` can partially overwrite stage-07 outputs and then fail.

### Determinism risks

Seeded or stable components:

| Component | Determinism note |
|---|---|
| Config seed | `SEED: 42` |
| Calibration folds | Seeded stratified K-fold |
| Bootstrap CIs | Seeded RNG with configured `bootstrap_resamples: 2000` |
| Inference order | Sorted by stable `EIN2` |
| DeBERTa inference precision | Runtime log confirms fp32 override |

Sources of non-byte-identical output:

| Source | Effect |
|---|---|
| Timestamps | `test_evaluation.json`, prediction metadata, and logs include run dates. |
| Plot metadata | Figure files may differ byte-for-byte even if visual data is the same. |
| Runtime environment | Hardware, package versions, or precision changes can cause tiny numeric differences. |

### Are artifacts overwritten?

Yes. Most stage 07-10 outputs use fixed filenames.

Overwritten artifacts include:

| Stage | Artifact |
|---|---|
| 07 | `data/processed/evaluation/anchor_oof_scores.parquet` |
| 07 | `data/processed/evaluation/calibrator.json` |
| 07 | `data/processed/evaluation/rule_validation.json` |
| 08 | `data/processed/predictions/predictions.parquet` |
| 08 | `data/processed/predictions/monitor_scores.json` |
| 09 | `data/processed/prevalence/prevalence_report.json` |
| 09 | `data/processed/prevalence/prevalence_by_ntee.csv` |
| 10 | `data/processed/figures/*.png`, `*.svg`, `*.pdf` with matching names |

Protected artifact:

| Stage | Artifact | Behavior |
|---|---|---|
| 07 | `data/processed/evaluation/test_evaluation.json` | Existing file causes stage 07 to raise rather than overwrite. |

Logs are timestamped and are not overwritten.

### Stale prediction shards

Stage 08 does not delete old shards. If a future inference run has fewer rows, or uses `--infer-limit`, old higher-numbered shard files may remain in `data/processed/predictions/shards/`.

The merge uses only the current run's shard paths, so stale shards are ignored, but they can confuse manual inspection.

Safe rerun recommendations:

1. Do not casually rerun stage 07 unless intentionally replacing frozen evaluation artifacts.
2. Archive current `data/processed/evaluation`, `data/processed/predictions`, `data/processed/prevalence`, and `data/processed/figures` before rerunning.
3. Clear or archive `data/processed/predictions/shards/` before a smaller or limited inference run.
4. Consider adding a run-id output directory or artifact archiving convention before future UCloud reruns.
5. Consider moving the stage-07 frozen-test guard earlier so reruns do not partially overwrite calibration artifacts before failing.

## Recommended Engineering Follow-Ups

Priority 1:

| Follow-up | Why |
|---|---|
| Fix `production_annotation_summary` sizing | Current output is unusable despite successful rendering. |
| Persist PR curve points from stage 07 | Enables the missing `precision_recall_curve` output. |
| Decide whether inference should map duplicate mission-text predictions back to all raw EIN2 | Current output classifies deduplicated EIN2 only. |

Priority 2:

| Follow-up | Why |
|---|---|
| Add prevalence decomposition visualization | Best bridge from ML metrics to substantive CSS estimate. |
| Add rule-layer validation interval plot | LOW tier materially affects prevalence uncertainty. |
| Add score distribution plots by label/tier | Makes threshold behavior and ambiguity visible. |
| Add frozen-test PR points if needed | Anchor OOF PR curve is useful, but frozen-test PR is cleaner for evaluation reporting. |

Priority 3:

| Follow-up | Why |
|---|---|
| Add opt-in canary drift orchestration | Useful for monitoring, but should not be default core pipeline. |
| Suppress or relabel unstable per-NTEE fallback estimates | Avoid overinterpreting low-support subgroup diagnostics. |
| Add run-id or archive mechanism | Prevent accidental overwrites and improve auditability. |

## Bottom Line

The model is strong enough to pass the current evaluation gate, and the full deduplicated prediction corpus has been classified. The main scientific caveats are the narrow PR-AUC margin, the low recall-oriented threshold, small subgroup cells, and LOW-tier prevalence uncertainty. The main engineering caveats are deduplicated rather than raw EIN2 coverage, missing persisted PR curve points, the broken production annotation summary layout, and fixed output paths that are overwritten on rerun.
