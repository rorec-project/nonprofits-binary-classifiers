# `predictions_full.parquet` Data Dictionary

`data/processed/predictions/predictions_full.parquet` is the **per-organization** release artifact. It expands the deduplicated inference results back to raw `EIN2` rows.

## What is guaranteed in this file

The file keeps the raw input columns from the source parquet and appends the guaranteed prediction-contract columns below.

| Column | Meaning |
|---|---|
| `EIN2` | Raw organization identifier. This is the join key for downstream use. |
| `pred_label` | Recall-first binary label at the operating threshold. This is the prevalence label used by stage 09. |
| `pred_label_maxf1` | Binary label at `threshold_maxf1`. |
| `pred_label_baserate` | Binary label at `threshold_baserate`. |
| `prob_raw` | Uncalibrated positive-class score from the classifier. Null for deterministic rule rows. |
| `prob_calibrated` | Calibrated positive-class score from stage 07. Null for deterministic rule rows. |
| `decision_source` | How the row was labeled: e.g. `classifier`, `low_via_classifier`, `rule_strong_positive`, `rule_short_negative`, or `rule_abstain`. |
| `tier` | Quality tier assigned from the text-quality rubric: `HIGH`, `MEDIUM`, or `LOW`. |
| `Q` | Numeric text-quality score used to derive `tier`. |
| `ntee_major_group` | NTEE major-group code carried through for subgroup analysis and prevalence reporting. |
| `model_id` | Encoder/model identifier used for inference. |
| `checkpoint_sha256` | SHA-256 digest of the reviewed checkpoint. |
| `calibrator_method` | Calibration method chosen in stage 07 (for example `platt`). |
| `calibrator_params_hash` | Hash of the fitted calibrator parameters. |
| `threshold` | Operating threshold used for `pred_label`. |
| `threshold_maxf1` | Threshold used for `pred_label_maxf1`. |
| `threshold_baserate` | Threshold used for `pred_label_baserate`. |
| `inference_date` | Timestamp for the inference run. |
| `pipeline_version` | Installed package version at inference time. |
| `config_hash` | Hash of the validated pipeline config used to produce the artifact. |

## How the three labels differ

| Column | Current threshold | Use |
|---|---:|---|
| `pred_label` | `0.05769250483141822` | Recall-first prevalence label |
| `pred_label_maxf1` | `0.6082766564370073` | Max-F1 release label |
| `pred_label_baserate` | `0.09368807964553742` | Base-rate-targeted release label |

## How to join it

Join downstream datasets on **raw `EIN2`**.

This file exists specifically so consumers do **not** need to reconstruct labels from the deduplicated `predictions.parquet` artifact.

## Important conventions

### Duplicate text = same mission

The expand-back step assumes that identical values of the configured text field describe the same mission for classification purposes. If two raw organizations share the same mission text, they receive the same copied score and labels in `predictions_full.parquet`.

### Rule rows can have null probabilities

Rows labeled by deterministic rules may have:

- `pred_label` / `pred_label_maxf1` / `pred_label_baserate` populated, but
- `prob_raw` and `prob_calibrated` null.

That is expected behavior, not missing data.

### `predictions.parquet` still exists

`predictions.parquet` remains the deduplicated scoring artifact. It is useful for internal scoring and reproducibility. `predictions_full.parquet` is the release artifact for organization-level use.

## Related

- Technical refresh report: [audits/20260702-local-evaluation-refresh.md](audits/20260702-local-evaluation-refresh.md)
- Plain-language overview: [nontechnical-overview.md](nontechnical-overview.md)
