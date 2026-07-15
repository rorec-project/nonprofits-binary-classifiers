# Human Gates (G1–G4)

Four human checkpoints gate the pipeline before it commits GPU work or irreversible operations.

## G1 — Labels gate

Before stages 02, 04, 06, and 07, the pipeline validates that `gold_to_code.csv` has complete `0/1` human labels for every row in the required split (`prompt_dev` for 02, `validation` for 04/06, `test` for 07). If labels are missing, blank, or non-`0/1`, the run exits gracefully (no GPU work wasted).

## G2 — Slate gate

Before stage 03 (full annotation), the pipeline requires a human-confirmed `data/processed/gold/production_slate.json`. If stages 02+03 are requested together and no confirmed slate exists, stage 02 runs (produces `proposed_slate.json`), then the pipeline exits gracefully before stage 03.

## G3 — Test-unlock gate

Before stage 07 (evaluation), the pipeline requires two sibling artifacts under `data/processed/gold/`. The gate protects the frozen test split: no model sees test data until a human has reviewed the selected checkpoint and signed off on acceptance thresholds.

**Purpose:** Prevent accidental test-set leakage during model iteration. The frozen test split is evaluated only once — after a human inspects the selected model's validation performance, confirms the acceptance criteria, and explicitly unlocks the test set.

**Wave-6 refinement:** stage 07 now treats the frozen-test report as a one-shot artifact. If `data/processed/evaluation/test_evaluation.json` already exists, the stage blocks before overwriting evaluation artifacts. That safeguard is what keeps the real test report re-openable only under the controlled post-sprint procedure.

### Two-file workflow

| File | Role | Created by |
|------|------|------------|
| `selected_model.json` | Pins which checkpoint to evaluate | Human copy of `selected_model_skeleton` from `selection_report.json` |
| `test_unlock.json` | Human sign-off that the frozen test may be run | Manually written by the human reviewer |

### 1. `selected_model.json`

After stage 06 produces `selection_report.json`, the human reviewer inspects the report and copies its `selected_model_skeleton` object verbatim into `data/processed/gold/selected_model.json`. This file tells the pipeline which checkpoint (by SHA-256) to load for evaluation.

Example (taken from the current `selection_report.json`):

```json
{
  "checkpoint_relpath": "checkpoints/microsoft__deberta-v3-base/default/s44/checkpoint-2690/model.safetensors",
  "checkpoint_sha256": "8fd26faa3abaf5f1a45fb884ff17ca6757a61ba219d9afbd313fb7ff9c06e885",
  "encoder_id": "microsoft/deberta-v3-base",
  "recipe_snapshot": {
    "arm": "default",
    "batch_size": 32,
    "early_stopping_patience": 3,
    "epochs": 10,
    "learning_rate": 2e-05,
    "warmup_fraction": 0.06,
    "weight_decay": 0.01
  },
  "selected_at": "2026-07-01T18:49:28.149192+00:00",
  "targets": "soft",
  "tokenizer_id": "microsoft/deberta-v3-base"
}
```

### 2. `test_unlock.json`

The human reviewer writes this file to record their sign-off. It snapshots the acceptance criteria in effect at unlock time so that later config changes do not silently re-evaluate the test set under different thresholds (the gate detects drift and blocks the run).

**Schema** (`TestUnlock` in `src/binary_classifier/config.py:276`):

| Field | Type | Description |
|-------|------|-------------|
| `confirmed` | `bool` | Must be `true` after human review authorises test evaluation. |
| `checkpoint` | `str` | Human-readable label or relative checkpoint path. |
| `checkpoint_sha256` | `str` | SHA-256 digest of the selected model checkpoint. **Must match** the value in `selected_model.json`. |
| `acceptance` | `object` | Snapshot of acceptance thresholds at unlock time. |
| `acceptance.min_pr_auc` | `float` | Minimum precision-recall AUC on the frozen test set. |
| `acceptance.min_minority_f1_ci_lower` | `float` | Minimum lower confidence bound for minority-class F1. |
| `acceptance.max_ece` | `float` | Maximum expected calibration error on anchor OOF scores. |
| `rationale` | `str` | Free-text human rationale for unlocking the test set. |

Example (matching the `selected_model.json` above and the current `AcceptanceCriteria` defaults):

```json
{
  "confirmed": true,
  "checkpoint": "checkpoints/microsoft__deberta-v3-base/default/s44/checkpoint-2690/model.safetensors",
  "checkpoint_sha256": "8fd26faa3abaf5f1a45fb884ff17ca6757a61ba219d9afbd313fb7ff9c06e885",
  "acceptance": {
    "min_pr_auc": 0.90,
    "min_minority_f1_ci_lower": 0.70,
    "max_ece": 0.05
  },
  "rationale": "default arm with soft targets clears min_pr_auc 0.90 and min_minority_f1_ci_lower 0.70 on validation — proceed to frozen test."
}
```

### Validation checks

The gate (`_validate_test_unlock` in `src/binary_classifier/qc/preflight.py:204`) runs every check below. All must pass before stage 07 begins.

1. `test_unlock.json` exists at `data/processed/gold/test_unlock.json`.
2. The file is valid JSON conforming to the `TestUnlock` schema.
3. `confirmed` is `true`.
4. The `acceptance` sub-object matches the current `cfg.evaluation.acceptance` field-for-field. If a config change drifts any value, the gate blocks.
5. `selected_model.json` exists at `data/processed/gold/selected_model.json`.
6. `checkpoint_sha256` in `test_unlock.json` matches `checkpoint_sha256` in `selected_model.json`.

### How to verify

```bash
# Run the pre-flight validator on gates G1–G4 directly:
uv run python -c "from binary_classifier.config import load_config; from binary_classifier.paths import PathRegistry; from binary_classifier.qc.preflight import validate_gates; cfg = load_config(); reg = PathRegistry.from_config(cfg); print(validate_gates(cfg, reg, {'07','09'}))"

# Or request stages 07–09 — the pipeline gates itself before any test work:
uv run python scripts/run_pipeline.py --stages 07,08,09
```

### Failure mode

If any check fails, the pipeline prints one or more problem messages prefixed with `G3:` and exits with code 2. If an existing `test_evaluation.json` is present, stage 07 also exits before overwriting stage-07 outputs. The frozen test set is never re-scored accidentally.

### Controlled reopen path (§7)

The only sanctioned way to replace the real frozen-test artifact is the controlled post-sprint UCloud re-evaluation:

1. archive the existing `test_evaluation.json`, `calibrator.json`, `rule_validation.json`, and `anchor_oof_scores.parquet` with the original git SHA;
2. run stage 07 exactly once on code-frozen UCloud infrastructure;
3. regenerate stages 08–10 from that one-shot stage-07 output; and
4. record the reproduce-assertion result in `data/processed/run_manifest.json`.

That procedure is documented for operators in `docs/audits/20260702-local-evaluation-refresh.md`. It is not a local-debugging workflow.

## G4 — Anchor-labels gate

Before stages 07 and 09 (prevalence), the pipeline requires a fully coded `anchor_to_code.csv` for the anchor sample. Without it, prevalence estimates on LOW-quality rows cannot be validated.

## Related

- [pipeline.md](pipeline.md) — which stages each gate protects
- [../overview.md](../overview.md) — project status
