# Local Evaluation Refresh — Wave 6 Documentation Pass

**Reviewed:** 2026-07-02
**Scope:** Documentation-only refresh after Waves 1–5. Uses the current local Wave 2–4 artifacts and known local Wave 1/3 results. Does **not** reopen the real frozen test. Frozen-test PR/ROC curves and multi-threshold confusion-matrix numbers are therefore still **pending §7**.
**Supersedes for current guidance:** [religious_evaluation_report.md](religious_evaluation_report.md)

---

## 1. Executive summary

- **Frozen-test acceptance is still the same archived one-shot result.** The current shared `test_evaluation.json` still supports the same headline metrics: F1 `0.8941176470588236`, recall `0.987012987012987`, precision `0.8172043010752689`, PR-AUC `0.9013902009153695`, ROC-AUC `0.9491121123774185`, and ECE `0.007083317363206847`.
- **The prevalence headline changed.** The current local corrected composite prevalence is `0.14239498698911504` (**14.24%**), down about **0.21 percentage points** from the older `14.45%` audit headline.
- **The LOW tier is now decomposed instead of treated as one rule-corrected block.** The current local LOW estimate is `0.17332868644104676` (**17.33%**). The old all-Rogan-Gladen LOW estimate was `0.1836004865` (**18.36%**).
- **Release-time thresholding is now explicitly documented for deployment prevalence.** The current anchor-OOF-derived base-rate threshold is `0.09368807964553742`; the `0.90` target is attainable, with base-rate-adjusted precision `0.9028691334068896` at derived base rate `0.11880075889216252`.
- **Inference now has two artifacts, not one.** Keep `predictions.parquet` for the deduplicated scoring corpus and release `predictions_full.parquet` for raw-`EIN2` organization-level use.
- **Stage 10 is now registered in `run_pipeline.py`, and orchestrated runs write `data/processed/run_manifest.json`.**

---

## 2. What changed since the earlier audit

| Area | Earlier audit | Current Wave-6-local status |
|---|---|---|
| Prevalence estimand | Effectively deduplicated-text corpus | Explicitly **per-organization** (`EIN2`) |
| LOW handling | One all-LOW Rogan-Gladen correction | `low_via_classifier` → PPI; pure-rule LOW → Rogan-Gladen |
| Released labels | `pred_label` only | `pred_label`, `pred_label_maxf1`, `pred_label_baserate` |
| Release artifact | `predictions.parquet` only | `predictions.parquet` + `predictions_full.parquet` |
| Base-rate deployment view | Not reported | `base_rate_precision.json` + base-rate threshold documented |
| Visualization orchestration | Stage 10 script-only | Stage 10 registered as stage `10` in `run_pipeline.py` |
| Reproducibility envelope | Distributed artifact notes | Single `data/processed/run_manifest.json` |

---

## 3. Current classifier-evaluation claim

The model-evaluation claim is unchanged from the archived one-shot frozen-test report:

| Metric | Current shared value |
|---|---:|
| Precision | `0.8172043010752689` |
| Recall | `0.987012987012987` |
| F1 | `0.8941176470588236` |
| MCC | `0.8092651388884946` |
| PR-AUC | `0.9013902009153695` |
| ROC-AUC | `0.9491121123774185` |
| ECE | `0.007083317363206847` |

Interpretation: this remains a **recall-first** operating point chosen on anchor OOF data (`pred_label`, threshold `0.05769250483141822`). It is good evidence that the classifier can find religious mission text on the hard, enriched frozen-test split, but it is **not** a deployment-prevalence claim by itself.

---

## 4. Current prevalence claim

### Headline

The current local corrected prevalence estimate is:

| Quantity | Value |
|---|---:|
| Composite prevalence | `0.14239498698911504` |
| Composite prevalence (%) | `14.24%` |
| 95% CI | `0.12632041470291652` to `0.15846955927531356` |
| Shift vs earlier `14.45%` headline | about `-0.21` percentage points |

### Decomposition now used

| Component | Value |
|---|---:|
| HIGH/MEDIUM weighted PPI | `0.13430909705821076` |
| LOW composite | `0.17332868644104676` |
| LOW classifier-routed PPI sub-stratum | `0.1377788936016389` |
| LOW pure-rule Rogan-Gladen sub-stratum | `0.2153826755768503` |
| Old LOW all-RG estimate | `0.1836004865` |

This is the main substantive change from the earlier audit: LOW is no longer summarized as one rule-corrected block when `59,704` LOW rows were actually classifier-routed.

### Local caveat on multiplicity

The current local prevalence report records:

```text
raw_multiplicity_source = unit_fallback_no_raw_match
```

That happened because the local dry-run did not have `predictions_full.parquet` and matching raw parquets available at the time of estimation. The **canonical per-organization artifacts are regenerated after the post-sprint UCloud re-evaluation (§7)**.

---

## 5. Current release-threshold claim

The released dataset now carries three binary labels for different uses:

| Label column | Threshold | Intended use |
|---|---:|---|
| `pred_label` | `0.05769250483141822` | Recall-first prevalence label; stage 09 uses this one |
| `pred_label_maxf1` | `0.6082766564370073` | Balanced single-threshold classifier view |
| `pred_label_baserate` | `0.09368807964553742` | Deployment label targeting `0.90` precision at the estimated population base rate |

Base-rate precision summary from the current anchor OOF derivation:

| Quantity | Value |
|---|---:|
| Target precision | `0.90` |
| Attainable? | Yes |
| Selected base-rate threshold | `0.09368807964553742` |
| Base-rate-adjusted precision | `0.9028691334068896` |
| Derived population base rate | `0.11880075889216252` |

---

## 6. Current artifact and figure status

### Inference and release artifacts

- `data/processed/predictions/predictions.parquet` remains the **deduplicated** scoring artifact used by existing local consumers.
- `data/processed/predictions/predictions_full.parquet` is the **per-organization** release artifact, expanding predictions back to every raw `EIN2` row.
- `data/processed/run_manifest.json` is the reproducibility envelope for orchestrated runs.

### Figure suite

Wave 4/Wave 5 documentation should now treat the following as the current figure surface:

- `bakeoff_summary`
- `production_annotation_summary`
- `documentation_curve`
- `precision_recall_curve`
- `frozen_test_confusion_matrices`
- `reliability_diagram`
- `score_distribution_by_tier_label`
- `prevalence_forest`
- `prevalence_decomposition`
- `rule_validation_intervals`
- `quantification_sensitivity`
- `subgroup_performance`
- `ngram_log_odds`
- optional `canary_drift`

---

## 7. Pending §7 controlled UCloud re-evaluation

### Frozen-test PR/ROC curves

**Pending §7 finalization.** The shared real `test_evaluation.json` still lacks `test_scores`, so this report does **not** publish real frozen-test PR/ROC points yet. Smoke renders them; the real versions are finalized only after the controlled UCloud rerun.

### Multi-threshold frozen-test confusion matrices

**Pending §7 finalization.** The report therefore does **not** publish real frozen-test confusion matrices for:

- operating threshold (`pred_label`)
- max-F1 threshold (`pred_label_maxf1`)
- base-rate threshold (`pred_label_baserate`)

Those sections should be populated only after the one-shot UCloud rerun persists real per-row frozen-test scores.

---

## 8. Reader map

- For operators/agents: [../agents/pipeline/pipeline.md](../agents/pipeline/pipeline.md)
- For non-technical readers: [../nontechnical-overview.md](../nontechnical-overview.md)
- For release consumers: [../predictions-full-data-dictionary.md](../predictions-full-data-dictionary.md)
