# Master Harmonization — Landing the Refactored Pipeline

**Date:** 2026-07-15
**Branch:** `refactor/harmonize-pipeline` → `master`
**Scope:** Full refactor of the original flat-script + notebook pipeline into a config-driven, package-structured, gated pipeline, with a finalized frozen-test evaluation and a documented release contract.

This document is the durable, self-contained reference for the harmonization refactor. It reads between a computational-social-science methods section and a technical changelog: it explains **what the pipeline now is, end to end**, **what changed relative to the old master**, and **how to reproduce the headline results**. The pull request that lands this branch carries the narrative; this file carries the detail. For the plain-language version see [../nontechnical-overview.md](../nontechnical-overview.md); for the released-dataset contract see [../predictions-full-data-dictionary.md](../predictions-full-data-dictionary.md).

---

## 1. What this branch does

The original `master` (archived; see §9) was a flat collection of scripts and notebooks (`split_data.py`, `generate_training_data.py`, `final_finetuning*.ipynb`, ad-hoc CSVs under `train_test_datasets/`). This branch replaces it with a **reusable package + thin CLI stages + a config-driven orchestrator**, adds a full evaluation/calibration/prevalence stack, and documents a release contract suitable for a public repository.

Divergence from the merge base (`6ab99cb`, 2026-06-03):

| Measure | Value |
|---|---|
| Commits | 109 (linear; master unmoved) |
| Files changed | 256 |
| Insertions / deletions | +58,038 / −29,823 |
| Commit types | 38 feat · 24 chore · 21 fix · 15 docs · 7 refactor · 2 perf · 1 style · 1 build |

The work was developed as a linear stack of feature branches (`feature/05-anchor-sample` … `feature/11-aggregation-compare`, over `fix/close-validation-loop` and `fix/pipeline-hardening`). Because ~24 later commits fix earlier stages, those intermediate branches are **stale snapshots** and are **not** used as separate PRs; the branch lands as a single `--no-ff` merge that preserves all 109 commits as honest history.

---

## 2. Task and approach

A **config-driven binary text classifier** that labels short US-nonprofit records as **religious (`1`) vs non-religious (`0`)**. The design is entity-agnostic: the religious × missions task is the first of several planned (activities, pregnancy centers, education, …), selected by `config/*.yaml` — never hard-coded.

**LLM-as-primary weak supervision.** A model × prompt ensemble (a closed-API reference model plus open-weight models served via vLLM) labels a large silver pool; the votes are aggregated into silver labels. A small hand-coded **gold** set drives prompt selection, validation, and a one-shot frozen test. `EIN2` (the raw IRS EIN) is the join key on every artifact.

---

## 3. Reproducibility envelope

The headline results below come from the controlled post-sprint UCloud re-evaluation. Its provenance (`data/processed/run_manifest.json`, now tracked in-repo):

| Field | Value |
|---|---|
| `generated_at` | 2026-07-05T18:23:10Z |
| `git_sha` | `9b668e245ebeec6660ae23ad275a4426018d3a53` |
| `git_tag` | `sprint-hp-20260705-9b668e2` |
| `config_hash` | `1365985bae96960cb845645217c066891e13409c187624a686cf94f3312a02d6` |
| Python | 3.13.12 |
| `uv.lock` sha256 | `eef1d6eb59c178458a9cb72984c997330d191b0c4b62634dec19b18aad20c0eb` |

**Input row counts:** silver manifest 20,000 · gold manifest 450 · anchor manifest 500 · `predictions.parquet` 531,660 · `predictions_full.parquet` 560,354.

**Release thresholds:** operating `0.05769`, max-F1 `0.60828`, base-rate `0.09369`.

Only `viz/prevalence_plots.py` (a plotting file) changed after `9b668e2`; **no evaluation logic changed after the frozen results were written**, so the numbers in §5 are consistent with the code that ships on master.

---

## 4. Pipeline, top to bottom

Reusable logic lives in `src/binary_classifier/` (`data/`, `annotate/`, `qc/`, `train/`, `evaluation/`, `inference/`, `prevalence/`, `viz/`, `repro/`). `scripts/01…11` are thin CLI wrappers that load the config + a `PathRegistry` and call a package function; `scripts/run_pipeline.py` chains 01→10 and emits `run_manifest.json`. Stage 11 is script-only.

1. **01 build_sample** — sample a silver pool + a small gold set; write seeded `EIN2` manifests.
2. **02 bakeoff_prompts** — model × prompt bake-off on a prompt-dev split; pick the model slate + prompts (selection by Cohen's κ + minority-F1 CI floor).
3. **03 annotate** — full matrix labeling into a resumable long/tidy store; OpenAI Batch API + vLLM providers; `--canary` produces a drift-audit report.
4. **04 quality_check** — freeze majority-vote silver labels after the LLM-vs-human agreement gate.
5. **05 build_anchor** — full-frame anchor sample (incl. LOW tier) with design weights for prevalence.
6. **06 train** — CPU baselines + DeBERTa/ModernBERT two-phase sweep+final training, OOF cross-fit scoring, model selection, final-seed refit; persists `selected_model.json` (with checkpoint SHA).
7. **07 evaluate** — cross-fit calibration, base-rate precision diagnostics, rule validation, and the **one-shot frozen-test acceptance gate**. Emits `test_evaluation.json`, `base_rate_precision.json`, `calibrator.json`, `rule_validation.json`, `anchor_oof_scores.parquet`.
8. **08 infer** — full-corpus sharded inference with LOW-tier rule routing, triple labels (`pred_label`, `pred_label_maxf1`, `pred_label_baserate`), and expand-back from deduplicated text rows to raw-`EIN2` `predictions_full.parquet`.
9. **09 prevalence** — per-organization prevalence: HIGH/MEDIUM PPI, LOW classifier-routed PPI, LOW rule-only Rogan–Gladen, and raw-`EIN2` tier-share recombination.
10. **10 visualize** — publication-grade figures (Okabe–Ito palette) over evaluation, inference, and prevalence artifacts.
11. **11 aggregation_compare** — script-only sensitivity diagnostics for alternative silver-label aggregation methods.

The pipeline runs behind four human gates ([G1–G4](../agents/pipeline/human-gates.md)). The legacy flat-script pipeline is preserved in `archive/legacy-pipe/` and is **not executed**.

---

## 5. Headline results (frozen test)

Frozen-test acceptance **passed** all gates (`max_ece ≤ 0.05`, `min_minority_f1_ci_lower ≥ 0.7`, `min_pr_auc ≥ 0.9`). Aggregate metrics at the operating threshold:

| Metric | Value |
|---|---|
| F1 (minority/religious) | 0.8941 |
| Precision | 0.8172 |
| Recall | 0.9870 |
| PR-AUC | 0.9014 |
| ROC-AUC | 0.9491 |
| Balanced accuracy | 0.9068 |
| Cohen's κ | 0.7958 |
| MCC | 0.8093 |
| ECE | 0.0071 |
| Minority-F1 bootstrap 95% CI | [0.8398, 0.9381] |

Per-threshold operating points and confusion matrices are reported in [20260702-local-evaluation-refresh.md §7](20260702-local-evaluation-refresh.md#7-frozen-test-results-finalized-via-controlled-ucloud-re-evaluation). Corrected composite **prevalence** is **14.24%** (per-organization estimand; LOW tier decomposed into classifier-routed PPI + rule-only Rogan–Gladen).

---

## 6. Data & release contract

The repository is intended to become public. The contract:

**Tracked in git** (small, provenance, replication-sufficient, no bulk corpus):
- `data/processed/gold/` — human-annotated gold/anchor sets (`EIN2`, split/tier, text, human_label; ~950 rows) plus model-selection provenance (`production_slate.json`, `selected_model.json`, `test_unlock.json`).
- `data/processed/evaluation/` — frozen-test artifacts (`test_evaluation.json`, `base_rate_precision.json`, `calibrator.json`, `rule_validation.json`, `anchor_oof_scores.parquet`). Text-free, ~464 KB, **one-shot** (deliberately not regenerated locally). Newly tracked by this landing (see §8).
- `data/processed/run_manifest.json` — the reproducibility envelope of §3.

**Not tracked** (heavy and/or raw corpus; available on request):
- Bulk silver labels (~20 k) and the full raw scoring corpus (`predictions*.parquet`, 531 k / 560 k rows).
- Regenerable figures (`data/processed/figures/`).
- Cross-section source parquets (from the sibling `NonProfitData` repo).

**Principle:** track join keys, labels, and provenance; never the bulk text corpus or anything regenerable. Mission text for the small gold set is retained (ground truth); mission text at silver/full scale is not redistributed.

---

## 7. Human gates

Landing does not bypass the [G1–G4 gates](../agents/pipeline/human-gates.md). In particular, the frozen test remains **one-shot**: the shared `test_evaluation.json` is finalized only via the controlled post-sprint UCloud re-evaluation (done 2026-07-05), never reopened locally.

---

## 8. What this landing changed (beyond the branch work)

Two housekeeping changes were made while preparing the merge:

1. **Tracked the frozen evaluation artifacts.** `data/processed/evaluation/` and `run_manifest.json` were de-symlinked from the cloud drive into real in-repo files, with targeted `.gitignore` exceptions mirroring the existing `gold/` treatment, so the numbers travel with a clone.
2. **Reconciled stale evaluation docs.** The `pending §7 / TBD` placeholders in [20260702-local-evaluation-refresh.md](20260702-local-evaluation-refresh.md) were replaced with the real finalized frozen-test numbers, so master lands internally consistent.

The ~21 `.agents/plans/*.md` agent scratchpads are **kept** as a development-timeline record (to be reprocessed later).

---

## 9. Archive of the old master

The pre-refactor `master` is preserved immutably as tag **`archive/master-2026-06-17`** (`6ab99cb`, on origin). Old-master-only artifacts (legacy CSVs under `train_test_datasets/`, root notebooks) live in that tag; nothing is lost by the clean-state merge.

---

## 10. Reader map

- Pipeline stage map — [../agents/pipeline/pipeline.md](../agents/pipeline/pipeline.md)
- Configuration & retasking — [../agents/pipeline/configuration.md](../agents/pipeline/configuration.md)
- Human gates — [../agents/pipeline/human-gates.md](../agents/pipeline/human-gates.md)
- Finalized frozen-test detail — [20260702-local-evaluation-refresh.md](20260702-local-evaluation-refresh.md)
- Released-dataset contract — [../predictions-full-data-dictionary.md](../predictions-full-data-dictionary.md)
- Plain-language overview — [../nontechnical-overview.md](../nontechnical-overview.md)
- Academic write-up — [../../paper/paper.md](../../paper/paper.md)
