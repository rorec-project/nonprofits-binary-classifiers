# Binary Classification of Religious vs. Non-Religious Nonprofit Missions

A reproducible, config-driven binary text classifier for US nonprofit missions. The pipeline labels short text records as religious (`1`) vs. non-religious (`0`) using an LLM-as-primary weak-supervision ensemble, then fine-tunes modern encoder models on the resulting silver dataset so the final research deliverable is a **calibrated, uncertainty-quantified population-prevalence estimate** (PPI++). Designed for extensibility beyond the religious task to pregnancy centers, education, international aid, and other nonprofit sectors.

> **What this is, in one sentence:** We use several AI models to cheaply label thousands of nonprofit mission statements, have a human spot-check a tiny gold sample to keep the AIs honest, train a small neural network on the silver labels, and then statistically correct the raw counts into a rigorous population prevalence estimate with confidence intervals.

> **Analogy:** imagine you want to know what fraction of a giant library's books are about religion. You can't read every book. Instead, you ask three fast (but occasionally sloppy) research assistants to tag 20,000 books, you personally re-read 450 of those to check the assistants' work, you train a careful librarian on the checked set, and then you use statistics to correct the librarian's remaining guesses into an unbiased estimate for the whole library.

---

## Big Picture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  01 Sample   │────▶│ 02 Bake-off  │────▶│ 03 Annotate  │
│   + Gold     │     │   + Slate    │     │  full matrix │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                           │
       │ [HUMAN G1]                                │ [HUMAN G2]
       │ code gold_to_code.csv 0/1                 │ confirm production_slate.json
       ▼                                           ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 04 QC / Freeze│────▶│ 05 Anchor    │────▶│ 06 Train     │
│ silver labels  │     │   sample     │     │ sweep → refit│
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
       │                    │ [HUMAN G4]          │ [HUMAN G3]
       │                    │ code anchor_to_code │ confirm test_unlock.json
       ▼                    ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 07 Evaluate  │────▶│ 08 Infer     │────▶│ 09 Prevalence│
│ calibration  │     │ full corpus  │     │ PPI++ + EMQ  │
│ frozen test  │     │ rule layer   │     │ population CI│
└──────────────┘     └──────────────┘     └──────────────┘
```

**The core loop:** cheap AI labels ⇄ small human gold check. A few models × a few prompts label the silver pool; a human codes a tiny gold set to pick the best models and validate quality. The gold is then **locked out** of training so the model never sees the final exam. The trained classifier is run over the whole corpus, and a statistical quantification layer turns the raw predictions into a corrected population share with bootstrap confidence intervals.

---

## Three Core Ideas

1. **AI bulk-labels, humans spot-check.** We do not hand-label 20,000 records. Instead, an ensemble of LLMs (OpenAI + open-weight models via vLLM) labels the bulk silver pool. A small, boundary-enriched gold set (~450 records) is hand-coded by a human. The gold drives prompt selection, QC gates, and frozen-test evaluation. The silver is the training fuel; the gold is the truth meter.

2. **The model never sees the final exam.** The test split is drawn during stage 01 and is **frozen** thereafter. A human gate (G3 — `test_unlock.json`) is required before stage 07 can touch it, ensuring no accidental leakage during model iteration. The model is trained on silver + validation; evaluated on the locked test.

3. **We statistically _correct_ counts for population estimates.** Classifier predictions are biased by the training distribution. We use PPI++ (Prediction-Powered Inference, Angelopoulos et al. 2023) as the primary prevalence estimator, with EMQ (vendored SLD implementation, Saerens 2002) as a single sensitivity cross-check. The anchor sample (stage 05, including LOW-quality rows) provides the labeled holdout needed to de-bias the full-corpus inference into a population-representative share with a bootstrap confidence interval.

---

## Quickstart

### Install

```bash
# Clone the repository
git clone https://github.com/rorec-project/nonprofits-binary-classifiers.git
cd nonprofits-binary-classifiers

# Install dependencies (requires uv — https://docs.astral.sh/uv/getting-started/installation/)
uv sync

# For optional extras (vLLM serving, quantification diagnostics, crowd-label diagnostics):
uv sync --all-extras

# Set secrets (OpenAI API key needed for the closed-reference annotator)
echo "OPENAI_API_KEY=your_key_here" > .env
```

The default `uv sync` installs the lean runtime. Optional extras are:

- `--extra serve` — vLLM for open-weight model serving
- `--extra quant` — QuaPy for KDEy quantification cross-checks
- `--extra diagnostics` — `crowd-kit` and `cleanlab` for stage-11 sensitivity diagnostics

Key dependency changes vs. the legacy stack: `pyarrow`, `python-dotenv`, `openai`, and `vllm` (via `--extra serve`).

### Smoke test (no upstream data)

```bash
uv run python scripts/run_pipeline.py --config config/smoke.yaml --stages 01
```

This generates a small synthetic dataset, stamps it `data_source="synthetic"`, and verifies the pipeline wiring. Intended for development only — never use synthetic data for production runs.

### Operator loop (the four gates)

The pipeline is driven by two human-coded artifacts and two human confirmations. Think of them as four gates the pipeline refuses to pass until a human signs off:

| Gate                   | Human action                                                                          | What it unlocks                                                           |
| ---------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **G1 — Labels**        | Code `gold_to_code.csv` with strict `0/1` for every row                               | Stage 02 (bake-off), stage 04 (QC), stage 06 (train), stage 07 (evaluate) |
| **G2 — Slate**         | Copy `proposed_slate.json` → `production_slate.json`, set `"confirmed": true`, commit | Stage 03 (full annotation)                                                |
| **G4 — Anchor labels** | Code `anchor_to_code.csv` with strict `0/1` for every row                             | Stage 07 (evaluate), stage 09 (prevalence)                                |
| **G3 — Test unlock**   | Create `test_unlock.json` with `"confirmed": true` + the selected checkpoint SHA      | Stage 07 (evaluate)                                                       |

> **G2 auto-exit:** if you request `--stages 02,03` together and no confirmed slate exists, stage 02 runs (produces `proposed_slate.json`), then the pipeline exits gracefully at G2 so you can review the scores before committing to the full annotation run.

Typical workflow:

```bash
# Stage 01: build sample + gold template. Then code gold_to_code.csv (G1).
uv run python scripts/run_pipeline.py --stages 01

# Stage 02: bake-off. Review scores in bakeoff_results.json, then copy proposed_slate.json → production_slate.json (G2).
uv run python scripts/run_pipeline.py --stages 02

# Stage 03-04: full annotation + QC freeze.
uv run python scripts/run_pipeline.py --stages 03,04

# Stage 05: anchor sample. Then code anchor_to_code.csv (G4).
uv run python scripts/run_pipeline.py --stages 05

# Stage 06: train sweep → final refit.
# Review selection_report.json, copy its selected_model_skeleton into
# data/processed/gold/selected_model.json, then create test_unlock.json (G3).
uv run python scripts/run_pipeline.py --stages 06

# Stage 07-09: evaluate → infer → prevalence.
uv run python scripts/run_pipeline.py --stages 07,08,09
```

If gates are not satisfied, the pipeline exits **gracefully** with a clear message — no GPU work or API spend is wasted.

---

## The 9 Stages in a nutshell

Below is a one-paragraph summary of each stage. For full technical details (I/O schemas, columns, thresholds, and CLI flags), see [Appendix B: Stage-by-stage technical reference](#appendix-b-stage-by-stage-technical-reference).

1. **Stage 01 — Build sample** draws the silver pool (20,000 HIGH+MEDIUM quality records) and the gold set (450 records split into `prompt_dev`, `validation`, `test`, and `monitor`). Writes `EIN2` manifests and `gold_to_code.csv`. [Details → B1](#b1-stage-01)

2. **Stage 02 — Bake-off** scores every configured `bakeoff_candidate` × `prompt` against the human-coded `prompt_dev` split. Writes `bakeoff_results.json` and an unconfirmed `proposed_slate.json`. [Details → B2](#b2-stage-02)

3. **Stage 03 — Annotate** labels the full silver pool (plus gold holdouts) using only the models confirmed in `production_slate.json` (G2). The label store is keyed by `(EIN2, source_id)` so crashes are resumable. [Details → B3](#b3-stage-03)

4. **Stage 04 — QC / Freeze** aggregates the labels (majority vote by default), measures agreement against the human-coded `validation` split, and writes a frozen `silver_labels.csv` that excludes every gold `EIN2` from the training pool. [Details → B4](#b4-stage-04)

5. **Stage 05 — Anchor sample** draws a representative anchor sample (n=500) from the **full** frame including LOW-quality rows, so prevalence estimates do not treat the HIGH+MEDIUM silver/gold frame as population-representative. [Details → B5](#b5-stage-05)

6. **Stage 06 — Train** runs a selection sweep (encoder arms × seeds) on the silver labels, then automatically refits the best configuration across **5 final seeds**. Produces `selection_report.json` and checkpoint directories. [Details → B6](#b6-stage-06)

7. **Stage 07 — Evaluate** reports minority-class precision/recall/F1, MCC, balanced accuracy, PR-AUC, bootstrap intervals, and calibration metrics on the **frozen test set** (G3). The acceptance gate uses `min_pr_auc`, `min_minority_f1_ci_lower`, and `max_ece`. [Details → B7](#b7-stage-07)

8. **Stage 08 — Infer** runs the selected classifier over HIGH/MEDIUM rows, routes LOW/bare-label rows through the high-precision rule layer, and persists `EIN2` + positive-class probabilities. [Details → B8](#b8-stage-08)

9. **Stage 09 — Prevalence** estimates the population share over all nonprofits with PPI++ as the primary estimator, plus an EMQ cross-check (vendored SLD). Reports bootstrap confidence intervals and, when enabled, per-NTEE-stratum breakdowns. [Details → B9](#b9-stage-09)

---

## The 4 Human Checkpoints (G1–G4)

| Gate   | Full name          | Before stage(s) | Artifact                                    | Required state                                                                                                                                  |
| ------ | ------------------ | --------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **G1** | Labels gate        | 02, 04, 06, 07  | `data/processed/gold/gold_to_code.csv`      | Every row in the needed split has a strict `0/1` `human_label` (no blanks, no other values)                                                     |
| **G2** | Slate gate         | 03              | `data/processed/gold/production_slate.json` | `"confirmed": true` and lists the exact model IDs for production                                                                                |
| **G4** | Anchor-labels gate | 07, 09          | `data/processed/gold/anchor_to_code.csv`    | Every row has a strict `0/1` `human_label`                                                                                                      |
| **G3** | Test-unlock gate   | 07              | `data/processed/gold/test_unlock.json`      | `"confirmed": true` + `checkpoint_sha256` matching the selected model + acceptance snapshot; the test split is never seen until this gate opens |

> **Why G3 is last:** the frozen test is the _final exam_. The model is selected and trained without ever touching it. Only after a human explicitly unlocks the test by recording the selected checkpoint SHA does evaluation run. This prevents any accidental leakage during model iteration.

---

## How to Run an Evaluation and Read the Results

### Running the evaluation

Evaluation is stage 07, which requires gates G1, G3, and G4:

```bash
uv run python scripts/run_pipeline.py --stages 07
```

The stage checks:

- `gold_to_code.csv` has coded `test` rows (G1)
- `test_unlock.json` is confirmed with the correct checkpoint SHA (G3)
- `anchor_to_code.csv` is fully coded (G4)

If any gate is missing, the pipeline exits gracefully before loading a model.

### Acceptance criteria

The frozen-test acceptance gate checks three thresholds (configurable in `config/*.yaml`):

| Criterion                  | Threshold | What it means                                                                                                           |
| -------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------- |
| `min_pr_auc`               | `0.90`    | The classifier must distinguish religious vs. non-religious with at least 0.90 precision-recall AUC on the frozen test. |
| `min_minority_f1_ci_lower` | `0.70`    | The lower bound of the 95% bootstrap CI for minority-class F1 must be ≥ 0.70.                                           |
| `max_ece`                  | `0.05`    | The expected calibration error (ECE) on the anchor out-of-fold scores must be ≤ 0.05.                                   |

If any threshold is missed, the pipeline exits non-zero and prints guidance. The gate does **not** automatically retrain; it reports so the operator can decide whether to revise prompts, adjust the slate, or override the config.

### Output files

- `data/processed/evaluation/test_evaluation.json` — full frozen-test metrics: PR-AUC, minority-class precision/recall/F1, MCC, balanced accuracy, bootstrap 2000-resample CIs, length-binned subgroup reports, and calibration diagnostics (Platt + temperature scaling).
- `data/processed/prevalence/prevalence_report.json` — population prevalence estimate with PPI++ primary and EMQ cross-check, 95% bootstrap CI, and per-NTEE stratum estimates where available.

### Reading the prevalence report

Open `prevalence_report.json`. The top-level fields of interest are:

- `hm.weighted_ppi` — the primary PPI++ estimate with design weights, plus upper/lower confidence bounds.
- `hm.unweighted_ppi` — the unweighted PPI++ estimate for comparison.
- `composite` — the Rogan-Gladen corrected composite over all tiers.
- `cross_checks.emq` — the EMQ (Saerens 2002) sensitivity check.
- `cross_checks.kdey` — the KDEy estimate (present if `quant` extra is installed).
- `low.sensitivity_band` — bounds on the estimate under systematic rule-layer misclassification (present when `low_tier_sensitivity: true`).

> **Important caveat — LOW-tier missions:** The silver/gold sampling frame is HIGH+MEDIUM only (`Q >= 3.0`). LOW-quality records (bare labels, fragments) are excluded from stage 01 and handled by the rule layer at inference. The prevalence report folds them back in via the anchor sample, but the LOW-tier rate is a high-precision rule estimate, not a classifier score. Any claim about the _full_ nonprofit population must include the LOW-tier sensitivity bounds, which the report provides when `prevalence.low_tier_sensitivity: true` in the config.

---

## Configuring in 60 Seconds

Everything is driven by a single YAML file. The default is `config/religious_missions.yaml`. To retask the pipeline for a new classification task (e.g., "activities" instead of "missions"):

1. Copy `config/religious_missions.yaml` to `config/<task>.yaml`.
2. Set `entity`, `field`, and `label_name` to the new task.
3. Point the orchestrator at it: `uv run python scripts/run_pipeline.py --config config/<task>.yaml --stages 01`.

No source code edits are needed, provided the upstream `NonProfitData` parquet exposes the chosen `field`.

→ **Full config reference:** [`config/README.md`](config/README.md) — every knob, default, and Pydantic class.

---

# Appendices

---

## Appendix A: Data Layout & DVC

The pipeline follows the cookiecutter-data-science layout ([cookiecutter-data-science, drivendata](https://cookiecutter-data-science.drivendata.org/)):

- `data/raw/` — immutable upstream parquet inputs. Not committed.
- `data/interim/` — manifests, bake-off outputs, and annotation stores. Not committed.
- `data/processed/` — final artifacts. `data/processed/gold/` is the **committed pointer layer** (`gold_to_code.csv`, `production_slate.json`, `test_unlock.json`, `selected_model.json`). `silver_labels.csv` is not committed.
- `data/models/` — fine-tuned checkpoints. Not committed.

Today, local/cloud symlinks stand in for DVC for every heavy, non-committed location (`raw/`, `interim/`, `processed/silver_labels.csv`, `models/`). The committed pointers are the small gold artifacts in `data/processed/gold/`.

**Local setup before your first non-synthetic run:**

```bash
mkdir -p data/raw data/interim data/models data/processed/gold
```

- `PathRegistry.ensure_dirs()` creates output directories such as `data/interim/`, `data/models/`, and `data/processed/gold/`, but it does **not** create `data/raw/` for you.
- Stage 01 needs `data/raw/missions_cross_section.parquet` and `data/raw/bmf_unified_processed.parquet`. These are produced by the sibling repository at `../NonProfitData/` (see that project's build instructions). To run without them, set `data.allow_synthetic: true` in your config for smoke tests.
- Move any existing committed gold artifacts into `data/processed/gold/` before running the human checkpoints.
- If your heavy silver artifacts still live behind the old processed-tree cloud symlink, re-point that local symlink so manifests, bake-off outputs, and annotation stores live under `data/interim/` instead. Moving gold alone does not move the large silver-side artifacts.

**Future DVC migration:** when the project adopts DVC, add the real upstream parquet files with `dvc add data/raw/*.parquet` and configure a `dvc remote` (DVC docs: `add`, cache link types, remotes, external data). Avoid stacking ad-hoc intermediate directory symlinks on top of DVC-managed paths because DVC manages its own cache links.

---

## Appendix B: Stage-by-Stage Technical Reference

### Package layout (quick reference)

```
src/binary_classifier/
  config.py            # Typed config (pydantic) loaded from config/*.yaml
  paths.py             # pathlib.Path registry (no string concatenation)
  metrics.py           # Core classification metrics (PR-AUC, MCC, balanced acc, bootstrap CI)
  data/
    load.py            # Cross-section load + EIN2→BMF NTEE-major-group join
    quality.py         # Computable Q rubric + HIGH/MED/LOW tiering + rule layer
    anchor.py          # Stratified anchor sampling with inclusion probabilities
    sample.py          # Stratified + positive-enriched sampling; EIN2 manifests
  annotate/
    schema.py          # Long/tidy label schema + pydantic JSON parse
    prompts/           # Versioned codebook prompt templates (v1, v2, v3)
    annotators/
      base.py          # Provider-agnostic Annotator interface
      openai_annotator.py
      vllm_annotator.py
    run_annotation.py  # Batch labeling; resume by (EIN2, source_id)
    aggregate.py       # Stage 04 majority vote + Stage 11 diagnostic arms
  qc/
    agreement.py       # Validation gate: κ/α reporting + minority-F1 CI floor
    preflight.py       # Human gates G1–G4
    aggregation_compare.py # Stage 11 sensitivity/diagnostic comparison report
  train/               # Stage 06: baselines, encoder sweep, cross-fit OOF, selection
  evaluation/          # Stage 07: calibration, thresholds, subgroups, frozen-test gate
  inference/           # Stage 08: rule/classifier routing + sharded full-corpus inference
  prevalence/          # Stage 09: PPI++ / Rogan–Gladen / quantification cross-checks
  viz/                 # Stage 10: n-gram log-odds + metric/calibration/prevalence plots
scripts/
  01_build_sample.py   # Stage 01: construct silver + gold + manifests
  02_bakeoff_prompts.py # Stage 02: model×prompt bake-off on prompt-dev
  03_annotate.py       # Stage 03: full model×prompt matrix labeling over silver ∪ gold
  04_quality_check.py  # Stage 04: QC gate + freeze silver-only labels
  05_build_anchor.py   # Stage 05: full-frame anchor sample (incl. LOW)
  06_train.py          # Stage 06: baselines + encoder sweep + final-seed refit
  07_evaluate.py       # Stage 07: calibration + frozen-test acceptance gate
  08_infer.py          # Stage 08: full-corpus inference with LOW-tier rule routing
  09_prevalence.py     # Stage 09: population prevalence (PPI++ + Rogan–Gladen)
  10_visualize.py      # Stage 10: script-only figure renderer
  11_aggregation_compare.py # Stage 11: script-only aggregation comparison
  run_pipeline.py      # Orchestrator: chains 01→09 with human gates G1–G4
config/
  religious_missions.yaml   # First task config (entity=missions, field=LONGEST_MISSION)
```

### B1. Stage 01 — Build sample

Stage 01 constructs the silver training pool and gold human-coding set from mission texts scored by the quality rubric Q, using proportional allocation across all 26 NTEE major groups with floors at 200 rows for thin strata (V, Y, U) and caps at 2,500 rows for large strata (B, P). Stratification by sector is essential because religious prevalence varies enormously — Education (B) is overwhelmingly secular while Religion (X) is almost entirely religious — and without it the silver pool would be dominated by non-religious rows from the largest sectors. Within each stratum, positive enrichment targets a 35% minority-class share (King & Zeng 2001), and cell-level inclusion probabilities are recorded in the manifest so downstream design weights invert the correct draw rate rather than a stratum-wide marginal rate (Horvitz & Thompson 1952). The gold set deliberately over-samples boundary cases — saint-named secular organizations, spiritual-not-religious framings, generic ministry language, and faith-heritage references — to stress-test human coders and prompt candidates. Each stratum is carved into disjoint quota cells (boundary, clear-positive, clear-negative, and a filler top-up) so the realized set hits its configured size exactly — 450 records — without the de-duplication shortfall that overlapping cells incur; the realized per-quota-cell draw rate is persisted as `quota_cell_rate` for audit transparency but is diagnostic only, not a design weight, because the filler stage makes it an invalid marginal inclusion probability. Put precisely, gold is a stratified, boundary-enriched evaluation set; selection is randomized within a partition of each stratum; reported per-cell draw rates are diagnostic only; and all population inference uses the design-weighted anchor sample with PPI. The set then splits into four disjoint, NTEE-stratified slices: prompt_dev for the bake-off, validation for QC gates, a frozen test for one-shot evaluation, and a monitor slice held out as drift canaries — while probability calibration and the operating threshold are fit on the anchor rather than this enriched set, so the enrichment never leaks into the population-targeting quantities. Re-running stage 01 after coding has started reshuffles these split tags, so the `--force` flag explicitly acknowledges the clobber risk and discards any existing human labels.

- **Inputs:** `data/raw/missions_cross_section.parquet`, `data/raw/bmf_unified_processed.parquet` (or synthetic if `data.allow_synthetic: true`).
- **Outputs:**
  - `data/interim/manifests/silver_manifest.csv` — `EIN2` list for silver pool.
  - `data/interim/manifests/gold_manifest.csv` — `EIN2` list for gold set.
  - `data/processed/gold/gold_to_code.csv` — human coding template.
- **Key options:** `--force` (regenerate from scratch, discarding human labels).
- **Caveat:** re-running stage 01 after coding starts reshuffles gold split tags.

### B2. Stage 02 — Bake-off

Stage 02 runs a bake-off that scores every model-candidate × prompt-template combination against the human-labeled prompt-dev split, selecting candidates that pass a dual threshold (κ ≥ 0.70 AND minority-F1 CI lower bound ≥ 0.70), then ranking passing candidates by minority-F1 point estimate. Cohen's κ measures chance-corrected agreement, eliminating the "always-say-majority" baseline that inflates raw accuracy when the positive class is rare, while the minority-F1 component guards against prompts that correctly classify negatives but fail to detect positives. The slate draws from two provider pools — OpenAI API models for reliability and open-weight models via vLLM for cost efficiency and local reproducibility — and adding a new model to either pool requires no code change beyond registering its identifier and provider. Each (model, prompt) combination receives a unique source_id in the long/tidy annotation schema, so the pipeline can add new candidates without schema migration. The top-ranking combinations enter a proposed slate serialized with `"confirmed": false` and await human review at gate G2 before advancing to production labeling.

- **Inputs:** `gold_to_code.csv` (prompt_dev split, G1), config `model_slate.bakeoff_candidates`.
- **Outputs:**
  - `data/interim/bakeoff/bakeoff_results.json` — scores per candidate × prompt.
  - `data/interim/bakeoff/proposed_slate.json` — auto-picked, unconfirmed slate.
- **Key options:** `--prompts`, `--human-labels`, `--limit`, `--store-path`, `--output`.
- **Caveat:** bake-off skips vLLM gracefully if server unreachable.

### B3. Stage 03 — Annotate

Stage 03 applies the human-confirmed production slate to the silver pool, scoring every row across all model × prompt combinations in the ensemble. The annotation store uses a long/tidy schema keyed by (EIN2, source_id), where source_id encodes the model × prompt identity, allowing the pipeline to add or remove annotators without altering the store structure. Crashes are resumable at the (EIN2, source_id) granularity: completed rows are skipped automatically on restart, and the `--no-resume` flag forces a fresh run from scratch. Checkpointing at configurable intervals and a `--canary` mode for small-scale smoke tests support iterative prompt development without exhausting the annotation budget.

- **Inputs:** `production_slate.json` (G2), silver manifests.
- **Outputs:** `data/interim/annotation_store.csv` — long/tidy label store keyed by `(EIN2, source_id)`.
- **Key options:** `--limit`, `--no-resume`, `--canary`, `--checkpoint-every`.
- **Caveat:** crashes resumable by `(EIN2, source_id)`; `--no-resume` starts fresh.

### B4. Stage 04 — QC / Freeze

Stage 04 freezes the weak-supervision labels by QC-gating the ensemble against the human-coded validation split before writing the silver label file. The primary gate uses Cohen's κ at a 0.70 threshold because chance-corrected agreement eliminates the inflation that raw agreement produces when models concur on easy negatives even when both are wrong — a common failure mode in imbalanced classification. A secondary gate checks that the lower bound of the 95% bootstrap percentile interval (2,000 resamples) for minority-class F1 stays above 0.70, ensuring the ensemble performs adequately on the rare class and not just overall. If either threshold fails, the script exits non-zero and prints guidance to revise prompts and re-annotate, never writing a frozen label file. Raw agreement is still logged for backward-compatible auditing, but the freeze decision — and the output that excludes all gold EIN2 rows to prevent double-dipping — depends on the chance-corrected criteria.

- **Inputs:** annotation store, `gold_to_code.csv` (validation split, G1).
- **Outputs:** `data/processed/silver_labels.csv` — frozen majority-vote labels, excluding gold `EIN2`s.
- **Key options:** `--human-validation`, `--store-path`, `--output`.
- **Caveat:** if QC gate misses thresholds, exits non-zero without writing frozen output.

### B5. Stage 05 — Anchor sample

Stage 05 draws an anchor sample for prevalence debiasing by stratifying the full corpus — including LOW-tier rows excluded from the silver pool — proportionally across NTEE major group × quality tier at the cell level. Including LOW-tier texts in the anchor is essential because they are too short or boilerplate for reliable classifier calibration, yet they constitute a large fraction of the population, so their misclassification rate must be quantified. Every sampled row carries its cell-level inclusion probability, which becomes the Horvitz-Thompson design weight that PPI++ inverts in stage 09 to produce asymptotically unbiased population estimates. The anchor must be fully coded before stages 07 and 09 can proceed (gate G4); without it, prevalence estimates on LOW-quality rows cannot be validated, and the calibration fitting in stage 07 lacks ground truth for the tier that most needs it.

- **Inputs:** full frame (all Q tiers, including LOW).
- **Outputs:** `data/processed/gold/anchor_to_code.csv` — human coding template.
- **Key options:** `--force`.
- **Caveat:** anchor must be coded before stages 07 and 09 (G4).

### B6. Stage 06 — Train

Stage 06 fine-tunes encoder models on the frozen silver labels, enumerating a grid of training-arm configurations across 3 seeds (42–44) for the selection sweep, then refitting the chosen configuration across 5 seeds (42–46) for reported variance. The loss function is a soft cross-entropy optionally multiplied by inverse-frequency class weights: L = -(1/N) Σ[y_i log(p_i)·w₁ + (1 - y_i) log(1 - p_i)·w₀], where w_k = n/(2·n_k). Soft targets (vote shares) are the default because they preserve annotator-confidence information and are inherently more robust to label noise than hard thresholds (Zhu et al. 2022), which makes label smoothing redundant. The default arms are hard and class-weighted; the pruned arm that drops cleanlab-flagged rows from the disagreement band (Northcutt, Jiang & Chuang 2021) is opt-in because soft targets already down-weight the same rows, and focal loss — designed for object detection — is excluded as a poor fit for text classification. DeBERTa-v3-base serves as primary encoder (He et al. 2021), chosen for its disentangled attention and relative position bias that gave SOTA classification performance at release, compared against ModernBERT-base (Warner et al. 2024) and TF-IDF/MiniLM (Reimers & Gurevych 2019) logistic-regression baselines. Numerical precision resolves to bf16 when the GPU supports it and falls back to fp32; the documentation-curve sweep defaults to the full-data run only, and training uses early stopping with patience 4. The selection report aggregates across seeds and picks the simplest cell through a sequential pairwise comparison: starting from the highest-ranked cell by mean PR-AUC, each challenger is accepted if its mean PR-AUC is within `max(incumbent_sd, challenger_sd)` of the current incumbent, preferring DeBERTa over ModernBERT, soft over hard targets, and default over class-weighted arms. Cross-fit out-of-fold probabilities are computed for all arms including the opt-in pruned arm, feeding the diagnostic stage 11 comparison even though they do not enter production.

- **Inputs:** frozen `silver_labels.csv`, config `training` section.
- **Outputs:** `data/models/selection_report.json`, checkpoint directories under `data/models/`.
- **Key options:** `--baselines-only`, `--sweep`, `--final`, `--encoder`, `--subset`, `--epochs`, `--seeds`, `--limit`.
- **Caveat:** orchestrator runs stage 06 as two-phase sweep → final automatically.

### B7. Stage 07 — Evaluate

Stage 07 loads the human-confirmed checkpoint, calibrates its raw probabilities on the anchor set, selects an operating threshold, and scores the frozen test split in a one-shot evaluation. Calibration compares Platt scaling (Platt 1999) against temperature scaling (Guo et al. 2017) through stratified K-fold cross-fitting: each method fits on K-1 folds, scores the held-out fold, and the winner is the method with the lowest mean out-of-fold Brier score, with log-loss as tiebreaker. Platt is retained as the default because its intercept can absorb prior-shift offsets from the enriched training pool; isotonic regression is excluded because it overfits at the anchor's typical size. The threshold search iterates over all unique validation probabilities and selects the highest-recall threshold whose precision meets the 0.80 floor; if no candidate achieves the floor, it falls back to the maximum-precision threshold and flags `floor_unattainable`. The rule layer applied to LOW-tier texts is separately validated on the anchor, with sensitivity and specificity reported alongside Wilson confidence intervals. The frozen-test report records minority-class precision/recall/F1, MCC, PR-AUC with bootstrap 2,000-resample CIs, length-binned subgroup analyses, and calibration diagnostics, then checks acceptance: PR-AUC ≥ 0.90, minority-F1 CI lower bound ≥ 0.70, and ECE ≤ 0.05. Failing any check blocks the pipeline but preserves the report for audit.

- **Inputs:** frozen test split (G1 + G3), selected checkpoint, `anchor_to_code.csv` (G4).
- **Outputs:** `data/processed/evaluation/test_evaluation.json`.
- **Caveat:** failing acceptance checks blocks pipeline but preserves report.

### B8. Stage 08 — Infer

Stage 08 runs full-corpus inference through a five-route decision matrix driven by the quality tier and the deterministic rule layer. HIGH- and MEDIUM-quality rows route directly to the classifier. LOW-quality and bare-label rows first encounter the rule layer: strong religious-lexicon hits are labeled positive (route `rule_strong_positive`), very short texts with no religious signal are labeled negative (`rule_short_negative`), and ambiguous texts either fall through to the classifier (`low_via_classifier`) or abstain (`rule_abstain`) depending on the `rule_ambiguous_to_classifier` configuration flag. The rule layer is safer for LOW-tier texts because they tend to be too short or boilerplate for reliable classifier calibration, whereas a lexicon rule with known precision bounds from anchor validation provides a verifiable floor on decision quality. The output is a parquet file with positive-class probabilities and model-version metadata for every row in the corpus.

- **Inputs:** full corpus, selected checkpoint, rule layer configuration.
- **Outputs:** `data/processed/predictions/predictions.parquet`.
- **Key options:** `--limit`.

### B9. Stage 09 — Prevalence

Stage 09 estimates population prevalence using prediction-powered inference (PPI++) as the primary method (Angelopoulos et al. 2023; arXiv:2311.01453). PPI++ constructs a debiasing term from the anchor sample's prediction errors and applies it to the full-corpus probabilities, yielding an asymptotically unbiased prevalence estimate with valid confidence intervals. The power-tuning parameter λ automatically interpolates between the unbiased but wide CI from labeled data alone (λ = 0) and the narrowest possible CI under a perfectly calibrated model (λ = 1). Design weights derived from the anchor's cell-level inclusion probabilities follow the Horvitz-Thompson estimator and are passed directly to ppi_py. A cross-check uses expectation-maximization for quantification (EMQ, vendored from Saerens 2002), which assumes the class-conditional score densities shift by a known factor between the anchor and the population — a different identifying assumption than PPI++'s correct-model-specification approach — and disagreement between the two methods serves as a diagnostic red flag for model misspecification or covariate shift. Settings default to 95% bootstrap CIs with 2,000 resamples, per-NTEE stratum reporting where n ≥ 10, and optional LOW-tier sensitivity bounds.

- **Inputs:** predictions, anchor sample labels (G4), design weights.
- **Outputs:** `data/processed/prevalence/prevalence_report.json`.

### B10. Stage 10 — Visualization (script-only, not orchestrated)

Stage 10 is a script-only figure renderer that produces PNG and SVG visualizations under the output directory. It skips missing inputs gracefully rather than failing. Signed n-gram log-odds bars replace word clouds because they are reproducible, statistically interpretable, and avoid introducing a new word-cloud dependency. The module runs independently of the main pipeline, rendering plots for whichever upstream stage produced the latest artifacts.

- **Outputs:** PNG and SVG figures under `data/processed/viz/`.
- **Caveat:** script-only; skips missing inputs gracefully.

### B11. Stage 11 — Aggregation diagnostics (script-only, not orchestrated)

Stage 11 is a script-only sensitivity diagnostic that compares multiple label-aggregation methods — simple majority vote, Dawid-Skene (Dawid & Skene 1979), and CROWDLAB — on the human-coded validation set. The comparison metric is each method's agreement with the human labels, producing a report that quantifies how much the choice of aggregation affects the final silver labels. These dependencies (`crowd-kit` and `cleanlab`) are gated behind the `diagnostics` extra because the comparisons are purely diagnostic — stage 04 production labels remain majority-vote for stability, and stage 11 never activates a replacement aggregation method. The cross-fit out-of-fold probabilities computed during stage 06 feed the CROWDLAB comparison arm.

- **Outputs:** `data/interim/aggregation_compare.json`.
- **Caveat:** script-only; diagnostic-only (`crowd-kit` and `cleanlab` behind `diagnostics` extra).

---

## Appendix C: Methodology & Citations

### C1. Simplifications and intentional skips (this design pass)

The project follows a **one principled primary method per concern + minimal robustness** principle. Tertiary and diagnostic machinery is pushed to optional extras or omitted entirely. All citations appear in C3.

| Design Decision                                                                                                                    | Rationale                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Label aggregation:** majority vote only in production; Dawid-Skene and CROWDLAB in optional `diagnostics` extra (stage 11).      | Majority vote is the simplest defensible aggregation and is standard in weak-supervision pipelines. Dawid-Skene provides no consistent benefit at our per-item annotation depth.                                                                                                                                                                                   |
| **Quantification cross-checks:** EMQ (vendored SLD) as single sensitivity check; KDEy via QuaPy is opt-in under the `quant` extra. | EMQ is lightweight, interpretable, and well-suited to binary prevalence. KDEy requires additional dependencies and is secondary.                                                                                                                                                                                                                                   |
| **Training arms:** default arms `[hard, class_weighted]`; `pruned` arm dropped.                                                    | Soft vote-share labels already down-weight the disagreement band that `pruned` targets. `pruned` remains opt-in via config.                                                                                                                                                                                                                                        |
| **Learning curve:** full-data run only (`[1.0]`); `{25%, 50%, 100%}` sweep dropped.                                                | The learning-curve sweep did not change model selection in initial experiments.                                                                                                                                                                                                                                                                                    |
| **Decision-curve analysis:** dropped.                                                                                              | Decision-curve analysis requires an explicit treatment-threshold cost ratio, which is undefined for a population-prevalence study where the downstream deliverable is a calibrated prevalence estimate (PPI++), not a treat-vs-abstain decision threshold. Net-benefit curves would require additional clinical-domain assumptions that do not apply to this task. |
| **Calibration:** Platt scaling and temperature scaling compared; isotonic regression excluded.                                     | Platt and temperature scaling are parametric, stable at small anchor size (n~500). Isotonic regression overfits at this sample size.                                                                                                                                                                                                                               |
| **Final seeds:** retained at 5 seeds (`[42, 43, 44, 45, 46]`).                                                                     | Research convention mandates ≥5 seeds for variance reporting (Camuffo et al. 2026). Seed variation captures optimization stochasticity that bootstrap alone does not.                                                                                                                                                                                              |
| **Acceptance gate (calibration component):** ECE-only; Brier / log-loss gate excluded.                                             | Brier score and log-loss are strictly proper scoring rules with desirable properties (ECE is not strictly proper), but ECE is the standard metric in the calibration literature (Guo et al. 2017) and is what reviewers and practitioners expect. Brier / log-loss gating is future work.                                                                          |
| **Data augmentation:** none applied.                                                                                               | Soft labels from the LLM ensemble already add noise robustness; augmentation would dilute ensemble disagreement signal for short (≤50 word) mission texts without clear benefit.                                                                                                                                                                                   |

### C2. What was _not_ simplified (research fidelity)

- **PPI++** remains the primary prevalence estimator (Angelopoulos et al. 2023; PPI++ arXiv:2311.01453).
- **Majority-vote** production aggregation remains the default; the stage-11 diagnostic arms are strictly non-production.
- **Frozen-test gate (G3)** remains untouched.
- **Gold/silver discipline** remains untouched: gold is hand-coded, silver is LLM-labeled, and gold `EIN2`s are never in the training pool.
- **Global seeds** (`SEED: 42`) and **bootstrap CIs** (2000 resamples) remain untouched.
- **Minority-class metrics** and the **LOW-tier rule layer** remain untouched.
- **DeBERTa-v3 vs. ModernBERT** comparison remains untouched.
- **5 final seeds** for variance reporting remain untouched.
- **EIN2** is carried through every artifact as the upstream join key.

### C3. Full citation list

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023). Prediction-Powered Inference. _arXiv:2311.01453_. — Primary prevalence estimator (PPI++).
- Camuffo, A., Gambardella, A., Kazemi, M., Malachowski, S., & Pandey, S. (2026). Variance-Aware Protocol for Minority-Class Evaluation. _arXiv:2601.02370_. — Minority-F1 CI floor motivation.
- Cheng, S., Mayya, V., & Sedoc, J. (2025). SILICON: A Generalization of the Agreement on Inference Framework to Diverse Decision Scenarios. _arXiv:2412.14461_. — Agreement benchmarking in weak-supervision contexts.
- Cookiecutter Data Science. https://github.com/drivendata/cookiecutter-data-science. — Data layout convention.
- Davis, J. & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. _Proceedings of ICML 2006_, 233–240. DOI: 10.1145/1143844.1143874. — Primary evaluation metric for classifier acceptance.
- Dawid, A. P. & Skene, A. M. (1979). Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. _Applied Statistics_, 28(1), 20–28. DOI: 10.2307/2346806. — Diagnostic label-aggregation method.
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. _Annals of Statistics_, 7(1), 1–26. DOI: 10.1214/aos/1176344552. — Bootstrap confidence intervals for all reported metrics and prevalence estimates.
- Gentzkow, M., Kelly, B., & Taddy, M. (2019). Text as Data. _Journal of Economic Literature_, 57(3), 535–574. DOI: 10.1257/jel.20181020. — Text-as-data methodology for economics.
- Gheibi, O. & Ghazizadeh, E. (2025). Understanding Disagreement in Weak Supervision. _arXiv:2605.20642_. — Disagreement-band rationale.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. _Proceedings of ICML 2017_, 1321–1330. arXiv:1706.04599. — Expected calibration error (ECE) and temperature scaling for classifier calibration.
- He, P., Gao, J., & Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. _arXiv:2111.09543_. — Primary encoder architecture.
- Hopkins, D. J. & King, G. (2010). A Method of Automated Nonparametric Content Analysis for Social Science. _American Journal of Political Science_, 54(1), 229–247. DOI: 10.1111/j.1540-5907.2009.00428.x. — Aggregate text prevalence estimation.
- Horvitz, D. G. & Thompson, D. J. (1952). A Generalization of Sampling Without Replacement from a Finite Universe. _Journal of the American Statistical Association_, 47(260), 663–685. DOI: 10.2307/2280784. — Design-weighted estimation.
- Keith, K. & O'Connor, B. (2018). Uncertainty-Aware Generative Models for Inferring Document Class Prevalence. _Proceedings of EMNLP 2018_, 4185–4195. DOI: 10.18653/v1/D18-1487. — Aggregate prevalence with uncertainty.
- King, G. & Zeng, L. (2001). Logistic Regression in Rare Events Data. _Political Analysis_, 9(2), 137–163. DOI: 10.1093/oxfordjournals.pan.a004868. — Positive-enrichment sampling rationale.
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. _Proceedings of SOSP '23_, 611–626. DOI: 10.1145/3600006.3613165. — LLM serving infrastructure for open-weight model arm.
- Landis, J. R. & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. _Biometrics_, 33(1), 159–174. DOI: 10.2307/2529310. — Kappa interpretation scale.
- Matthews, B. W. (1975). Comparison of the Predicted and Observed Secondary Structure of T4 Phage Lysozyme. _Biochimica et Biophysica Acta_, 405(2), 442–451. DOI: 10.1016/0005-2795(75)90109-9. — Reported evaluation metric.
- Meyer, B. D. & Mittag, N. (2017). Misclassification in Binary Choice Models. _Journal of Econometrics_, 200(2), 244–259. DOI: 10.1016/j.jeconom.2017.06.012. — Econometric misclassification correction.
- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict. _Political Analysis_, 16(4), 372–403. DOI: 10.1093/pan/mpn018. — Visualization method for signed n-gram log-odds bars.
- Moreo, A., Esuli, A., & Sebastiani, F. (2021). QuaPy: A Python Framework for Quantification. _Proceedings of CIKM '21_, 4060–4064. DOI: 10.1145/3459637.3482015. — Quantification library.
- Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). Confident Learning: Estimating Uncertainty in Dataset Labels. _Journal of Artificial Intelligence Research_, 70, 1373–1411. DOI: 10.1613/jair.1.12125. — Pruned arm / disagreement-band removal.
- Platt, J. C. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. In _Advances in Large Margin Classifiers_, 61–74. MIT Press. — Platt scaling.
- Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. _Proceedings of the VLDB Endowment_, 11(3), 269–282. DOI: 10.14778/3157794.3157797. — Weak-supervision label model.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. _Proceedings of EMNLP-IJCNLP 2019_, 3982–3992. DOI: 10.18653/v1/D19-1410. — Baseline embedding model for logistic regression.
- Rogan, W. J. & Gladen, B. (1978). Estimating Prevalence from the Results of a Screening Test. _American Journal of Epidemiology_, 107(1), 71–76. DOI: 10.1093/oxfordjournals.aje.a112510. — Prevalence adjustment.
- Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the Outputs of a Classifier to New a Priori Probabilities. _Neural Computation_, 14(1), 217–247. DOI: 10.1162/089976602753284446. — EMQ / SLD cross-check.
- Singh, S., Bhargav, A., Ratner, A., & Ré, C. (2025). Disagreement as a Signal for Weak Supervision. _arXiv:2511.14117_. — Disagreement-band rationale.
- Vickers, A. J. & Elkin, E. B. (2006). Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. _Medical Decision Making_, 26(6), 565–574. DOI: 10.1177/0272989X06295361. — Decision-curve analysis.
- Warner, B. et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Fine-Tuning and Inference. _arXiv:2412.13663_. — Comparison encoder architecture.
- Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. _Journal of the American Statistical Association_, 22(158), 209–212. DOI: 10.2307/2276774. — Confidence intervals for rule-layer validation.
- Zhu, Z., Shi, D., Liu, T., & Sugiyama, M. (2022). Robust Learning under Label Noise with Iterative Noise-Filtering. _arXiv:2203.05677_. — Soft target robustness rationale.

---

## Appendix D: Sampling Frame vs. Population

The current sampling frame is the **HIGH + MEDIUM quality strata only** (`Q >= 3.0`). LOW-quality records (bare labels, fragments) are excluded from stage-01 sampling and are handled later by the rule layer at inference. That means the sampled frame is **not** the full nonprofit population.

Any population-share claim over all nonprofits must fold LOW back in explicitly using the rule-layer label rate multiplied by the LOW-count mass. The anchor sample (stage 05) includes LOW rows precisely so the prevalence estimator can perform this fold-in.

This framing is deliberate: the pipeline optimizes annotation and QC on text that is informative enough for LLM review, while keeping a clear hook for the later all-nonprofits prevalence estimator via the anchor + rule-layer combination.

---

## Appendix E: UCloud Runtime

GPU jobs run on the **UCloud SDU/DeiC Interactive HPC** platform. See [`docs/RUNNING_ON_UCLOUD.md`](docs/RUNNING_ON_UCLOUD.md) for the full runbook, including:

- Job submission (`gpu-nvidia-b200` SKU, full GPUs, not `-mig` fractional)
- `/work` persistence rules (only `/work` survives job termination; mount a Drive there)
- `utils/init.sh` environment setup (uv pre-installed, Python 3.13 self-managed, B200 `sm_100` requires PyTorch 2.7+ from the CUDA 12.8 `cu128` wheel index)
- vLLM serve on localhost (`--tensor-parallel-size 8`)
- SSH access and secret management (store `OPENAI_API_KEY` in a `.env` inside a private `/work` Drive)
- GPU compatibility check (`nvidia-smi`, `torch.cuda.get_device_name(0)`)

Quick excerpt for the impatient:

```bash
# On UCloud, after cloning to /work/nonprofits-binary-classifiers
cd /work/nonprofits-binary-classifiers
bash utils/init.sh
set -a
source /work/.env
set +a

# Serve open-weight model (only on jobs that need it)
uv run vllm serve google/gemma-3-27b-it --tensor-parallel-size 8 --port 8000

# Train (lean sync, no serve extra)
uv sync
uv run python scripts/run_pipeline.py --stages 06
```

---

## Appendix F: Troubleshooting

**`command not found: uv`** — Install uv first: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`, `brew install uv`).

**`FileNotFoundError: data/raw/missions_cross_section.parquet`** — The upstream parquet comes from the sibling `NonProfitData` project at `../NonProfitData/`. For smoke testing without it, use `--config config/smoke.yaml` or set `data.allow_synthetic: true` in your config.

**Pipeline stops at G1/G2 gate** — This is expected. Fill the human coding template (G1) or confirm the production slate (G2) as described in the [operator loop](#quickstart), then re-run. No GPU/API work was wasted.

**Stage 03 crashes mid-run** — Just re-run. The annotation store is keyed by `(EIN2, source_id)`; completed rows are skipped automatically. Use `--no-resume` on `scripts/03_annotate.py` to start fresh.

**`uv sync` fails with Python version error** — Make sure your uv is up to date (`uv --version`, `uv self update`). The project requires Python 3.13 (pinned in `.python-version`).

**OpenAI API errors** — Check that `OPENAI_API_KEY` is set in `.env` and the key has access to the model IDs in your config/slate. Rate-limit errors trigger automatic retries with backoff (`annotation.max_retries: 5`).

**vLLM arm not found** — The bake-off skips the vLLM candidate gracefully if the server is unreachable. Ensure the vLLM server is running on the expected host/port and that the model ID in the config matches the served model.

---

## Appendix G: Legacy Pipeline

The original flat-script pipeline (`generate_training_data.py`, `split_data.py`, and the five Jupyter notebooks) has been moved verbatim to `archive/legacy-pipe/` for reference. It is **not executed** by the new pipeline and is preserved only for historical comparison.

---

## Authors

carobs9, chickymonkeys, JeanetBentzen
