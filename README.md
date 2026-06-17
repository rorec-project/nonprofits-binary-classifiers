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

1. **AI bulk-labels, humans spot-check.** We do not hand-label 20,000 records. Instead, an ensemble of LLMs (OpenAI + open-weight models via vLLM) labels the bulk silver pool. A small, representative gold set (~450 records) is hand-coded by a human. The gold drives prompt selection, QC gates, and frozen-test evaluation. The silver is the training fuel; the gold is the truth meter.

2. **The model never sees the final exam.** The test split is drawn during stage 01 and is **frozen** thereafter. A human gate (G3 — `test_unlock.json`) is required before stage 07 can touch it, ensuring no accidental leakage during model iteration. The model is trained on silver + validation; evaluated on the locked test.

3. **We statistically *correct* counts for population estimates.** Classifier predictions are biased by the training distribution. We use PPI++ (Prediction-Powered Inference, Angelopoulos et al. 2023) as the primary prevalence estimator, with EMQ (vendored SLD implementation, Saerens 2002) as a single sensitivity cross-check. The anchor sample (stage 05, including LOW-quality rows) provides the labeled holdout needed to de-bias the full-corpus inference into a population-representative share with a bootstrap confidence interval.

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

Key dependency changes vs. the legacy stack: `pyarrow`, `python-dotenv` (replaces the misnamed `dotenv` package), `openai`, and `vllm` (via `--extra serve`).

### Smoke test (no upstream data)

```bash
uv run python scripts/run_pipeline.py --config config/smoke.yaml --stages 01
```

This generates a small synthetic dataset, stamps it `data_source="synthetic"`, and verifies the pipeline wiring. Intended for development only — never use synthetic data for production runs.

### Operator loop (the four gates)

The pipeline is driven by two human-coded artifacts and two human confirmations. Think of them as four gates the pipeline refuses to pass until a human signs off:

| Gate | Human action | What it unlocks |
|------|-------------|-----------------|
| **G1 — Labels** | Code `gold_to_code.csv` with strict `0/1` for every row | Stage 02 (bake-off), stage 04 (QC), stage 06 (train), stage 07 (evaluate) |
| **G2 — Slate** | Copy `proposed_slate.json` → `production_slate.json`, set `"confirmed": true`, commit | Stage 03 (full annotation) |
| **G4 — Anchor labels** | Code `anchor_to_code.csv` with strict `0/1` for every row | Stage 07 (evaluate), stage 09 (prevalence) |
| **G3 — Test unlock** | Create `test_unlock.json` with `"confirmed": true` + the selected checkpoint SHA | Stage 07 (evaluate) |

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

## The 9 Stages in Plain English

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

| Gate | Full name | Before stage(s) | Artifact | Required state |
|------|-----------|-----------------|----------|----------------|
| **G1** | Labels gate | 02, 04, 06, 07 | `data/processed/gold/gold_to_code.csv` | Every row in the needed split has a strict `0/1` `human_label` (no blanks, no other values) |
| **G2** | Slate gate | 03 | `data/processed/gold/production_slate.json` | `"confirmed": true` and lists the exact model IDs for production |
| **G4** | Anchor-labels gate | 07, 09 | `data/processed/gold/anchor_to_code.csv` | Every row has a strict `0/1` `human_label` |
| **G3** | Test-unlock gate | 07 | `data/processed/gold/test_unlock.json` | `"confirmed": true` + `checkpoint_sha256` matching the selected model + acceptance snapshot; the test split is never seen until this gate opens |

> **Why G3 is last:** the frozen test is the *final exam*. The model is selected and trained without ever touching it. Only after a human explicitly unlocks the test by recording the selected checkpoint SHA does evaluation run. This prevents any accidental leakage during model iteration.

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

| Criterion | Threshold | What it means |
|-----------|-----------|---------------|
| `min_pr_auc` | `0.90` | The classifier must distinguish religious vs. non-religious with at least 0.90 precision-recall AUC on the frozen test. |
| `min_minority_f1_ci_lower` | `0.70` | The lower bound of the 95% bootstrap CI for minority-class F1 must be ≥ 0.70. |
| `max_ece` | `0.05` | The expected calibration error (ECE) on the anchor out-of-fold scores must be ≤ 0.05. |

If any threshold is missed, the pipeline exits non-zero and prints guidance. The gate does **not** automatically retrain; it reports so the operator can decide whether to revise prompts, adjust the slate, or override the config.

### Output files

- `data/processed/evaluation/test_evaluation.json` — full frozen-test metrics: PR-AUC, minority-class precision/recall/F1, MCC, balanced accuracy, bootstrap 2000-resample CIs, length-binned subgroup reports, and calibration diagnostics (Platt + temperature scaling).
- `data/processed/prevalence/prevalence_report.json` — population prevalence estimate with PPI++ primary and EMQ cross-check, 95% bootstrap CI, and per-NTEE stratum estimates where available.

### Reading the prevalence report

Open `prevalence_report.json`. The top-level fields of interest are:

- `ppi_estimate` — the corrected population share of religious nonprofits.
- `ppi_ci_lower` / `ppi_ci_upper` — the 95% bootstrap confidence interval.
- `emq_estimate` — the EMQ (Expectation-Maximization with priors) sensitivity check.
- `low_tier_sensitivity` — bounds on the estimate if the LOW-quality rule-layer routing is systematically off.

> **Important caveat — LOW-tier missions:** The silver/gold sampling frame is HIGH+MEDIUM only (`Q >= 3.0`). LOW-quality records (bare labels, fragments) are excluded from stage 01 and handled by the rule layer at inference. The prevalence report folds them back in via the anchor sample, but the LOW-tier rate is a high-precision rule estimate, not a classifier score. Any claim about the *full* nonprofit population must include the LOW-tier sensitivity bounds, which the report provides when `prevalence.low_tier_sensitivity: true` in the config.

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
  data/
    load.py            # Cross-section load + EIN2→BMF NTEE-major-group join
    quality.py         # Computable Q rubric + HIGH/MED/LOW tiering + rule layer
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

### B1. Stage 01

- **Script:** `scripts/01_build_sample.py`
- **Inputs:** `data/raw/missions_cross_section.parquet`, `data/raw/bmf_unified_processed.parquet` (or synthetic if `data.allow_synthetic: true`).
- **Outputs:**
  - `data/interim/manifests/silver_manifest.csv` — `EIN2` list for the silver pool (HIGH+MEDIUM, `Q >= 3.0`).
  - `data/interim/manifests/gold_manifest.csv` — `EIN2` list for the gold set.
  - `data/processed/gold/gold_to_code.csv` — human coding template with columns `EIN2`, `split`, `text`, `human_label`.
- **Key options:** `--force` (regenerate from scratch, discarding human labels).
- **Caveat:** if you re-run stage 01 before coding starts, the gold split tags can reshuffle. Re-split before anyone codes, or regenerate the coding template from the new manifests.

### B2. Stage 02

- **Script:** `scripts/02_bakeoff_prompts.py`
- **Inputs:** `gold_to_code.csv` (prompt_dev split, G1), config `model_slate.bakeoff_candidates`.
- **Outputs:**
  - `data/interim/bakeoff/bakeoff_results.json` — scores for every candidate × prompt.
  - `data/interim/bakeoff/proposed_slate.json` — auto-picked, **unconfirmed** slate (`"confirmed": false`).
- **Key options:** `--prompts`, `--human-labels`, `--limit`, `--store-path`, `--output`.
- **Caveat:** the bake-off skips the vLLM arm gracefully if the server is unreachable. For a pure OpenAI bake-off, comment out the vLLM candidate in the config.
- **Adding open-weight models:** adding the Gemma (or any open-weight) model to the production ensemble requires no code change. Simply include its `id` in `production_slate.json` alongside the OpenAI models. The pipeline routes each candidate to its configured `provider` (`openai` or `vllm`) automatically. To remove it later, edit the slate.

### B3. Stage 03

- **Script:** `scripts/03_annotate.py`
- **Inputs:** `production_slate.json` (G2), silver manifests.
- **Outputs:** `data/interim/annotation_store.csv` — long/tidy label store keyed by `(EIN2, source_id)`.
- **Key options:** `--limit`, `--no-resume`, `--canary`, `--checkpoint-every`.
- **Caveat:** crashes are resumable by `(EIN2, source_id)`; completed rows are skipped automatically. Use `--no-resume` to start fresh.

### B4. Stage 04

- **Script:** `scripts/04_quality_check.py`
- **Inputs:** annotation store, `gold_to_code.csv` (validation split, G1).
- **Outputs:** `data/processed/silver_labels.csv` — frozen majority-vote labels, excluding all gold `EIN2` rows.
- **QC thresholds:**
  - `qc.kappa_threshold: 0.70` (Cohen's kappa against human validation, chance-corrected; reproduces the old ~0.85 raw-agreement operating point on the roughly balanced validation split).
  - `qc.f1_ci_floor: 0.70` (lower bound of 95% bootstrap CI for minority-class F1).
- **Key options:** `--human-validation`, `--store-path`, `--output`.
- **Caveat:** if the QC gate misses the threshold, it **blocks** — exits non-zero, does not write a frozen output, and prints guidance to revise prompts and re-label. Raw LLM-vs-human validation agreement is still logged for continuity, but the freeze decision is driven by chance-corrected κ/α reporting plus the minority-F1 CI floor.

### B5. Stage 05

- **Script:** `scripts/05_build_anchor.py`
- **Inputs:** full frame (all Q tiers, including LOW).
- **Outputs:** `data/processed/gold/anchor_to_code.csv` — human coding template for the anchor sample.
- **Key options:** `--force`.
- **Caveat:** the anchor must be coded before stages 07 and 09 (G4). Without it, prevalence estimates on LOW-quality rows cannot be validated.

### B6. Stage 06

- **Script:** `scripts/06_train.py`
- **Inputs:** frozen `silver_labels.csv`, config `training` section.
- **Outputs:** `data/models/selection_report.json`, checkpoint directories under `data/models/`.
- **Training defaults (simplified):**
  - **Arms:** `[hard, class_weighted]` — the `pruned` arm is dropped by default (soft targets already down-weight the disagreement band; see arXiv:2605.20642). `pruned` remains opt-in via config override.
  - **Targets:** soft vote-share labels are the default target; hard majority-vote labels are retained as a check arm.
  - **Learning curve:** `[1.0]` — only the full-data run is produced by default; the `{25%, 50%, 100%}` sweep is dropped.
  - **Final seeds:** 5 seeds (`[42, 43, 44, 45, 46]`) for reported variance.
  - **Encoders:** DeBERTa-v3-base (primary) vs. ModernBERT-base (comparison), plus TF-IDF/MiniLM baselines. RoBERTa, DistilBERT, label smoothing, focal/resampling, and confidence-weighted loss are intentionally skipped.
- **Key options:** `--baselines-only`, `--sweep`, `--final`, `--encoder`, `--subset`, `--epochs`, `--seeds`, `--limit`.
- **Orchestrator note:** `run_pipeline.py` runs stage 06 as a two-phase sweep → final automatically, so `--final` is not needed on the orchestrator. The orchestrator forwards `--config`, `--stages`, `--annotate-limit` (stage 03), `--infer-limit` (stage 08), and `--force` (stages 01/05) to the individual stage scripts.

### B7. Stage 07

- **Script:** `scripts/07_evaluate.py`
- **Inputs:** frozen test split (G1 + G3), selected checkpoint, `anchor_to_code.csv` (G4).
- **Outputs:** `data/processed/evaluation/test_evaluation.json`.
- **Metrics reported:** minority-class precision/recall/F1, MCC, balanced accuracy, PR-AUC, bootstrap 2000-resample CIs, length-binned subgroup reports, and calibration diagnostics (Platt + temperature scaling). **Decision-curve analysis was dropped** (orthogonal to a prevalence study).
- **Acceptance thresholds:** `min_pr_auc: 0.90`, `min_minority_f1_ci_lower: 0.70`, `max_ece: 0.05`.
- **Threshold policy:** `precision_floor` (default 0.80) — the operating threshold is selected so that precision on the validation set stays ≥ 0.80.
- **Calibration:** Platt + temperature scaling are both compared and deployed; Platt is retained as the default calibration method. Isotonic regression is intentionally excluded (overfits at anchor n=500). The acceptance gate is `max_ece`-only for this design pass.

### B8. Stage 08

- **Script:** `scripts/08_infer.py`
- **Inputs:** full corpus, selected checkpoint, rule layer configuration.
- **Outputs:** `data/processed/predictions/predictions.parquet` — `EIN2` + positive-class probabilities + model-version metadata.
- **Routing:** HIGH/MEDIUM rows go to the classifier; LOW/bare-label rows go to the high-precision rule layer; rule-layer abstentions fall through to the classifier if `inference.rule_ambiguous_to_classifier: true`.
- **Key options:** `--limit`.

### B9. Stage 09

- **Script:** `scripts/09_prevalence.py`
- **Inputs:** predictions, anchor sample labels (G4), design weights.
- **Outputs:** `data/processed/prevalence/prevalence_report.json`.
- **Methods:**
  - **Primary:** PPI++ (Angelopoulos et al. 2023; PPI++ arXiv:2311.01453).
  - **Cross-check:** EMQ (vendored SLD, Saerens 2002). KDEy via QuaPy is opt-in under the `quant` extra.
- **Settings:** 95% bootstrap CI (`alpha: 0.05`), design-weighted estimation, per-NTEE stratum reporting where `ntee_min_n: 10` is satisfied, and LOW-tier sensitivity bounds when `low_tier_sensitivity: true`.

### B10. Stage 10 (script-only, not orchestrated)

- **Script:** `scripts/10_visualize.py`
- **Outputs:** PNG and SVG figures under `data/processed/viz/`.
- **Caveat:** script-only renderer; skips missing inputs gracefully. Uses signed n-gram log-odds bars instead of word clouds (reproducible, statistically interpretable, and avoids a new word-cloud dependency).

### B11. Stage 11 (script-only, not orchestrated)

- **Script:** `scripts/11_aggregation_compare.py`
- **Outputs:** `data/interim/aggregation_compare.json`.
- **Purpose:** sensitivity diagnostics comparing majority vote with optional Dawid-Skene arms on the human validation set. `crowd-kit` and `cleanlab` are diagnostic-only (stage 11) and behind the `diagnostics` extra. Stage 04 production labels remain majority-only; stage 11 does not activate a replacement aggregation method.

---

## Appendix C: Methodology & Citations

### C1. Simplifications and intentional skips (this design pass)

The project follows a **one principled primary method per concern + minimal robustness** principle. Tertiary and diagnostic machinery is pushed to optional extras or omitted entirely.

| What | Default | Rationale |
|------|---------|-----------|
| **Label-aggregation diagnostics** | `crowd-kit`, `cleanlab` → optional `diagnostics` extra | Dawid-Skene and CROWDLAB are diagnostic-only (stage 11). Stage 04 production is intentionally majority-only. |
| **Quantification cross-checks** | `[emq]` (EMQ / vendored SLD) | EMQ is the single sensitivity check that matches the project's research goals. KDEy via QuaPy is opt-in under the `quant` extra. |
| **Training arms** | `[hard, class_weighted]` | The `pruned` arm is dropped by default. Soft vote-share targets already down-weight the disagreement band that `pruned` / cleanlab removes (arXiv:2511.14117; arXiv:2605.20642). `pruned` remains opt-in. |
| **Learning curve** | `[1.0]` | Only the full-data run is produced. The `{25%, 50%, 100%}` sweep is dropped. |
| **Decision-curve analysis** | Dropped | Vickers-Elkin net-benefit is orthogonal to a prevalence study; it does not improve the calibrated population estimate. |
| **Calibration** | Platt + temperature (retained) | Both are compared and the better is deployed. Isotonic regression is excluded because it overfits at anchor n=500. |
| **Final seeds** | 5 seeds (`[42, 43, 44, 45, 46]`) | Retained. Research mandates ≥5 seeds for reported variance. |
| **Acceptance gate** | `max_ece`-only | Brier/log-loss gating is future work and out of scope for this design pass. |

### C2. What was *not* simplified (research fidelity)

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

- Angelopoulos, A. N., et al. (2023). *Prediction-Powered Inference*. arXiv:2311.01453. — Primary prevalence estimator (PPI++).
- Saerens, M., et al. (2002). *Adjusting the Outputs of a Classifier to New a Priori Probabilities*. — Basis for EMQ / SLD cross-check.
- Landis, J. R., & Koch, G. G. (1977). *The Measurement of Observer Agreement for Categorical Data*. — Kappa interpretation scale.
- SILICON (arXiv:2412.14461). — Agreement benchmarking in weak-supervision contexts.
- Variance-Aware protocol (arXiv:2601.02370). — Minority-F1 CI floor motivation.
- Disagreement-band down-weighting (arXiv:2511.14117; arXiv:2605.20642). — Rationale for dropping the `pruned` arm from defaults.
- cookiecutter-data-science, drivendata. — Data layout convention.

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

carobs9, chickymonkeys
