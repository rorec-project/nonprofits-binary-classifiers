# Binary Classification of Religious vs. Non-Religious Nonprofit Missions

A reproducible, config-driven binary text classifier for US nonprofit missions. The pipeline labels short text records as religious (`1`) vs. non-religious (`0`) using LLM-as-primary annotation, aggregates noisy labels across a multi-model × multi-prompt ensemble, and is designed to fine-tune modern encoder models on the resulting silver dataset in a later stage. Designed for extensibility beyond the religious task to pregnancy centers, education, international aid, and other nonprofit sectors.

> **Status:** Stages 01–04 are implemented: sampling, bake-off, annotation, and QC/freeze. Stages 05–11 are roadmap work: anchor sample, training, evaluation, inference-at-scale, prevalence, visualization, and aggregation comparison. The legacy flat-script pipeline has been moved to `archive/legacy-pipe/` and is preserved for reference but not executed.

## Architecture

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
    aggregate.py       # Majority vote default; other aggregation arms are gated
  qc/
    agreement.py       # Validation gate: κ/α reporting + minority-F1 CI floor
scripts/
  01_build_sample.py   # Stage 01: construct silver + gold + prompt/validation/test/monitor manifests
  02_bakeoff_prompts.py # Stage 02: model×prompt bake-off on prompt-dev
  03_annotate.py       # Stage 03: full model×prompt matrix labeling over silver ∪ gold
  04_quality_check.py  # Stage 04: QC gate + freeze silver-only labels
  run_pipeline.py      # Orchestrator: chains 01→04
config/
  religious_missions.yaml   # First task config (entity=missions, field=LONGEST_MISSION)
```

## Data layout

The pipeline follows the cookiecutter-data-science layout (`raw/` → `interim/` → `processed/` → `models/`; [cookiecutter-data-science, drivendata](https://cookiecutter-data-science.drivendata.org/)):

- `data/raw/` — immutable upstream parquet inputs. Not committed.
- `data/interim/` — manifests, bake-off outputs, and annotation stores. Not committed.
- `data/processed/` — final artifacts. `data/processed/gold/` is the committed pointer layer (`gold_to_code.csv`, `production_slate.json`); `silver_labels.csv` is not committed.
- `data/models/` — future fine-tuned checkpoints. Not committed.

Today, local/cloud symlinks stand in for DVC for every heavy, non-committed location (`raw/`, `interim/`, `processed/silver_labels.csv`, `models/`). The committed pointers are the small gold artifacts in `data/processed/gold/`.

> **Future DVC migration:** when the project adopts DVC, add the real upstream parquet files with `dvc add data/raw/*.parquet` and configure a `dvc remote` (DVC docs: `add`, cache link types, remotes, external data). Avoid stacking ad-hoc intermediate directory symlinks on top of DVC-managed paths because DVC manages its own cache links.

## Environment

- **Python:** 3.13 (pinned in `.python-version`)
- **Dependency manager:** `uv` — `pyproject.toml` is the source of truth; `requirements.txt` is legacy and ignored.
- **GPU target:** UCloud `gpu-nvidia-b200` (8× B200 192 GB) for open-weight annotation and training; CPU-only stages run locally.

## Installation

```bash
# Clone the repository
git clone https://github.com/rorec-project/nonprofits-binary-classifiers.git
cd nonprofits-binary-classifiers

# Install dependencies (uv manages Python 3.13 automatically)
uv sync

# Set secrets (OpenAI API key needed for the closed-reference annotator)
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Local data setup

Create the real local directories before your first non-synthetic run:

```bash
mkdir -p data/raw data/interim data/models data/processed/gold
```

- `PathRegistry.ensure_dirs()` creates output directories such as `data/interim/`, `data/models/`, and `data/processed/gold/`, but it does **not** create `data/raw/` for you.
- Stage 01 hard-fails if `data/raw/missions_cross_section.parquet` or `data/raw/bmf_unified_processed.parquet` is missing, unless you set `data.allow_synthetic: true` for smoke tests.
- Move any existing committed gold artifacts into `data/processed/gold/` before running the human checkpoints.
- If your heavy silver artifacts still live behind the old processed-tree cloud symlink, re-point that local symlink so manifests, bake-off outputs, and annotation stores live under `data/interim/` instead. Moving gold alone does not move the large silver-side artifacts.

## Operator Guide: Two-Checkpoint Pipeline Loop

The pipeline is config-driven and gated by two human checkpoints. The main entry point is `uv run python scripts/run_pipeline.py --stages <stages>`. The default config is `config/religious_missions.yaml`.

### Workflow overview

```
LOCAL (CPU):  01 build_sample → gold_to_code.csv
               └─[HUMAN 1: code human_label 0/1]─┘
UCLOUD (GPU): 02 bake-off → bakeoff_results.json + proposed_slate.json
               └─[HUMAN 2: review scores → confirm production_slate.json]─┘
UCLOUD (GPU): 03 annotate → 04 QC gate (validation labels) → freeze
```

### Step 1 — Build the sample and code the gold set (local, CPU)

Run stage 01 to generate the silver sample and the gold coding template:

```bash
uv run python scripts/run_pipeline.py --stages 01
```

This writes:

- Silver `EIN2` manifests to `data/interim/manifests/`.
- `gold_to_code.csv` to `data/processed/gold/gold_to_code.csv`.

Open `gold_to_code.csv`. It contains four columns: `EIN2`, `split`, `text`, `human_label`. Fill **every** `human_label` cell with a strict `0` or `1`. Do not leave blanks and do not use other values. Commit the file. The template now includes `prompt_dev`, `validation`, `test`, and `monitor` rows.

> **Re-split caveat:** if you re-run stage 01 before coding starts, the gold split tags can reshuffle. Re-split before anyone codes, or regenerate the coding template from the new manifests before continuing.

> **Gate 1 (G1 — labels gate):** The pipeline will refuse to run stages 02 or 04 if any `human_label` is missing, blank, or non-`0/1`. It exits gracefully with a clear message so no GPU work is wasted.

If you need to regenerate the template from scratch (discarding all human labels), use:

```bash
uv run python scripts/run_pipeline.py --stages 01 --force
```

### Step 2 — Optional: serve the open-weight arm on UCloud (GPU)

If your config includes the open-weight comparison arm (e.g. Gemma served via vLLM), start the server on UCloud before running stage 02:

```bash
# On UCloud (see docs/RUNNING_ON_UCLOUD.md)
vllm serve <gemma-model-id> --tensor-parallel-size 8
```

The bake-off will skip the vLLM arm gracefully if the server is unreachable. For a pure OpenAI bake-off, you can comment out the vLLM candidate in the config.

### Step 3 — Run the bake-off (stage 02)

With the gold set fully coded, run:

```bash
uv run python scripts/run_pipeline.py --stages 02
```

This scores every configured `bakeoff_candidate` × `prompt` against the human-coded `prompt_dev` split. It writes:

- `data/interim/bakeoff/bakeoff_results.json` — scores for every candidate.
- `data/interim/bakeoff/proposed_slate.json` — an auto-picked, **unconfirmed** slate (`"confirmed": false`).

Review the scores in `bakeoff_results.json`.

### Step 4 — Human checkpoint 2: confirm the production slate

Copy `proposed_slate.json` into the gold directory as `production_slate.json`:

```bash
cp data/interim/bakeoff/proposed_slate.json data/processed/gold/production_slate.json
```

Edit it to contain the exact model IDs you want in production, and set:

```json
"confirmed": true
```

Commit `production_slate.json`.

> **Gate 2 (G2 — slate gate):** The pipeline will refuse to run stage 03 unless a valid `production_slate.json` with `"confirmed": true` exists in the gold directory. If you request `--stages 02,03,04` together, stage 02 will run first, then the pipeline exits gracefully at G2 so you can review the results before committing to the full annotation run.

### Step 5 — Annotate and QC gate (stages 03–04)

With the slate confirmed, run:

```bash
uv run python scripts/run_pipeline.py --stages 03,04
```

- **Stage 03** labels the full silver pool plus the gold holdouts used by QC/canary checks, using only the models confirmed in `production_slate.json`. It writes `annotation_store.csv` (resumable by `(EIN2, source_id)`).
- **Stage 04** aggregates the labels, measures agreement against the human-coded `validation` split, and writes a frozen `silver_labels.csv` that excludes every gold `EIN2` from the training pool.

> **Gate 1 (G1, again):** Stage 04 also checks that every `validation` row in `gold_to_code.csv` has a valid `0/1` label.

If the QC gate misses the configured chance-corrected agreement threshold or the minority-class F1 confidence-floor threshold, it **blocks** — it exits non-zero, prints guidance to revise prompts and re-label, and does **not** write a frozen output. Raw agreement is still logged for continuity, but the freeze decision is driven by κ/α reporting plus the minority-F1 CI floor (Landis & Koch 1977; SILICON, arXiv:2412.14461; Variance-Aware protocol, arXiv:2601.02370).

### Step 6 — Enabling the open-weight model for production

Adding the Gemma (or any open-weight) model to the production ensemble requires no code change. Simply include its `id` in `production_slate.json` alongside the OpenAI models. The pipeline routes each candidate to its configured `provider` (`openai` or `vllm`) automatically. To remove it later, edit the slate.

### Smoke testing without upstream data

If the upstream `missions_cross_section.parquet` is not present (e.g. for local testing or CI), set the following in your config YAML:

```yaml
data:
  allow_synthetic: true
```

Stage 01 will generate a small synthetic dataset, stamp it `data_source="synthetic"`, and proceed. This is intended for development smoke tests only — never use synthetic data for production runs.

### End-to-end single command

Once the pipeline is fully set up and both human checkpoints are satisfied, you can run the entire chain:

```bash
uv run python scripts/run_pipeline.py --stages 01,02,03,04
```

## Key design decisions

- **Entity-agnostic:** `entity` and `field` are config parameters, not hard-coded.
- **Reproducibility:** one global `SEED`; every stochastic step is seeded.
- **Weak-supervision-ready:** the long/tidy label store keeps model×prompt votes so future aggregation arms can be evaluated against the human holdout before adoption.
- **Rule layer:** LOW-quality / bare-label missions are handled by a high-precision rule layer at inference, not dropped.
- **EIN2 everywhere:** the upstream join key is carried through every artifact.

## Sampling frame vs. population

The current sampling frame is the HIGH+MEDIUM quality strata only: `Q >= 3.0`. LOW-quality records are excluded from stage-01 sampling and are handled later by the rule layer at inference. That means the sampled frame is **not** the full nonprofit population. Any population-share claim over all nonprofits must fold LOW back in explicitly using the rule-layer label rate multiplied by the LOW-count mass.

This framing is deliberate: the current pipeline optimizes annotation and QC on text that is informative enough for LLM review, while keeping a clear hook for the later all-nonprofits prevalence estimator.

## Roadmap (not built yet)

The detailed decision record lives in
`.agents/plans/we-work-on-the-floofy-wreath.md`, especially the appended
**Superseded decisions (June 2026)** memo. That memo replaces the old broad
training sweep with the staged plan below.

- **Stage 05 — anchor sample:** draw a representative anchor sample at the real
  prior over the full frame, including LOW-quality rows, so later prevalence
  estimates do not treat the HIGH+MEDIUM silver/gold frame as population
  representative.
- **Stage 06 — training:** train on the full silver pool by default and keep only
  a `{25%, 50%, 100%}` one-seed documentation curve. Use soft vote-share labels
  as the default target, retain hard majority vote as the check, and compare
  DeBERTa-v3-base with ModernBERT-base plus TF-IDF/MiniLM baselines. RoBERTa,
  DistilBERT, label smoothing, focal/resampling, and confidence-weighted loss
  are intentionally skipped.
- **Stage 07 — evaluation:** report minority-class precision/recall/F1, MCC,
  balanced accuracy, PR-AUC, bootstrap intervals, and calibration metrics, then
  add decision-curve / net-benefit analysis where useful.
- **Stage 08 — inference:** run the selected classifier over HIGH/MEDIUM rows,
  route LOW/bare-label rows through the rule layer, keep `EIN2`, and persist
  model-version metadata with positive-class probabilities.
- **Stage 09 — prevalence:** estimate population share over all nonprofits with
  PPI++ as the primary estimator (Angelopoulos et al. 2023; PPI++
  arXiv:2311.01453), with SLD/EMQ and KDEy/DyS via QuaPy as cross-checks and
  per-NTEE-stratum calibration where prior-shift assumptions are fragile.
- **Stage 10 — visualization:** produce auditable n-gram log-odds bars plus
  metric and calibration plots.
- **Stage 11 — aggregation comparison:** evaluate cleanlab/crowd-kit,
  uncertainty-weighted aggregation, and classifier-assisted evidence-checking as
  gated comparison arms; adopt only if they beat majority vote on the human
  held-out data.

## UCloud runtime

See `docs/RUNNING_ON_UCLOUD.md` for:

- Job submission (`gpu-nvidia-b200` SKU)
- `/work` persistence rules
- `utils/init.sh` setup
- vLLM serve on localhost (`--tensor-parallel-size 8`)
- SSH access and secret management

## Legacy pipeline

The original flat-script pipeline (`generate_training_data.py`, `split_data.py`, and the five Jupyter notebooks) has been moved verbatim to `archive/legacy-pipe/` for reference. It is not executed by the new pipeline.

## Dependencies

See `pyproject.toml` for the full list. Key additions vs. the legacy stack:

- `pyarrow`, `crowd-kit`, `cleanlab` — label aggregation and noise detection
- `openai`, `vllm` — LLM annotation (closed API + open-weight serving)
- `python-dotenv` — secret loading (replaces the misnamed `dotenv` package)

## Author

carobs9, chickymonkeys
