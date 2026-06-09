# Binary Classification of Religious vs. Non-Religious Nonprofit Missions

A reproducible, config-driven binary text classifier for US nonprofit missions. The pipeline labels short text records as religious (`1`) vs. non-religious (`0`) using LLM-as-primary annotation, aggregates noisy labels across a multi-model × multi-prompt ensemble, and fine-tunes modern encoder models on the resulting silver dataset. Designed for extensibility beyond the religious task to pregnancy centers, education, international aid, and other nonprofit sectors.

> **Status:** Re-engineering in progress (points 1–2 of the roadmap). The legacy flat-script pipeline has been moved to `archive/legacy-pipe/` and is preserved for reference but not executed.

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
    aggregate.py       # Majority vote + crowd-kit / cleanlab hooks
  qc/
    agreement.py       # LLM-vs-human agreement + full sklearn metric bundle
scripts/
  01_build_sample.py   # Stage 2.1: construct silver (~20k) + gold (~400) + splits
  02_bakeoff_prompts.py # Stage 2.2: model×prompt bake-off on prompt-dev
  03_annotate.py       # Stage 2.3: full model×prompt matrix labeling run
  04_quality_check.py  # Stage 2.4: aggregation + QC gate (≥85% agreement)
  run_pipeline.py      # Orchestrator: chains 01→04
config/
  religious_missions.yaml   # First task config (entity=missions, field=LONGEST_MISSION)
```

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

- Silver `EIN2` manifests to the configured silver directory.
- `gold_to_code.csv` to the configured gold directory (default `data/processed/train_test_datasets/gold/gold_to_code.csv`).

Open `gold_to_code.csv`. It contains four columns: `EIN2`, `split`, `text`, `human_label`. Fill **every** `human_label` cell with a strict `0` or `1`. Do not leave blanks and do not use other values. Commit the file.

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

- `results/bakeoff_results.json` — scores for every candidate.
- `results/proposed_slate.json` — an auto-picked, **unconfirmed** slate (`"confirmed": false`).

Review the scores in `bakeoff_results.json`.

### Step 4 — Human checkpoint 2: confirm the production slate

Copy `proposed_slate.json` into the gold directory as `production_slate.json`:

```bash
cp results/proposed_slate.json data/processed/train_test_datasets/gold/production_slate.json
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

- **Stage 03** labels the full silver pool using only the models confirmed in `production_slate.json`. It writes `annotation_store.csv` (resumable by `(EIN2, source_id)`).
- **Stage 04** aggregates the labels and measures agreement against the human-coded `validation` split.

> **Gate 1 (G1, again):** Stage 04 also checks that every `validation` row in `gold_to_code.csv` has a valid `0/1` label.

If agreement is below the configured threshold (default 0.85), the QC gate **blocks** — it exits non-zero, prints guidance to revise prompts and re-label, and does **not** write a frozen output. If agreement is above the threshold, the silver set is frozen and versioned.

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
- **Weak-supervision-ready:** the long/tidy label store is designed for crowd-kit (Dawid-Skene) and cleanlab (CROWDLAB) as drop-in comparison arms.
- **Rule layer:** LOW-quality / bare-label missions are handled by a high-precision rule layer at inference, not dropped (protects prevalence).
- **EIN2 everywhere:** the upstream join key is carried through every artifact.

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

## License

This project is part of research on economics of religion and nonprofit classification.

## Author

carobs9

## Acknowledgments

- Domain knowledge from economics of religion scholarship
- Encoder models from Hugging Face Transformers
- Open-source annotation stack (vLLM, crowd-kit, cleanlab)
- UCloud / DeiC Interactive HPC for GPU compute
