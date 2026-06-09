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
    agreement.py       # LLM-vs-human agreement + Krippendorff α
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
git clone https://github.com/carobs9/BINARY-CLASSIFIER-MISSIONS.git
cd BINARY-CLASSIFIER-MISSIONS

# Install dependencies (uv manages Python 3.13 automatically)
uv sync

# Set secrets (OpenAI API key needed for the closed-reference annotator)
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Pipeline (run in order)

1. **Build sample** (`uv run python scripts/01_build_sample.py`)
   - Loads `missions_cross_section.parquet`, joins BMF for NTEE major group.
   - Computes the `Q` quality score and tiers (HIGH/MED/LOW).
   - Samples silver (~20k, HIGH+MEDIUM, stratified, positive-enriched) and gold (~400, boundary-case-rich).
   - Carves gold into prompt-dev / validation / frozen test.
   - Persists seeded `EIN2` manifests with inclusion probabilities.

2. **Prompt bake-off** (`uv run python scripts/02_bakeoff_prompts.py`)
   - Runs model-set × prompt-set on prompt-dev (~50) against human labels.
   - Selects the production model slate and retains 2–3 prompt variants.

3. **Annotate** (`uv run python scripts/03_annotate.py`)
   - Runs the full model × prompt matrix (OpenAI API + local vLLM endpoint).
   - Writes the long/tidy label store: one row per `(EIN2, source_id)`.
   - Resumes by `(EIN2, source_id)`; temp 0, fixed seed, guided JSON.

4. **Quality check & freeze** (`uv run python scripts/04_quality_check.py`)
   - Aggregates per-`EIN2` labels by majority vote.
   - LLM-vs-human agreement on validation ≥ 85% gate.
   - Krippendorff α on double-coded gold subset.
   - Versions and freezes the labelled file.

5. **End-to-end** (`uv run python scripts/run_pipeline.py`)
   - Chains 01→04 in a single invocation.

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
