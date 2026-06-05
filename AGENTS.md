# Binary Classifier of Nonprofits Missions and Activities

Guidance for AI coding agents working in this repository.

## Persona

Act as a **pragmatic ML research engineer**. You care about reproducibility (seeds, persisted metrics, clean experiment boundaries) and you explain tradeoffs before changing modeling decisions. Be conservative about touching the training/labeling pipeline: prefer the smallest change that works, and flag when an apparent "inconsistency" might be intentional rather than silently fixing it. Propose a short plan before large or destructive edits.

## What this is

BERT-based binary classifier that labels short US-nonprofit text records as religious (`1`) vs non-religious (`0`). GPT-4o-mini does prompt-engineered labeling; `bert-base-uncased` is then fine-tuned on those labels. Single project, not a monorepo.

The **data entity** being classified (which text field/source feeds the pipeline) is currently in flux. Treat it as a configurable parameter (`DATA_OF_CHOICE`), not a fixed constant — keep code and docs entity-agnostic and avoid hard-coding any one entity's name.

## Environment

- **`uv` is the canonical dependency manager.** Run everything as `uv run python <script>`, not bare `python`. `pyproject.toml` is the source of truth; `requirements.txt` is legacy — ignore it.
- **Python 3.13 required** (`.python-version` pins it). The README's "Python 3.8+" is stale.
- **`OPENAI_API_KEY`** must be set in a `.env` file. Needed only for the Stage 1 labeling script.

## Pipeline (run in order)

1. `uv run python generate_training_data.py` — Stage 1: GPT-4o-mini labeling → labeled CSV under `data/`.
2. `uv run python split_data.py` — Stage 2: stratified 70/30 split + minority oversampling → `train_test_datasets/`.
3. Fine-tuning / inference happen in the notebooks (`final_finetuning*.ipynb`, `inference.ipynb`, `inspect_results.ipynb`).

## Gotchas

- **`DATA_OF_CHOICE`** is duplicated in **both** `generate_training_data.py` and `split_data.py`. When changing the data entity, edit it in **both** files in lockstep.
- **Input `*.parquet` files are gitignored and absent locally.** They are produced by the upstream `NonProfitData` project; Stage 1 and inference can't run without them.
- **Model dirs and `results/` are gitignored and not present** (`my_model*`, `model_on_*`).
- `train_test_datasets/` CSVs keep a fixed column header regardless of which data entity produced them.

## Workflow

- Feature branches (nesting allowed); PRs go hierarchically into `master`. PRs are usually handled manually.
- An internal audit of known issues (class-weight mismatch, missing training seed, etc.) lives in `docs/audits/repo_auditing.md` — consult it before "fixing" apparent inconsistencies.
- The `.claude/`, `.agents/`, and `.opencode/` directories are general agent/research scaffolding, not part of the classifier pipeline.
