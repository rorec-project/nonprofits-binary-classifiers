# Configuration

The pipeline is **config-driven**: a YAML in `config/` is the source of truth for paths, seeds, thresholds, model slate, and sample sizes.

`load_config()` validates a YAML into the pydantic `BinaryClassifierConfig` (`binary_classifier/config.py`); `PathRegistry` (`binary_classifier/paths.py`) resolves it into `pathlib.Path`s. Stages consume only `cfg` + `registry`.

**The knobs and their defaults live in `config/religious_missions.yaml` and `config.py` — read those; don't mirror them here** (they change as the pipeline evolves).

## Key sections

### `paths`
- `gold_dir` — Directory for committed, human-coded gold artifacts (`gold_to_code.csv`, `production_slate.json`). Kept in git.
- `silver_dir` — Directory for the cloud-symlinked machine-labelled silver pool (`annotation_store.csv`, `bakeoff_labels.csv`). Not committed.

### `model_slate`
- `bakeoff_candidates` — List of `{id, provider, reasoning_effort?}` dicts. `provider` routes annotator construction (`openai` → closed API, `vllm` → local endpoint). `reasoning_effort` is forwarded only when present (e.g. `minimal` for GPT-5-class models).
- `production` — Default production model id. The actual slate stage 03 runs is the human-confirmed `production_slate.json` (gate G2); this key is the seed/default.

### `data`
- `allow_synthetic` — When `false` (default), a missing upstream parquet is a hard error. When `true`, a synthetic dataset is generated (with a loud warning and a `data_source="synthetic"` stamp) for local smoke-testing only.

### `qc`
- `agreement_threshold` — Minimum LLM-vs-human validation agreement required to freeze silver labels. Below this the QC gate blocks (raises / non-zero exit) and writes nothing.
- `abstain_on_fabricated_positive` — When `true`, any positive label that carries a fabricated evidence span is treated as an abstain (`None`) before aggregation.

## Retasking (new classification task)

1. Copy `config/religious_missions.yaml` → `config/<task>.yaml`.
2. Set `entity`, `field`, `label_name` (plus any paths / model slate / sizes).
3. `uv run python scripts/run_pipeline.py --config config/<task>.yaml`.

No source edits, provided the upstream parquet exposes the chosen `field`.

## Related

- [pipeline.md](pipeline.md) — how stages consume this config
