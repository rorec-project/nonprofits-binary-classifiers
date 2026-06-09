# Configuration

The pipeline is **config-driven**: a YAML in `config/` is the source of truth for paths, seeds, thresholds, model slate, and sample sizes.

`load_config()` validates a YAML into the pydantic `BinaryClassifierConfig` (`binary_classifier/config.py`); `PathRegistry` (`binary_classifier/paths.py`) resolves it into `pathlib.Path`s. Stages consume only `cfg` + `registry`.

**The knobs and their defaults live in `config/religious_missions.yaml` and `config.py` — read those; don't mirror them here** (they change as the pipeline evolves).

## Retasking (new classification task)

1. Copy `config/religious_missions.yaml` → `config/<task>.yaml`.
2. Set `entity`, `field`, `label_name` (plus any paths / model slate / sizes).
3. `uv run python scripts/run_pipeline.py --config config/<task>.yaml`.

No source edits, provided the upstream parquet exposes the chosen `field`.

## Related

- [pipeline.md](pipeline.md) — how stages consume this config
