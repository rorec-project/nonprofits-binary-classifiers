# Gotchas and Situational Notes

## Data layout

- **`data/` is a real repo directory, not one giant symlink.** The intended layout is cookiecutter-style `data/raw`, `data/interim`, `data/processed`, `data/models`. In local setups, only the heavy non-committed locations may be cloud-backed symlinks.
- **Cloud symlinks currently stand in for DVC.** Treat `raw/`, `interim/`, `processed/silver_labels.csv`, and `models/` as local/cloud-managed storage. The small committed pointer layer is `data/processed/gold/`, which should contain `gold_to_code.csv` and the human-confirmed `production_slate.json`.

## Local setup

- **Local setup is partly manual.** `PathRegistry.ensure_dirs()` creates output directories but not `data/raw/`; stage 01 hard-fails on missing upstream parquet unless `data.allow_synthetic: true`. If a local setup still points the heavy silver-side artifacts at an old processed-tree symlink, re-point that storage under `data/interim/` before documenting or debugging path issues.

## Upstream data

- **Upstream `*.parquet` inputs are gitignored and absent locally.** They are produced by the sibling `NonProfitData` project, expected at `../NonProfitData`. Stages that read parquet can't run without it.

## Artifact locations

- **Manifests** live under `interim_dir/manifests/` — they are `EIN2` lists + sampling metadata, not text/labels. The text is re-joined from the upstream parquet by `EIN2`.
- **Bake-off artifacts** live under `interim_dir/bakeoff/` — scores + proposed slate. All interim pipeline outputs (manifests, bake-off, annotation stores) share the cloud symlink.

## Sampling frame

- **The sampled frame is HIGH+MEDIUM only (`Q >= 3.0`).** LOW-quality rows are excluded from stage-01 sampling and handled by the rule layer later. Do not describe silver/gold as population-representative over all nonprofits without folding LOW back in.

## Key rules

- **`EIN2` is the upstream join key** — carry it through every artifact.
- **`archive/legacy-pipe/` is reference-only.** Don't run it, and don't "fix" it to match the new pipeline.
- **The `.claude/`, `.agents/`, and `.opencode/` directories** are general agent/research scaffolding, not part of the classifier pipeline.
- **Roadmap facts** live in the README and configuration docs. Future DVC migration, prevalence estimation, encoder choice, and evaluation upgrades are documented there. Keep AGENTS as pointers, not the canonical long-form roadmap.

## Stage 08 inference fixes (Jul 2026)

Two fixes were applied to `src/binary_classifier/inference/predict.py`:

- **max_length alignment:** The tokenizer call in `_load_checkpoint_predictor` now passes the encoder-specific `max_length` from the training config (default 256). Previously it defaulted to the tokenizer's `model_max_length` (512 for DeBERTa), feeding out-of-distribution token positions the model never saw during training.
- **Precision override for FP32-trained encoders:** `resolve_device_precision` now checks whether the selected encoder has an explicit `precision: fp32` in the training config and overrides inference precision accordingly. This prevents silent degradation under BF16 autocast for models (like DeBERTa-v3-base) that were trained with FP32.

## Related

- [../pipeline/pipeline.md](../pipeline/pipeline.md) — stage I/O detail
- [../pipeline/configuration.md](../pipeline/configuration.md) — config knobs and local layout
