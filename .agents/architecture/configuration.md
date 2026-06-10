# Configuration

The pipeline is **config-driven**: a YAML in `config/` is the source of truth for paths, seeds, thresholds, model slate, and sample sizes.

`load_config()` validates a YAML into the pydantic `BinaryClassifierConfig` (`binary_classifier/config.py`); `PathRegistry` (`binary_classifier/paths.py`) resolves it into `pathlib.Path`s. Stages consume only `cfg` + `registry`.

**The knobs and their defaults live in `config/religious_missions.yaml` and `config.py` — read those; don't mirror them here** (they change as the pipeline evolves).

## Key sections

### `paths`
- `raw_dir` — Upstream parquet inputs. This directory is local/cloud-managed and not committed.
- `interim_dir` — Intermediate artifacts: manifests, bake-off outputs, annotation stores, and monitor/canary inputs. This is the main cloud-symlink candidate in today's pre-DVC setup.
- `processed_dir` — Final artifacts. The committed pointer layer is `processed_dir / "gold"`, which holds `gold_to_code.csv` and `production_slate.json`; `silver_labels.csv` lives directly under `processed_dir` and is not committed.
- `models_dir` — Future fine-tuned checkpoints and related model artifacts.

`PathRegistry` derives the higher-level artifact locations from those four roots. In practice, use registry properties such as `gold_coding_template`, `production_slate`, `silver_manifest`, and `annotation_store` instead of rebuilding paths by hand.

### Local layout and DVC note

- The intended on-disk layout is cookiecutter-data-science (`raw/` → `interim/` → `processed/` → `models/`; cookiecutter-data-science, drivendata).
- Today, cloud symlinks substitute for DVC for the heavy non-committed artifacts. The committed pointers are the gold files under `data/processed/gold/`.
- `PathRegistry.ensure_dirs()` creates output directories but does **not** create `raw_dir`; local setup still needs a real `data/raw/` plus the upstream parquet files.
- If you are migrating an older local setup, move the committed gold artifacts into `data/processed/gold/` and re-point any heavy silver-side cloud symlink so manifests, bake-off outputs, and annotation stores live under `data/interim/`.
- Future DVC migration is intentionally deferred. When adopted, the plan is to `dvc add data/raw/*.parquet`, configure a `dvc remote`, and let DVC manage its own cache links instead of layering extra intermediate symlinks on top (DVC docs: `add`, cache link types, remotes, external data).

### `model_slate`
- `bakeoff_candidates` — List of `{id, provider, reasoning_effort?}` dicts. `provider` routes annotator construction (`openai` → closed API, `vllm` → local endpoint). `reasoning_effort` is forwarded only when present (e.g. `minimal` for GPT-5-class models).
- `production` — Default production model id. The actual slate stage 03 runs is the human-confirmed `production_slate.json` (gate G2); this key is the seed/default.

### `data`
- `allow_synthetic` — When `false` (default), a missing upstream parquet is a hard error. When `true`, a synthetic dataset is generated (with a loud warning and a `data_source="synthetic"` stamp) for local smoke-testing only.

### `q_thresholds` and `sample_sizes`

- `q_thresholds` defines the HIGH / MEDIUM / LOW bands for the computable mission-quality rubric.
- The current sampling **frame** is HIGH+MEDIUM only: `Q >= 3.0` with the shipped defaults.
- LOW-quality rows are excluded from stage-01 sampling and are handled later by the high-precision rule layer. Do not treat silver/gold as population-representative over all nonprofits unless LOW is folded back in.
- `sample_sizes.gold` now includes the incremental `monitor` slice so prompt-dev, validation, and test do not shrink when drift monitoring is enabled.

### `qc`
- `agreement_threshold` — Legacy raw LLM-vs-human agreement benchmark. Logged for continuity, but no longer the sole freeze gate.
- `kappa_threshold` — Minimum Cohen's κ required to freeze silver labels. The shipped default matches the old operating point on the roughly balanced validation gate.
- `f1_ci_floor` — Minimum lower bound on the bootstrap confidence interval for minority-class F1 required to freeze silver labels.
- `abstain_on_fabricated_positive` — When `true`, any positive label that carries a fabricated evidence span is treated as an abstain (`None`) before aggregation.

## Retasking (new classification task)

1. Copy `config/religious_missions.yaml` → `config/<task>.yaml`.
2. Set `entity`, `field`, `label_name` (plus any paths / model slate / sizes).
3. `uv run python scripts/run_pipeline.py --config config/<task>.yaml`.

No source edits, provided the upstream parquet exposes the chosen `field`.

## Roadmap hooks (documented, not built)

- **Population share over all nonprofits:** add a representative anchor sample over the full frame, including LOW-quality rows, and estimate prevalence with PPI++ as the primary estimator (Angelopoulos et al. 2023; PPI++ arXiv:2311.01453), with SLD/EMQ and KDEy/DyS via QuaPy as cross-checks (Saerens, Latinne & Decaestecker 2002; QuaPy) plus per-NTEE-stratum calibration.
- **Fine-tuning:** default encoder target is DeBERTa-v3-base, with ModernBERT reserved for throughput-sensitive comparisons. Planned training upgrades include soft-label or confidence-weighted losses, label smoothing, and `bf16` on Blackwell B200 hardware (DeBERTa-v3 vs ModernBERT controlled study arXiv:2504.08716; NVIDIA).
- **Evaluation:** extend the current metric bundle with decision-curve / net-benefit analysis and ECE calibration reporting (Vickers & Elkin 2006).
- **Future weak-supervision arms:** uncertainty-weighted aggregation and classifier-assisted evidence verification remain gated comparison arms; each should only ship if it beats the current majority-vote baseline on the human held-out set.

## Related

- [pipeline.md](pipeline.md) — how stages consume this config
