# Configuration

The pipeline is **config-driven**: a YAML in `config/` is the source of truth for paths, seeds, thresholds, model slate, and sample sizes.

`load_config()` validates a YAML into the pydantic `BinaryClassifierConfig` (`binary_classifier/config.py`); `PathRegistry` (`binary_classifier/paths.py`) resolves it into `pathlib.Path`s. Stages consume only `cfg` + `registry`.

**The knobs and their defaults live in `config/religious_missions.yaml` and `config.py` — read those; don't mirror them here** (they change as the pipeline evolves).

## Key sections

### `paths`
- `raw_dir` — Upstream parquet inputs. This directory is local/cloud-managed and not committed.
- `interim_dir` — Intermediate artifacts: manifests, bake-off outputs, annotation stores, and monitor/canary inputs. This is the main cloud-symlink candidate in today's pre-DVC setup.
- `processed_dir` — Final artifacts. The committed pointer layer is `processed_dir / "gold"`, which holds `gold_to_code.csv` and `production_slate.json`; `silver_labels.csv` lives directly under `processed_dir` and is not committed. Stage 07 now also writes `evaluation/base_rate_precision.json`, stage 08 writes both `predictions/predictions.parquet` and per-organization `predictions/predictions_full.parquet`, and orchestrated runs write `run_manifest.json` at the processed root.
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

### `names`
- `panel_scope_values` — Trimmed, non-empty, unique panel classification values
  selected for the names arm. The shipped configuration selects `501C3 CHARITY`.
  The emitted panel population is `panel_scoped`; `bmf_only` remains the full BMF
  minus the full panel universe.

### `aggregation`
- `method` — Production Stage 04 aggregation method. This is intentionally majority-only.
- `comparison_arms` — Optional Dawid-Skene/CROWDLAB arms for Stage 11 sensitivity diagnostics only; they are not production continuation methods. These require the `diagnostics` optional extra (`crowd-kit`, `cleanlab`).

### Wave-6 contracts to remember

- **Per-organization estimand:** stage 09 is defined over raw `EIN2` organizations, not just the deduplicated inference rows. Canonical runs should therefore prefer `predictions_full.parquet` when it exists.
- **Triple labels:** stage 08 preserves `pred_label` as the recall-first prevalence label and also writes `pred_label_maxf1` and `pred_label_baserate` for release consumers.
- **One-shot frozen test:** stage 07 may enrich the frozen-test schema with `test_scores`, PR/ROC points, and multi-threshold confusion matrices, but the shared production artifact is only reopened in the controlled §7 UCloud rerun.

## Retasking (new classification task)

1. Copy `config/religious_missions.yaml` → `config/<task>.yaml`.
2. Set `entity`, `field`, `label_name` (plus any paths / model slate / sizes).
3. `uv run python scripts/run_pipeline.py --config config/<task>.yaml`.

No source edits, provided the upstream parquet exposes the chosen `field`.

## Built stages and script-only extensions

Stages 05–10 are built and wired into the orchestrator; stage 11 is still a
script-only helper. The decision record is in
`.agents/plans/we-work-on-the-floofy-wreath.md`, especially the appended
**Superseded decisions (June 2026)** memo, which replaced the old broad
training-size sweep and RoBERTa/DistilBERT encoder grid with stages 05–11.
- **Stage 05 — anchor sample:** add a representative anchor sample over the full
  frame, including LOW-quality rows, so population prevalence can be estimated
  without treating the HIGH+MEDIUM silver/gold frame as representative.
- **Stage 06 — training:** train on the full silver pool by default
  (`curve_fractions: [1.0]`), use soft vote-share targets by default with hard
  majority vote as the check, and compare DeBERTa-v3-base against ModernBERT-base
  plus TF-IDF/MiniLM baselines. Default arms are `[hard, class_weighted]`; `pruned`
  is opt-in. Label smoothing, focal/resampling, and confidence-weighted loss are
  intentionally skipped.
- **Stage 07 — evaluation:** keep minority-class precision/recall/F1, MCC,
  balanced accuracy, PR-AUC, bootstrap intervals, calibration reporting, and
  base-rate precision diagnostics. The frozen test stays one-shot.
- **Stage 08 — inference:** run the selected classifier over HIGH/MEDIUM rows,
  route LOW/bare-label rows through the rule layer, keep `EIN2`, persist
  model-version metadata with positive-class probabilities, and expand the
  deduplicated predictions back to raw `EIN2` rows in `predictions_full.parquet`.
- **Stage 09 — prevalence:** estimate per-organization population share over all
  nonprofits with PPI++ as the primary estimator for HIGH/MEDIUM and LOW
  classifier-routed rows, plus Rogan-Gladen for LOW rule-only rows. Cross-checks
  default to `[emq]` (vendored SLD/EMQ); KDEy via QuaPy is opt-in under the
  `quant` extra. Per-NTEE-stratum calibration is applied where prior-shift
  assumptions are fragile.
- **Stage 10 — visualization:** render the current figure suite from the
  orchestrator or standalone script: auditable n-gram log-odds bars, metric and
  calibration plots, score distributions, prevalence decomposition, rule
  validation intervals, quantification sensitivity, and subgroup performance.
- **Stage 11 — aggregation comparison:** script-only sensitivity diagnostics that compare
   majority vote with configured Dawid-Skene and CROWDLAB arms (requires the
   `diagnostics` extra) on the human validation set. Stage 04 production labels
   remain majority-only; Stage 11 does not continue production or activate a
   replacement aggregation method.

## Related

- [pipeline.md](pipeline.md) — how stages consume this config
