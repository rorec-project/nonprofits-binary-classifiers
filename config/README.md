# Configuration Reference

## Overview

This project is **config-driven**: a single YAML file in `config/` is the source of truth for all pipeline behavior -- paths, seeds, thresholds, model slates, sample sizes, and hyperparameters. No pipeline logic inspects hard-coded constants; every stage reads from the same validated Pydantic object.

There are two configuration files shipped with the repository:

| File                             | Purpose                                                                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `config/religious_missions.yaml` | Production configuration for the "religious vs. non-religious missions" binary classification task.                             |
| `config/smoke.yaml`              | Minimal override for smoke testing: synthetic data, smaller sample sizes, a tiny random encoder, relaxed acceptance thresholds. |

Both are consumed identically:

```
uv run python scripts/run_pipeline.py --config config/religious_missions.yaml
uv run python scripts/run_pipeline.py --config config/smoke.yaml
```

The `--config` argument defaults to `config/religious_missions.yaml` if omitted.

To retask the pipeline for a new classification task (e.g. "activities" instead of "missions"), copy `religious_missions.yaml` to a new file, adjust `entity`, `field`, `label_name`, and any other knobs, then point `run_pipeline.py --config` at the new file. No source edits are required, provided the upstream parquet exposes the chosen text field.

---

## What you actually decide

This checklist covers the decisions you must make before a run. Everything else can be left at default.

### 1. The retasking triple

These three fields define the semantic task. Change them only when retasking the pipeline for a new classification problem.

| Decision             | YAML key     | Default           | If unsure                                                 |
| -------------------- | ------------ | ----------------- | --------------------------------------------------------- |
| Entity name          | `entity`     | `missions`        | Keep the default unless the upstream parquet changes      |
| Text column          | `field`      | `LONGEST_MISSION` | Must match a column in the upstream parquet; do not guess |
| Positive class label | `label_name` | `religious`       | Keep the default unless the semantic label changes        |

### 2. Confirm the model slate

The default slate is calibrated for short nonprofit text. Override only if you have validated a better candidate.

| Decision            | YAML key                         | Default                                                            | If unsure                                                                   |
| ------------------- | -------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Production model    | `model_slate.production`         | `gpt-5-mini`                                                       | Use the default                                                             |
| Bake-off candidates | `model_slate.bakeoff_candidates` | `gpt-4o-mini`, `gpt-5-mini`, `gpt-5-nano`, `google/gemma-3-27b-it`, `deepseek-ai/DeepSeek-V4-Flash` | Use the default; comment out either/both `vllm` entries if no vLLM server is available |

### 3. The four human gates

These artifacts are produced by the pipeline and confirmed by a human. They cannot be skipped; downstream stages exit gracefully if a gate is incomplete.

| Gate               | Artifact                                    | What you decide                                                         | If unsure                                                                                              |
| ------------------ | ------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| G1 (labels)        | `data/processed/gold/gold_to_code.csv`      | Ensure every row in the required split has a complete `0/1` human label | Do not proceed until labels are complete; the pipeline will exit                                       |
| G2 (slate)         | `data/processed/gold/production_slate.json` | Confirm the proposed model slate from stage 02                          | Use the proposed slate unless you have a reason to change it                                           |
| G3 (test unlock)   | `data/processed/gold/test_unlock.json`      | Confirm the test unlock after the best model is selected in stage 06    | Match the checkpoint SHA; do not unlock prematurely                                                    |
| G4 (anchor labels) | `data/processed/gold/anchor_to_code.csv`    | Ensure every anchor row has a complete `0/1` human label                | Do not proceed until labels are complete; prevalence on LOW-quality rows cannot be validated otherwise |

---

## What you can safely leave at default

The following sections are already tuned for the production religious-missions task and work for most retasks without modification:

- **`paths`** — Standard cookiecutter-data-science layout (`data/raw`, `data/interim`, `data/processed`, `data/models`).
- **`q_thresholds`** — The Q rubric is entity-agnostic (`HIGH >= 5.0`, `MEDIUM >= 3.0`, `LOW >= 0.0`).
- **`sample_sizes`** — `silver: 20000`, `gold: 450`, `prompt_dev: 50`, `monitor: 50` is the production standard.
- **`anchor`** — `n: 500`, `oversample_low_factor: 1.5`, `min_stratum_frame: 200` is calibrated for prevalence estimation.
- **`annotation`** — `temperature: 0.0`, `max_retries: 5`, `checkpoint_every: 100`, `guided_json: true`, `openai_max_concurrency: 2`, `vllm_max_concurrency: 8`, and production `openai_batch: true` are best-practice defaults.
- **`data`** — `allow_synthetic: false` is the production default; only set to `true` for smoke tests.
- **`qc`** — `kappa_threshold: 0.70`, `f1_ci_floor: 0.70` is the validated freeze gate.
- **`aggregation`** — `method: majority` with `comparison_arms: []` is the default production path.
- **`training`** — `targets: soft`, `arms: [hard, class_weighted]`, `curve_fractions: [1.0]`, `final_seeds: [42, 43, 44, 45, 46]`, DeBERTa-v3 as primary encoder and ModernBERT as comparison are the research defaults.
- **`evaluation`** — `calibration_methods: [platt, temperature]`, `threshold_policy: precision_floor`, `precision_floor: 0.80`, `max_ece: 0.05` are the validated acceptance gate.
- **`inference`** — `batch_size: 512`, `route_low_to_rules: true`, `rule_ambiguous_to_classifier: true`.
- **`prevalence`** — `cross_checks: [emq]`, `alpha: 0.05`, `use_design_weights: true`, `per_ntee: true`.

---

## How Config Loading Works

1. **YAML is read and parsed.** `load_config()` in `src/binary_classifier/config.py` reads the YAML file and parses it into a raw dictionary with `yaml.safe_load()`.
2. **Pydantic validates the dictionary.** The raw dictionary is passed to `BinaryClassifierConfig.model_validate()`, which recursively validates each section against its typed Pydantic `BaseModel` subclass. Extra keys not defined on the model are rejected (`extra="forbid"`).
3. **Defaults fill in gaps.** Every field on every sub-config has a sensible default defined in the Pydantic class. A YAML file needs only to override the fields that differ from those defaults. This is how `smoke.yaml` works: it sets only 24 lines of overrides and inherits all other values from the class defaults.
4. **PathRegistry resolves paths.** The `PathRegistry` class in `src/binary_classifier/paths.py` takes a config object and resolves the four root directories (`raw_dir`, `interim_dir`, `processed_dir`, `models_dir`) into concrete `pathlib.Path` objects for every artifact in the pipeline. Stages use `registry.silver_manifest`, `registry.annotation_store`, etc. rather than rebuilding paths by hand.

```
YAML file ---> load_config() ---> BinaryClassifierConfig ---> PathRegistry
                     (config.py)        (Pydantic model)      (paths.py)
```

All config classes live in `src/binary_classifier/config.py`. All artifact path properties live in `src/binary_classifier/paths.py`.

---

## Section-by-Section Reference

### Root-Level Settings

These are flat keys on the root `BinaryClassifierConfig` object, not nested under a subsection.

| YAML Key     | Type  | Default             | Description                                                                                                                               | Relevant Stages / Code                                    |
| ------------ | ----- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `SEED`       | `int` | `42`                | Global random seed for all stochastic steps: sampling, splits, training, and annotation temperature. Ensures reproducibility across runs. | All stages                                                |
| `entity`     | `str` | `"missions"`        | Name of the data entity being classified. Used for labeling artifacts and as a logical namespace. Changes when retasking.                 | `BinaryClassifierConfig`                                  |
| `field`      | `str` | `"LONGEST_MISSION"` | Column name in the upstream parquet that contains the text to classify. Must exist in the source data.                                    | Stage 01 (sampling)                                       |
| `label_name` | `str` | `"religious"`       | Semantic label for the positive class (class `1`). Used in reports, plots, and threshold selection.                                       | Stage 04 (QC), stage 06 (training), stage 07 (evaluation) |

---

### `paths`

Input and output directory roots. All paths are relative to the project root. `PathRegistry` derives every artifact path from these four roots.

| YAML Key              | Type   | Default          | Description                                                                                                                                                      | Artifacts Derived                                                        |
| --------------------- | ------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `paths.raw_dir`       | `Path` | `data/raw`       | Immutable upstream parquet inputs. Not created by `ensure_dirs()`; must exist before stage 01.                                                                   | `missions_cross_section.parquet`, `bmf_unified_processed.parquet`        |
| `paths.interim_dir`   | `Path` | `data/interim`   | Intermediate pipeline artifacts: manifests, bake-off scores, annotation stores. This is the primary cloud-symlink target in the pre-DVC setup.                   | Manifests, label stores, bake-off outputs, embeddings, OOF predictions   |
| `paths.processed_dir` | `Path` | `data/processed` | Final ready-to-train datasets. The `gold/` subdirectory is git-committed (human-coded labels, production slate, test unlock). `silver_labels.csv` is gitignored. | `silver_labels.csv`, evaluation reports, predictions, prevalence reports |
| `paths.models_dir`    | `Path` | `data/models`    | Persisted fine-tuned model artifacts: training runs, checkpoints, selection reports.                                                                             | `runs/`, `checkpoints/`, `selection_report.json`                         |

**PathRegistry properties** (in `src/binary_classifier/paths.py`) that use these roots:

| Property              | Resolved Path                                       |
| --------------------- | --------------------------------------------------- |
| `missions_parquet`    | `{raw_dir}/missions_cross_section.parquet`          |
| `bmf_parquet`         | `{raw_dir}/bmf_unified_processed.parquet`           |
| `interim_dir`         | `{interim_dir}`                                     |
| `processed_dir`       | `{processed_dir}`                                   |
| `gold_dir`            | `{processed_dir}/gold`                              |
| `bakeoff_dir`         | `{interim_dir}/bakeoff`                             |
| `bakeoff_store`       | `{interim_dir}/bakeoff/bakeoff_labels.csv`          |
| `models_dir`          | `{models_dir}`                                      |
| `silver_manifest`     | `{interim_dir}/manifests/silver_manifest.csv`       |
| `gold_manifest`       | `{interim_dir}/manifests/gold_manifest.csv`         |
| `annotation_store`    | `{interim_dir}/annotation_store.csv`                |
| `silver_labels`       | `{processed_dir}/silver_labels.csv`                 |
| `test_evaluation`     | `{processed_dir}/evaluation/test_evaluation.json`   |
| `predictions_parquet` | `{processed_dir}/predictions/predictions.parquet`   |
| `prevalence_report`   | `{processed_dir}/prevalence/prevalence_report.json` |

All config classes live in `src/binary_classifier/config.py`. All artifact path properties live in `src/binary_classifier/paths.py`.

**Pydantic class:** `PathsConfig` (line 19 of `config.py`)

---

### `model_slate`

Controls which LLMs compete in the stage-02 bake-off and which model is the production default.

| YAML Key                         | Type                     | Default        | Description                                                                                                                                                           | Relevant Stages                            |
| -------------------------------- | ------------------------ | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `model_slate.bakeoff_candidates` | `list[BakeoffCandidate]` | See below      | Model identities for the bake-off. Each entry has `id`, `provider` (`openai` or `vllm`), and optional `reasoning_effort`. The provider routes annotator construction. | Stage 02 (bake-off), stage 03 (annotation) |
| `model_slate.production`         | `str`                    | `"gpt-5-mini"` | Default production model id. The actual production slate that stage 03 runs is the human-confirmed `production_slate.json` (gate G2); this key is the seed/default.   | Stage 03                                   |

**Default bake-off candidates:**

| id                            | provider | reasoning_effort |
| ------------------------------ | -------- | ---------------- |
| `gpt-4o-mini`                  | `openai` | `null`           |
| `gpt-5-mini`                   | `openai` | `minimal`        |
| `gpt-5-nano`                   | `openai` | `minimal`        |
| `google/gemma-3-27b-it`        | `vllm`   | `null`           |
| `deepseek-ai/DeepSeek-V4-Flash` | `vllm`   | `null`           |

**Note:** OpenAI ids in the default are floating aliases. Pin them to dated snapshots before a production run. Each `vllm` arm requires its own vLLM server running on its own port during the bake-off (see `docs/RUNNING_ON_UCLOUD.md`'s "One node, many GPUs, many ports" and "DeepSeek-V4-Flash setup" sections); comment out either or both for a pure-OpenAI bake-off, or to skip just one open-weight arm. DeepSeek-V4-Flash additionally needs vLLM's DeepSeek-specific `--tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4` flags and a workaround for a broken `nvidia-cutlass-dsl` wheel (`utils/vllm_compat/sitecustomize.py`) — both already wired into `utils/serve-llm.sh` when this id is configured.

**Pydantic classes:**

- `BakeoffCandidate` (line 46 of `config.py`): `id: str`, `provider: Literal["openai", "vllm"]`, `reasoning_effort: str | None`
- `ModelSlateConfig` (line 90 of `config.py`)
- `Slate` (line 109 of `config.py`): Shape of `proposed_slate.json` and `production_slate.json` artifacts, with `confirmed: bool`, `models: list[BakeoffCandidate]`, and `selected: list[dict]`.

---

### `q_thresholds`

Quality-score thresholds for the computable rubric Q, which tiers mission descriptions by their informativeness.

| YAML Key              | Type    | Default | Description                                                                                             |
| --------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `q_thresholds.HIGH`   | `float` | `5.0`   | Minimum score for the HIGH tier: concrete missions with purpose, beneficiary, and activity all present. |
| `q_thresholds.MEDIUM` | `float` | `3.0`   | Minimum score for the MEDIUM tier: decent descriptions that are thin on one dimension.                  |
| `q_thresholds.LOW`    | `float` | `0.0`   | Floor for the LOW tier: fragments or bare labels handled by the rule layer.                             |

**Important:** The sampling frame in stage 01 is HIGH + MEDIUM only (`Q >= 3.0` with shipped defaults). LOW-quality rows are excluded from silver/gold sampling and are later handled by the high-precision rule layer during inference. Silver and gold datasets are **not** population-representative unless LOW is folded back in via the anchor sample.

**Pydantic class:** `QThresholdsConfig` (line 309 of `config.py`)

---

### `sample_sizes`

Target sizes for sampling. These are **targets**, not hard limits -- actual sizes may differ slightly due to frame availability after filtering.

| YAML Key                  | Type  | Default | Description                                                                                                                                                     | Relevant Stage     |
| ------------------------- | ----- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `sample_sizes.silver`     | `int` | `20000` | Target size for the silver pool (LLM-labeled, unverified). Over-provisioned to support the learning-curve sweep.                                                | Stage 01           |
| `sample_sizes.gold`       | `int` | `450`   | Target size for the gold set (hand-coded by a human engineer). Includes the monitor slice -- validation/test do not shrink when drift monitoring is enabled.    | Stage 01           |
| `sample_sizes.prompt_dev` | `int` | `50`    | Number of rows used for prompt development and the bake-off. These rows see every candidate prompt, so they must be kept small.                                 | Stage 01, stage 02 |
| `sample_sizes.monitor`    | `int` | `50`    | Held-out drift-monitor slice drawn from the gold allocation. `sample_sizes.gold` must be bumped by this amount to keep validation/test at their expected sizes. | Stage 01           |

**Pydantic class:** `SampleSizesConfig` (line 324 of `config.py`)

---

### `anchor`

Controls the stage-05 anchor sample, which covers the FULL frame including LOW-quality rows for population-prevalence estimation.

| YAML Key                       | Type    | Default | Description                                                                                                                                | Relevant Stage |
| ------------------------------ | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| `anchor.n`                     | `int`   | `500`   | Target number of anchor rows to sample from the full frame (all Q tiers).                                                                  | Stage 05       |
| `anchor.oversample_low_factor` | `float` | `1.5`   | Multiplier applied to the LOW-stratum sampling rate. Ensures enough LOW-quality rows appear in the anchor for prevalence validation.       | Stage 05       |
| `anchor.min_stratum_frame`     | `int`   | `200`   | Minimum frame count for a stratum to be eligible for independent sampling. Strata below this threshold are collapsed into adjacent strata. | Stage 05       |

**Pydantic class:** `AnchorConfig` (line 346 of `config.py`)

---

### `annotation`

Hyperparameters for the LLM-as-primary labeler that runs across the full silver pool.

| YAML Key                                    | Type    | Default                                       | Description                                                                                                                                                                                                | Relevant Stage     |
| ------------------------------------------- | ------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `annotation.temperature`                    | `float` | `0.0`                                         | LLM sampling temperature. Fixed at 0.0 to keep annotations low-variance and best-effort reproducible (closed APIs do not guarantee determinism across backend changes).                                    | Stage 02, stage 03 |
| `annotation.max_retries`                    | `int`   | `5`                                           | Maximum number of retry attempts per API call on transient failures (rate limits, timeouts).                                                                                                               | Stage 02, stage 03 |
| `annotation.checkpoint_every`               | `int`   | `100`                                         | Number of rows annotated between checkpoints. The label store is flushed to disk at this interval for resumability.                                                                                        | Stage 03           |
| `annotation.guided_json`                    | `bool`  | `true`                                        | Whether to use structured JSON output mode (function calling / guided decoding). When `true`, the annotator requests a structured JSON response; when `false`, falls back to raw text parsing.             | Stage 02, stage 03 |
| `annotation.openai_max_concurrency`         | `int`   | `2`                                           | Maximum simultaneous `annotator.annotate` calls sent to OpenAI across all model×prompt workers. Must be positive.                                                                                          | Stage 02, stage 03 |
| `annotation.vllm_max_concurrency`           | `int`   | `8`                                           | Maximum simultaneous `annotator.annotate` calls sent to vLLM across all model×prompt workers. The default targets the UCloud 2-8x B200 endpoint; reduce it if the server queues or OOMs. Must be positive. | Stage 02, stage 03 |
| `annotation.openai_batch`                   | `bool`  | `false` (`true` in `religious_missions.yaml`) | When `true`, Stage 03 routes OpenAI production annotators through the OpenAI Batch API. Stage 02 bake-off remains live calls. Set to `false` for shorter live runs.                                        | Stage 03           |
| `annotation.openai_batch_poll_seconds`      | `int`   | `30`                                          | Seconds between OpenAI Batch API status polls. Must be positive.                                                                                                                                           | Stage 03           |
| `annotation.openai_batch_completion_window` | `str`   | `"24h"`                                       | Completion window sent when creating OpenAI batches. OpenAI currently supports only `24h`.                                                                                                                 | Stage 03           |

**Note:** Resume is keyed by `(EIN2, source_id)` rather than row count, ensuring idempotency across interrupted runs (addresses legacy audit issue R-08).

**Pydantic class:** `AnnotationConfig` (line 354 of `config.py`)

---

### `data`

Data-loading behaviour, primarily for smoke-test support.

| YAML Key               | Type   | Default | Description                                                                                                                                                                                                                    | Relevant Stage     |
| ---------------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| `data.allow_synthetic` | `bool` | `false` | When `false` (production default), a missing upstream parquet is a hard error. When `true`, a synthetic dataset is generated with a loud warning and a `data_source="synthetic"` stamp. Intended for local smoke-testing only. | Stage 01, stage 05 |

**Pydantic class:** `DataConfig` (line 370 of `config.py`)

---

### `qc`

Quality-control gate thresholds for the stage-04 silver-label freeze.

| YAML Key                            | Type    | Default | Description                                                                                                                                                                  | Relevant Stage |
| ----------------------------------- | ------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `qc.agreement_threshold`            | `float` | `0.85`  | Legacy raw LLM-vs-human agreement benchmark on the validation overlap. Logged for continuity but not the sole blocking gate.                                                 | Stage 04       |
| `qc.kappa_threshold`                | `float` | `0.70`  | Minimum Cohen's kappa required to freeze silver labels. The default of 0.70 reproduces the old ~0.85 raw-agreement operating point on the roughly balanced validation split. | Stage 04       |
| `qc.f1_ci_floor`                    | `float` | `0.70`  | Minimum lower bound of the bootstrap 95% confidence interval for minority-class F1 required to freeze silver labels. This is the deliberate new variance gate.               | Stage 04       |
| `qc.abstain_on_fabricated_positive` | `bool`  | `false` | When `true`, any positive label (`binary_label == "religious"`) that carries a fabricated evidence span is treated as an abstain (`binary_label = None`) before aggregation. | Stage 04       |

**Pydantic class:** `QCConfig` (line 384 of `config.py`)

---

### `aggregation`

Production aggregation method (stage 04) and diagnostic comparison arms (stage 11).

| YAML Key                      | Type                                       | Default      | Description                                                                                                                       | Relevant Stage |
| ----------------------------- | ------------------------------------------ | ------------ | --------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `aggregation.method`          | `Literal["majority"]`                      | `"majority"` | Production aggregation method. Intentionally majority-only; no other methods are currently supported for production.              | Stage 04       |
| `aggregation.comparison_arms` | `list[Literal["dawid_skene", "crowdlab"]]` | `[]`         | Optional diagnostic comparison arms for the script-only stage 11 sensitivity analysis. These do **not** affect production labels. | Stage 11       |

**Pydantic class:** `AggregationConfig` (line 267 of `config.py`)

---

### `training`

Fine-tuning, baseline, and learning-curve configuration for stage 06.

| YAML Key                           | Type                                    | Default                                    | Description                                                                                                                                                                                                                     |
| ---------------------------------- | --------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training.dev_fraction`            | `float`                                 | `0.1`                                      | Fraction of silver labels reserved for development during training-data construction.                                                                                                                                           |
| `training.targets`                 | `Literal["soft", "hard"]`               | `"soft"`                                   | Whether encoder training consumes soft aggregate scores (vote-share targets) or hard majority-vote labels.                                                                                                                      |
| `training.arms`                    | `list[TrainingArm]`                     | `["hard", "class_weighted"]`               | Training-data variants included in the sweep. Default is `hard` and `class_weighted`; `pruned` is kept for opt-in but excluded by default because soft targets already down-weight the disagreement band that `pruned` removes. |
| `training.curve_fractions`         | `list[float]`                           | `[1.0]`                                    | Fractions of available training data used for training. Default `[1.0]` runs a single full-data pass; the `{25,50,100}%` learning curve is dropped by default.                                                                  |
| `training.sweep_seeds`             | `list[int]`                             | `[42, 43, 44]`                             | Random seeds for the model-selection sweep (encoder arms x curve fractions x seeds).                                                                                                                                            |
| `training.final_seeds`             | `list[int]`                             | `[42, 43, 44, 45, 46]`                     | Random seeds for final model refits after the best cell is selected.                                                                                                                                                            |
| `training.crossfit_folds`          | `int`                                   | `5`                                        | Number of cross-fit folds for silver-label prediction during training-data construction.                                                                                                                                        |
| `training.encoders`                | `list[EncoderArm]`                      | See below                                  | Encoder models evaluated in the training stage. See the EncoderArm sub-table below.                                                                                                                                             |
| `training.baselines`               | `list[BaselineName]`                    | `["tfidf_logreg", "minilm_logreg"]`        | Baseline model families evaluated alongside encoders for comparison.                                                                                                                                                            |
| `training.minilm_id`               | `str`                                   | `"sentence-transformers/all-MiniLM-L6-v2"` | Sentence-transformer model id for the MiniLM logistic regression baseline.                                                                                                                                                      |
| `training.learning_rate`           | `float`                                 | `2.0e-5`                                   | Optimizer learning rate for encoder fine-tuning.                                                                                                                                                                                |
| `training.batch_size`              | `int`                                   | `32`                                       | Per-device training batch size.                                                                                                                                                                                                 |
| `training.epochs`                  | `int`                                   | `10`                                       | Maximum number of training epochs.                                                                                                                                                                                              |
| `training.weight_decay`            | `float`                                 | `0.01`                                     | Optimizer weight decay (AdamW).                                                                                                                                                                                                 |
| `training.warmup_fraction`         | `float`                                 | `0.06`                                     | Fraction of training steps used for linear learning-rate warmup.                                                                                                                                                                |
| `training.early_stopping_patience` | `int`                                   | `4`                                        | Number of validation checks without improvement before stopping.                                                                                                                                                                |
| `training.metric_for_best_model`   | `str`                                   | `"pr_auc"`                                 | Validation metric used for checkpoint selection.                                                                                                                                                                                |
| `training.greater_is_better`       | `bool`                                  | `true`                                     | Whether larger values of `metric_for_best_model` are preferred.                                                                                                                                                                 |
| `training.save_total_limit`        | `int`                                   | `2`                                        | Maximum number of checkpoints retained per training run.                                                                                                                                                                        |
| `training.precision`               | `Literal["auto", "bf16", "fp32"]`       | `"auto"`                                   | Numeric precision policy for encoder fine-tuning.                                                                                                                                                                               |
| `training.device`                  | `Literal["auto", "cuda", "mps", "cpu"]` | `"auto"`                                   | Compute device selection policy.                                                                                                                                                                                                |
| `training.report_to`               | `list[str]`                             | `[]`                                       | External training loggers enabled for HuggingFace `Trainer` integrations (e.g. `["wandb"]`).                                                                                                                                    |

**Default encoders:**

| id                            | arm          | max_length |
| ----------------------------- | ------------ | ---------- |
| `microsoft/deberta-v3-base`   | `primary`    | `256`      |
| `answerdotai/ModernBERT-base` | `comparison` | `256`      |

**Relevant stages:** Stage 06.

**Pydantic classes:**

- `TrainingConfig` (line 470 of `config.py`): All training knobs.
- `EncoderArm` (line 454 of `config.py`): `id: str`, `arm: Literal["primary", "comparison"]`, `max_length: int`.
- `TrainingArm` type (line 409): `Literal["hard", "pruned", "class_weighted"]`.
- `BaselineName` type (line 410): `Literal["tfidf_logreg", "minilm_logreg"]`.

---

### `evaluation`

Evaluation, calibration, and frozen-test acceptance configuration for stage 07.

| YAML Key                         | Type                                    | Default                    | Description                                                           |
| -------------------------------- | --------------------------------------- | -------------------------- | --------------------------------------------------------------------- |
| `evaluation.acceptance`          | `AcceptanceCriteria`                    | See below                  | Thresholds that must be met for test-unlock approval.                 |
| `evaluation.calibration_methods` | `list[Literal["platt", "temperature"]]` | `["platt", "temperature"]` | Calibration methods considered for score scaling.                     |
| `evaluation.crossfit_folds`      | `int`                                   | `5`                        | Number of folds for anchor out-of-fold scoring.                       |
| `evaluation.threshold_policy`    | `Literal["precision_floor", "max_f1"]`  | `"precision_floor"`        | Rule for selecting the operating threshold on the score distribution. |
| `evaluation.precision_floor`     | `float`                                 | `0.80`                     | Minimum precision under the `precision_floor` threshold policy.       |
| `evaluation.ece_bins`            | `int`                                   | `10`                       | Number of bins for expected calibration error (ECE) computation.      |
| `evaluation.bootstrap_resamples` | `int`                                   | `2000`                     | Number of bootstrap draws for confidence interval estimation.         |
| `evaluation.length_bins`         | `list[int]`                             | `[10, 25, 50]`             | Word-count bin edges for text-length subgroup reporting.              |

**Acceptance criteria sub-table (`evaluation.acceptance`):**

| YAML Key                                         | Type    | Default | Description                                                                                            |
| ------------------------------------------------ | ------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `evaluation.acceptance.min_pr_auc`               | `float` | `0.90`  | Minimum precision-recall AUC on the frozen test set.                                                   |
| `evaluation.acceptance.min_minority_f1_ci_lower` | `float` | `0.70`  | Minimum lower bound of the bootstrap confidence interval for minority-class F1 on the frozen test set. |
| `evaluation.acceptance.max_ece`                  | `float` | `0.05`  | Maximum expected calibration error on anchor out-of-fold scores.                                       |

**Relevant stages:** Stage 07.

**Pydantic class:** `EvaluationConfig` (line 180 of `config.py`), `AcceptanceCriteria` (line 146 of `config.py`).

---

### `inference`

Batch inference configuration for stage 08.

| YAML Key                                 | Type                                    | Default  | Description                                                                                   |
| ---------------------------------------- | --------------------------------------- | -------- | --------------------------------------------------------------------------------------------- |
| `inference.batch_size`                   | `int`                                   | `512`    | Number of rows scored per classifier inference batch.                                         |
| `inference.shard_size`                   | `int`                                   | `50000`  | Number of input rows processed per inference shard (controls memory usage for large corpora). |
| `inference.device`                       | `Literal["auto", "cuda", "mps", "cpu"]` | `"auto"` | Compute device selection policy for classifier inference.                                     |
| `inference.route_low_to_rules`           | `bool`                                  | `true`   | Whether LOW-quality rows use the rule layer before classifier scoring.                        |
| `inference.rule_ambiguous_to_classifier` | `bool`                                  | `true`   | Whether rule-layer abstentions on LOW-quality rows fall through to the classifier.            |

**Pydantic class:** `InferenceConfig` (line 207 of `config.py`).

**Relevant stages:** Stage 08.

---

### `prevalence`

Population-prevalence estimation configuration for stage 09.

| YAML Key                          | Type                           | Default   | Description                                                                                                                                                                                                |
| --------------------------------- | ------------------------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prevalence.alpha`                | `float`                        | `0.05`    | Significance level for prevalence confidence intervals (produces 95% CIs by default).                                                                                                                      |
| `prevalence.cross_checks`         | `list[Literal["emq", "kdey"]]` | `["emq"]` | Secondary quantification estimators reported alongside the primary prevalence estimate. EMQ (vendored SLD) is the single default quantification sensitivity check; KDEy via QuaPy is available for opt-in. |
| `prevalence.use_design_weights`   | `bool`                         | `true`    | Whether anchor-sample design weights are used when estimating prevalence.                                                                                                                                  |
| `prevalence.per_ntee`             | `bool`                         | `true`    | Whether prevalence is also reported by NTEE major group (taxonomy code).                                                                                                                                   |
| `prevalence.ntee_min_n`           | `int`                          | `10`      | Minimum labeled or inferred rows required for an NTEE subgroup estimate to be reported.                                                                                                                    |
| `prevalence.low_tier_sensitivity` | `bool`                         | `true`    | Whether to report sensitivity bounds for LOW-quality missions routed through the rule layer.                                                                                                               |

**Pydantic class:** `PrevalenceConfig` (line 240 of `config.py`).

**Relevant stages:** Stage 09.

---

## Artifact Gateway Files

Several YAML knobs control the human-gate artifacts that are validated by `validate_gates()` in `src/binary_classifier/qc/preflight/`. These are JSON/CSV files produced by pipeline stages and confirmed by a human before downstream stages proceed.

| Artifact               | Path (from PathRegistry) | Gate | Controlled By / Produced By                                          |
| ---------------------- | ------------------------ | ---- | -------------------------------------------------------------------- |
| Gold coding template   | `gold_coding_template`   | G1   | Stage 01 writes it; human fills `human_label` (0/1).                 |
| Production slate       | `production_slate`       | G2   | Stage 02 writes `proposed_slate.json`; human copies and confirms it. |
| Test unlock            | `test_unlock`            | G3   | Human creates after stage 06 selects the best model.                 |
| Anchor coding template | `anchor_coding_template` | G4   | Stage 05 writes it; human fills `human_label` (0/1).                 |

---

## Worked Retasking Walkthrough: Missions → Activities

Below is a concrete, step-by-step retasking from the default `missions` / `religious` task to an `activities` / `educational` task. No source code is edited.

### Step 1: Copy the production config

```bash
cp config/religious_missions.yaml config/educational_activities.yaml
```

### Step 2: Change the retasking triple

Edit the three root-level keys:

```yaml
entity: activities
field: LONGEST_ACTIVITY
label_name: educational
```

These are the only _required_ changes. Everything else can stay at default.

### Step 3: Review the model slate (optional)

The default bake-off candidates (`gpt-4o-mini`, `gpt-5-mini`, `gpt-5-nano`, `google/gemma-3-27b-it`, `deepseek-ai/DeepSeek-V4-Flash`) are text-agnostic. If the new `field` is still short nonprofit text, keep the slate. If you have no vLLM server, comment out the `vllm` entries.

### Step 4: Run the pipeline

```bash
uv run python scripts/run_pipeline.py --config config/educational_activities.yaml
```

Stage 01 will sample, write manifests, and emit a `gold_coding_template.csv`. Stop and fill the human labels (G1).

### Step 5: Confirm the gates

- **G1** — Fill `data/processed/gold/gold_to_code.csv` with `0/1` labels.
- **G2** — After stage 02 proposes a slate, confirm `production_slate.json`.
- **G3** — After stage 06 selects the best model, create `test_unlock.json` matching the checkpoint SHA.
- **G4** — After stage 05 writes the anchor template, fill `anchor_to_code.csv`.

Because `entity` changed, the pipeline will emit task-labeled artifacts under the configured `paths` roots.

### Summary of changes

| File                                 | Lines changed | What changed                    |
| ------------------------------------ | ------------- | ------------------------------- |
| `config/educational_activities.yaml` | 3 root keys   | `entity`, `field`, `label_name` |
| `data/processed/gold/`               | Human-written | Four gate artifacts             |

---

## Adding a New Classification Task

1. Copy `config/religious_missions.yaml` to `config/<task>.yaml`.
2. Set `entity` to the new entity name (e.g. `"activities"`).
3. Set `field` to the text column in the upstream parquet (e.g. `"LONGEST_ACTIVITY"`).
4. Set `label_name` to the positive-class label (e.g. `"educational"`).
5. Adjust `paths` if task-specific storage is desired.
6. Optionally update `model_slate`, `sample_sizes`, `anchor`, or any other section.
7. Run: `uv run python scripts/run_pipeline.py --config config/<task>.yaml`.

No source code edits are needed, provided the upstream `NonProfitData` parquet exposes the chosen `field`.

---

## Related Documentation

- `src/binary_classifier/config.py` -- Pydantic model definitions and `load_config()`.
- `src/binary_classifier/paths.py` -- `PathRegistry` with every artifact path.
- `.agents/architecture/configuration.md` -- Architecture-level config design and retasking notes.
- `.agents/architecture/pipeline.md` -- How each pipeline stage consumes the config.
- `scripts/run_pipeline.py` -- Orchestrator that wires config to stages with human gates.
