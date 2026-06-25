# Execution Plan: Pipeline Roadmap Completion (Stages 05–11)

> **Audience**: per-PR orchestrating agents and their specialized subagents, all
> starting with FRESH CONTEXT. This document is self-contained: every code fact cited
> here was verified against the repo on 2026-06-12 (branch `refactor/harmonize-pipeline`).
> Read §1–§5 before executing any PR. Each PR work order (§7) is independently
> executable given §1–§5.

---

## 1. Mission and deliverables

The repo is a config-driven weak-supervision pipeline that classifies 560,351 US
nonprofit mission statements (keyed `EIN2`) as religious (1) vs non-religious (0).
Stages 01–04 (sampling → LLM bake-off → LLM annotation → QC/freeze of silver labels)
are **complete and tested**. This plan implements everything that remains:

| Stage | Name | PR | Deliverable |
|---|---|---|---|
| 05 | Anchor sample | PR-1 | Representative human-coding sample (n=500) over the FULL corpus incl. LOW tier |
| 06 | Training | PR-2 | Soft-label fine-tuned encoder + baselines + gated arms + selection report |
| 07 | Evaluation | PR-3 | Calibrator + threshold + one-shot frozen-test evaluation behind unlock gate |
| 08 | Inference | PR-4 | Calibrated predictions over all 560k missions (classifier + rule router) |
| 09 | Prevalence | PR-5 | Weighted PPI++ composite prevalence, overall + per-NTEE |
| 10 | Visualization | PR-6 | Figures from stage artifacts (no model loading) |
| 11 | Aggregation compare | PR-7 | CROWDLAB/Dawid-Skene unlock + comparison vs majority vote |
| — | Dependency modernization | PR-0 | transformers v5, torch 2.7, vllm→extra, sentencepiece |

PR dependency order: PR-0 → PR-1 → PR-2 → PR-3 → PR-4 → PR-5; PR-6 needs PR-2
(full value after PR-5); PR-7 needs PR-2 (the OOF-probs artifact) and PR-4 conventions.

---

## 2. Orchestrator protocol and subagent contract

**Orchestrator (one per PR):**
1. Create the PR branch from the current mainline (`git switch -c <branch>`).
2. Read §3 (context pack), §4 (decision record), §5 (target architecture), §6
   (config/registry spec), and your PR's work order in §7 — fully, before spawning.
3. Spawn one subagent per task (T-numbers). Tasks within a PR may run in parallel
   ONLY where the work order marks them `[parallel-ok]`; otherwise sequential.
4. After all tasks report, run the PR acceptance gate (§7, per PR) yourself:
   `uv run pytest -m "not slow and not network"` + `uv run ruff check . &&
   uv run ruff format --check . && uv run ty check` must be green, plus the
   PR-specific checks.
5. Commit with a conventional message (`feat:`/`chore:`/`fix:`), ending with
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Do not push or open a
   PR unless the human asks.

**Subagent input contract** (the orchestrator includes in each spawn prompt):
(a) the task's full text from §7; (b) the §3 code-map rows it references; (c) the §6
spec blocks it implements; (d) the repo conventions block (§3.1).

**Subagent report-back contract** (every subagent's final message MUST contain):
1. `FILES`: every file created/modified, with absolute path and a one-line summary each.
2. `TESTS`: exact commands run and their pass/fail counts (paste the summary line).
3. `DEVIATIONS`: any departure from the task spec, with justification — or "none".
4. `RISKS`: anything discovered that affects other tasks/PRs — or "none".
5. `ARTIFACTS`: any new artifact schema actually produced (if it differs from §5.4, that is a DEVIATION).

A task is done only when its listed acceptance checks pass locally. Subagents must
not modify files owned by another task in the same PR (ownership is listed per task);
shared files (`config.py`, `paths.py`, `run_pipeline.py`, `preflight.py`,
`pyproject.toml`, the YAMLs) are owned by exactly one task per PR.

---

## 3. Context pack (verified code facts — trust these, re-verify only on conflict)

### 3.1 Repo conventions (from AGENTS.md, binding)

- Package manager **uv**; run everything `uv run python <script>` / `uv run pytest`.
- **Python 3.13** (`.python-version`). Lint = ruff (line length 88); types = **ty**
  (`uv run ty check`); both configured in `pyproject.toml`.
- Layout: logic in `src/binary_classifier/` (hatch package); `scripts/NN_name.py` are
  **thin CLI wrappers** (argparse: `--config` default `config/religious_missions.yaml`,
  build `cfg = load_config(args.config)`, `registry = PathRegistry(args.config)`,
  call the stage function); orchestrator `scripts/run_pipeline.py`.
- Stage entrypoints have signature `run_x(cfg, registry, **kwargs) -> None` and live
  in the package, never in scripts.
- **EIN2 is the join key — carry it through every artifact.** Global `cfg.SEED` seeds
  every stochastic step (`np.random.default_rng(cfg.SEED)`, `transformers.set_seed`).
- Logging: `logger = logging.getLogger(__name__)` per module; INFO milestones,
  WARNING skips; long-running training logs to FILE not terminal.
- `data/` is a real directory; `data/processed/gold/` is the **git-committed** pointer
  layer; everything else under `data/` is gitignored/cloud-symlinked.
- `archive/legacy-pipe/` is reference-only. Never import from it.
- Docstrings: Google style with Args/Returns, as in existing modules.

### 3.2 Existing code map (file → verified facts)

| File | Verified facts |
|---|---|
| `src/binary_classifier/config.py` | Pydantic models. Root `BinaryClassifierConfig` (line ~257): fields `SEED=42, entity="missions", field="LONGEST_MISSION", label_name="religious", paths, model_slate, q_thresholds, sample_sizes, annotation, data, qc, training` — all `default_factory`. No `model_config` is set, so **pydantic v2's default `extra="ignore"` applies at every nesting level** (unknown YAML keys, incl. inside nested blocks, are silently dropped; removing keys is safe; forward-referencing YAML blocks load fine). A final reviewer claimed the default is `forbid` — that is incorrect for pydantic v2 `BaseModel`; T1.2 nevertheless sets `model_config = ConfigDict(extra="ignore")` explicitly on the root model to make the contract visible, and its acceptance check (loading both YAMLs) verifies it. `Slate` model + `load_slate()` at lines 109–143 (`confirmed: bool=False, models: list[BakeoffCandidate], selected: list[dict]`) — the human-confirmed-JSON pattern to clone. `TrainingConfig` stub at lines 238–254 (placeholder: `learning_rate=5e-5, batch_size=16, epochs=10, weight_decay=0.01, metric_for_best_model="pr_auc", greater_is_better=True, early_stopping_patience=4, save_total_limit=2, fp16=True`) — **to be replaced**. `load_config(path)` at 296–309. |
| `src/binary_classifier/paths.py` | `PathRegistry(config_path)`; `PathRegistry.from_config(cfg, root=None)` classmethod (tests use `root=tmp_path`). All paths are `@property -> Path`. Existing properties: `missions_parquet, bmf_parquet, interim_dir, processed_dir, gold_dir (processed/gold), bakeoff_dir, models_dir, silver_manifest, gold_manifest, prompt_dev_manifest, validation_manifest, test_manifest, monitor_manifest (interim/manifests/*.csv), gold_coding_template (gold/gold_to_code.csv), proposed_slate, production_slate (gold/production_slate.json), bakeoff_results, annotation_store (interim/annotation_store.csv), bakeoff_store, prompts_dir`. `ensure_dirs()` (lines 172–182) creates 6 dirs. |
| `scripts/run_pipeline.py` | `_STAGE_MODULES` dict at line 28: `{"01": ("binary_classifier.data.sample","build_sample"), "02": (…,"run_bakeoff"), "03": (…,"run_annotation"), "04": ("binary_classifier.qc.agreement","run_quality_check")}`. `_GATE_EXIT = 2`. `_run_stage(stage_id, cfg, registry, annotate_limit, force)` imports via importlib and calls `func(cfg, registry)` (stage 01 gets `force=`, stage 03 gets `limit=`). `_report_gate(title, problems, hint)` prints to stderr. `run_pipeline(cfg, registry, requested, …)` checks G1 (labels) for `{"02","04"}` before model stages, G2 (slate) before `"03"`; `sys.exit(_GATE_EXIT)` on failure. CLI: `--config, --stages, --annotate-limit, --force`. |
| `src/binary_classifier/qc/preflight.py` | `_STAGE_SPLITS = {"02": "prompt_dev", "04": "validation"}` (line 28). `validate_gates(cfg, registry, stages: Iterable[str]) -> list[str]` (lines 33–61) — pure, returns problem strings. Label check requires strict 0/1 coding for the split in `gold_to_code.csv`. |
| `src/binary_classifier/qc/agreement.py` | `run_quality_check(cfg, registry)` = stage 04. Private `_compute_metrics` (lines 345–416): confusion matrix (tn/fp/fn/tp), minority precision/recall/F1, MCC, balanced accuracy, Cohen's κ, Krippendorff's α, PR-AUC (when confidence scores available), + bootstrap CIs. `_bootstrap_ci(y_true, y_pred, minority_class, seed, n_resamples=1000, confidence_level=0.95)` (lines 457–511) → `{"accuracy": {"lower","upper"}, "minority_f1": {...}}`. Gold exclusion before freeze: `_exclude_gold_manifest_ein2s` (line ~297) with **string-normalized EIN2 comparison** (line ~339 — replicate this everywhere EIN2 sets are compared, dtype drift is real). `_load_validation_labels` (line ~514) shows how human labels are read from the template filtered by `split=="validation"`. Writes `processed_dir / "silver_labels.csv"`. Freeze gate: κ ≥ `cfg.qc.kappa_threshold` (0.70) AND minority-F1 bootstrap CI lower ≥ `cfg.qc.f1_ci_floor` (0.70); raises on failure. |
| `src/binary_classifier/annotate/aggregate.py` | `majority_vote(df)` (lines 19–79) → wide df, one row per EIN2, columns exactly: `EIN2, silver_label (1/0/NaN), silver_confidence, num_votes, num_abstain, agreement, tie`. **Quarantined**: `aggregate_dawid_skene` (85–110), `aggregate_crowdlab` (113–135) — both `raise NotImplementedError` with explanatory docstrings. Dispatch `aggregate_labels(df, method="majority")` (141–166) via dict `{"majority","dawid_skene","crowdlab"}`. |
| `src/binary_classifier/annotate/schema.py` | `AnnotationStore` (CSV-backed long/tidy store). `AnnotationStore.COLUMNS` (lines 276–294): `EIN2, source_id, source_type, label, confidence, model_id, prompt_id, temperature, seed, run_timestamp, raw_response, reason, domains_present, evidence_spans, boundary_notes, binary_label, system_fingerprint`. Two label-ish columns, do not confuse them: `label` is the numeric working vote `1/0/NaN` (NaN = abstain) — **this is the column soft targets are computed from**; `binary_label` is the source enum string from the LLM JSON. `source_id` identifies the (model × prompt) vote source. Store supports resume via `already_done` keyed by (EIN2, source_id). |
| `src/binary_classifier/data/sample.py` | Stage 01 `build_sample(cfg, registry, force=False)`. Reusable: `_sample_stratified_by_group(df, target_size, rng) -> pd.DataFrame` (lines 511–541, stratifies by `ntee_major_group`). `_write_manifest` (474–495) writes columns `[EIN2, stratum, tier, inclusion_prob, is_positive_enriched, split]`; `_write_split_manifest` (498–505) resolves the registry path. Silver `inclusion_prob` computed per stratum × enrichment cell; **gold `inclusion_prob` is explicitly diagnostic-only** (boundary cases over-sampled; comments at lines 188–189, 244–246). `split_human_sets` (269–319) makes prompt_dev (50) / monitor (50) / validation (~175) / test (~175) from gold=450. Stage-01 coding template has **clobber protection** (refuses overwrite without `force=True`) — replicate for the anchor template. |
| `src/binary_classifier/data/quality.py` | `compute_quality_score(text) -> float` (425–484, Q ∈ [0,~6]); `assign_tier(q_score, thresholds=None) -> str` ("HIGH"/"MEDIUM"/"LOW", 582–601); `apply_rule_label(text) -> int | None` (604–635: 1 = strong-tradition lexicon hit, 0 = very short + no religious lexicon, None = ambiguous → classifier). **Hardcoded religious lexicons** `RELIGIOUS_LEXICON`, `_STRONG_TRADITION_WORDS` (lines 21–98) — task-specific by design (see §4.4). |
| `src/binary_classifier/data/load.py` | `load_missions(cfg) -> pd.DataFrame` (46–134) with columns `EIN2, mission_text, ntee_major_group, is_truncated, NTEE_IRS, data_source`. Returns the **FULL corpus incl. LOW** (no tier filtering — tiering is computed downstream). Synthetic fallback (151–352): when parquets missing AND `cfg.data.allow_synthetic=True`, generates **n=1,000** rows with realistic NTEE distribution, ~15% religious-word injection, stamps `data_source="synthetic"`, loud warning, temp files auto-cleaned. |
| `tests/conftest.py` | `tiny_config()` (15–25): in-memory `BinaryClassifierConfig` with `qc.f1_ci_floor=0.0`. `tiny_registry(tiny_config, tmp_path)` (29–33): `PathRegistry.from_config(cfg, root=tmp_path)` + `ensure_dirs()`. Tests fabricate stores/templates/manifests per-test (see `tests/test_agreement.py` for the pattern: build LabelRecords → append to store → write template CSVs). 13 test files exist; all CPU, no network. |
| `config/religious_missions.yaml` | The production config. Currently has `training.fp16: true` (replaced by this plan; safe to remove — extra keys ignored). No anchor/evaluation/inference/prevalence/aggregation blocks yet. |
| `pyproject.toml` | Current pins: `transformers>=4.56.0,<5.0.0`, `torch>=2.6.0`, `vllm>=0.5.0` (in core deps), `accelerate>=0.26.0, cleanlab>=2.9.0, crowd-kit>=1.4.2, datasets>=4.8.5, evaluate>=0.4.6, matplotlib>=3.10.9, numpy>=1.26.0, openai>=1.0.0, pandas>=3.0.3, pydantic>=2.0.0, pydantic-settings, pyarrow>=15.0.0, python-dotenv, pyyaml, scikit-learn>=1.9.0`. dev group: ipykernel, pytest>=9.0.3, requests. `[tool.pytest.ini_options] testpaths=["tests"]` (no markers registered yet). `[tool.deptry.package_module_name_map]` exists (extend it for new deps with diverging import names). ruff excludes `.agents`, `archive`, `tests`. |
| `src/binary_classifier/annotate/annotators/vllm_annotator.py` | Uses `from openai import OpenAI` (line 11) against a served vLLM endpoint. **Zero `import vllm` anywhere in src/ or scripts/** — the `vllm` dependency exists only to serve models out-of-process. |
| `docs/RUNNING_ON_UCLOUD.md` | UCloud B200 runbook: persistent `/work`, `utils/init.sh`, `.env` sourcing. Update it in PR-0 with the torch/cu128 note. |

### 3.3 Data facts

- Corpus: 560,351 missions (unique EIN2). Quality tiers via Q rubric: HIGH 31.9% /
  MEDIUM 38.6% / LOW 29.5%. Religious base rate ≈ 13%. Sampling frame for stages
  01–04 was HIGH+MEDIUM only (`Q ≥ 3.0`); LOW was excluded from silver/gold.
- Silver pool ≈ 20k (positive-enriched, stratified by NTEE major group). Gold = 450:
  prompt_dev 50 / monitor 50 / validation ~175 / test ~175 (boundary-oversampled by
  design — fine for discrimination metrics, INVALID for calibration/prevalence).
- Human labels live in `data/processed/gold/gold_to_code.csv`
  (columns `EIN2, split, text, human_label`).
- `silver_labels.csv` (stage-04 output, under `processed_dir`, NOT committed):
  columns `EIN2, silver_label, silver_confidence, num_votes, num_abstain, agreement,
  tie`; NaN `silver_label` = abstain/tie rows; gold EIN2s already excluded.

---

## 4. Decision record (user decisions + independent evidence, June 2026)

### 4.1 User decisions (interviewed, final)

1. **Anchor sample n=500** (floor; per-NTEE CIs will be honest-but-wide; suppression below n=10 per group).
2. **Soft-label training default** (targets = vote shares); hard majority-vote is a gated arm.
3. **Slim documentation curve + comparison arms** instead of the original {0.5k..16k} × encoders × seeds sweep.
4. **Threshold = precision floor 0.80** (max recall s.t. precision ≥ 0.80); max-F1 reported as secondary reference; decision-curve analysis as report-only figure. Calibrated probabilities are always released alongside hard labels.
5. **Full remaining roadmap in phased PRs** incl. the CROWDLAB unlock.
6. **All testing is LOCAL** (developer machine, platform-agnostic: CUDA, MPS, or CPU — resolved at runtime, never assumed). UCloud B200 is the production run only, not a test tier.

### 4.2 Statistical design (literature-verified)

- **Positivity requirement** decides the anchor: PPI/PPI++ (Angelopoulos et al.,
  Science 2023, arXiv:2301.09633; PPI++ arXiv:2311.01453), Stratified PPI (Fisch et
  al., NeurIPS 2024, arXiv:2406.04291) and DSL (Egami et al., NeurIPS 2023,
  arXiv:2306.04746) all require strictly positive known inclusion probability over
  the full target frame. The existing gold set gives LOW tier π=0 and oversamples the
  decision boundary ⇒ it cannot support corpus-wide or per-NTEE prevalence. The gold
  splits remain valid for prompt-dev / model selection / discrimination metrics.
- **One anchor, four roles** via K-fold cross-fitting: (i) calibration fitting,
  (ii) prior correction, (iii) PPI/DSL labeled set, (iv) rule-layer validation on its
  LOW cells. Cross-fit = fit calibrator on out-of-fold rows, predict in-fold; no row
  both fits and certifies itself (Kumar/Liang/Ma NeurIPS 2019 on calibration sample
  complexity; standard double-dipping remedy).
- **Calibration**: Platt scaling (2 params, **has an intercept that absorbs the
  enrichment-induced prior shift**) is the default; temperature scaling (1 param, no
  intercept — cannot shift the prior) is the comparison; **isotonic excluded** (needs
  ~1000+ labels at 13% base rate; anchor has ~65 positives). Calibrate FIRST, then
  prior-correct (Alexandari et al., ICML 2020); SLD/EMQ doubles as the EM prior
  re-estimation cross-check (Saerens-Latinne-Decaestecker 2002).
- **Prevalence composite**: stratified recombination `p̂ = Σ_k w_k p̂_k`,
  `Var = Σ w_k² Var_k` (Fisch et al.). HIGH+MEDIUM stratum via weighted PPI++ with
  design weights `w = 1/sample_prob`; LOW stratum via rule labels corrected by
  **Rogan–Gladen** (1978): `p = (p_obs + spec − 1) / (sens + spec − 1)` using rule
  sensitivity/specificity estimated on the anchor's LOW cells, with a sensitivity
  band over the rule-precision range.
- **Selection metric**: seed-averaged PR-AUC (Chicco & Jurman 2020/2023 decision
  rule); MCC + minority-F1 reported at the chosen threshold. With ~175 validation
  rows, differences < 4–6 points are noise (Card et al., EMNLP 2020) ⇒ **tie rule**:
  an arm wins only if its seed-mean advantage exceeds the across-seed SD; ties go to
  the simpler arm.

### 4.3 Training recipe (literature-verified)

- **Soft targets (vote shares)**: match hard labels on accuracy, win on calibration
  (arXiv:2511.14117; arXiv:2412.09579; Davani et al. TACL; RED-CT arXiv:2408.08217;
  Pangakis & Wolken arXiv:2406.17633). Vote shares from correlated LLMs are
  optimistically concentrated ⇒ keep the hard-MV arm as the check.
- **Skip entirely** (do not implement): label smoothing (harms probability ranking,
  arXiv:2403.14715; redundant with soft targets — Wei et al. ICML 2022), focal loss
  & resampling (wrong imbalance regime — Henning et al. EACL 2023; distorts
  calibration), co-teaching/loss-filtering and SiDyP (wrong noise regime at the
  ensemble's ~10–15%; cite SiDyP KDD 2025 as frontier in docs), confidence-weighted
  loss (redundant with soft targets).
- **Gated arms (4)**: hard-MV; cleanlab confident-learning prune (flags from OOF
  probs ∩ low vote-share rows, drop + retrain — Northcutt et al. JAIR 2021);
  class-weighted CE (scored AFTER recalibration); ModernBERT-base vs DeBERTa-v3-base.
- **Plateau evidence**: LLM-label distillation saturates early; the binding error is
  the LLMs' systematic bias, not data volume (arXiv:2504.15432) ⇒ full-pool training
  + a {25%, 50%, 100%} × 1-seed documentation curve, not a 6-point sweep.
- Hyperparameters (Mosbach et al. ICLR 2021 stability): LR 2e-5, batch 32, ≤10
  epochs with early stopping (patience 4), weight decay 0.01, warmup fraction 0.06.
  3 seeds per arm, 5 seeds for the final pick.
- Encoders: **DeBERTa-v3-base primary** (best classification sample-efficiency,
  arXiv:2504.08716), **ModernBERT-base comparison** (throughput; native in
  transformers ≥4.48). RoBERTa/DistilBERT (in the old roadmap grid) are dropped —
  dominated by DeBERTa-v3; the TF-IDF and MiniLM baselines cover the cheap floor.
  NeoBERT excluded (`trust_remote_code` + stale xformers pin; CUDA-only).

### 4.4 Scope clarifications

- **Rule layer is task-specific**: `quality.py` lexicons are hardcoded (pre-existing
  design). The config-driven retasking promise covers sampling/annotation/training/
  evaluation; retasking the LOW rule layer needs a new lexicon. Making it
  config-injectable is a documented follow-up, NOT in scope.
- **Monitor split** (gold ~50): untouched by training/selection/calibration. Stage 08
  scores it every inference run into `monitor_scores.json` for run-over-run drift
  diffing (classifier-era analogue of stage-03 canary fingerprinting). Report-only.
- **Docs superseding notes are deliverables**: PR-2 records the sweep replacement +
  encoder-grid reduction in `.agents/plans/we-work-on-the-floofy-wreath.md` (memo),
  `docs/agents/configuration.md` (roadmap hooks), and `README.md`; PR-6
  records word-clouds → n-gram log-odds bars (statistically interpretable, no new dep).
- DVC migration stays deferred (configuration.md) — out of scope, do not touch.

### 4.5 Engineering facts (verified June 2026)

- **transformers 5.11.0 current**; pin `>=5.8.0,<6.0.0` (5.10.0 was yanked; 5.0–5.3
  had a silent DeBERTa-v3 tokenizer regression dropping [CLS]/[SEP], fixed 2026-03-24
  PR #44570). v5 API: `Trainer(processing_class=tokenizer)` (the `tokenizer=` kwarg
  is REMOVED); `eval_strategy` (not `evaluation_strategy`); **no `warmup_ratio`** —
  pass a float in [0,1) to `warmup_steps`; `report_to` defaults `"none"`;
  `compute_loss_func(outputs, labels, num_items_in_batch)` Trainer kwarg exists —
  **the custom loss must divide by `num_items_in_batch` itself** (Trainer skips
  normalization when it's supplied); `EarlyStoppingCallback(early_stopping_patience=)`
  unchanged, requires `load_best_model_at_end=True`; `label_smoothing_factor` still
  exists but is NOT used (see §4.3); bf16/bf16_full_eval flags unchanged.
- **vllm ≥0.22 requires transformers ≥5** — no conflict. vllm moves to optional extra
  `serve` (nothing imports it; it serves annotation models out-of-process).
- **torch ≥2.7**: B200 is sm_100, needs cu128 wheels (UCloud installs from the cu128
  index). macOS arm64 torch 2.7+ wheels exist. If `uv lock` fails on any local
  platform, fallback: keep floor 2.6 and document the 2.7/cu128 UCloud requirement in
  `docs/RUNNING_ON_UCLOUD.md`.
- **ppi_py 0.2.3**: `ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam=None,
  coord=None, w=None, w_unlabeled=None)` and `ppi_mean_ci(..., alpha=0.1, ...)` —
  weighted via `w`/`w_unlabeled`; `lam=None` auto power-tunes (PPI++). **Default
  alpha is 0.1 — always pass `alpha=cfg.prevalence.alpha` (0.05).** Py3.13 risk:
  numba (needs ≥0.61) — import smoke-test at lock time.
- **QuaPy 0.2.0**: `EMQ(classifier, fit_classifier=False)`; new API `fit(X, y)`;
  aggregative quantifiers expose `aggregate(posteriors)` for precomputed posterior
  arrays; `KDEyML` available. Risk: its `abstention` dep under numpy 2 — smoke-test;
  fallback: vendor the ~50-line EMQ EM loop (spec in §7 PR-5 T5.5).
- **cleanlab 2.9.0** (Py3.10–3.14): `cleanlab.multiannotator.
  get_label_quality_multiannotator(labels_multiannotator, pred_probs, *,
  consensus_method='best_quality', quality_method='crowdlab', ...)` — accepts an
  (N × M annotators) DataFrame with **NaN for missing votes** and (N, 2) pred_probs.
  Returns dict of DataFrames incl. `label_quality` (consensus label + quality).
  `crowdkit.aggregation.classification.DawidSkene` takes a long df with columns
  `task, worker, label` (already a dependency via crowd-kit).
- **Tracking**: JSONL + per-run log files are canonical. `trackio` (HF local tracker,
  in v5's `report_to` list) is an OPTIONAL knob (`training.report_to`), default off.
  No MLflow/W&B/aim.
- **Device/precision policy (platform-agnostic, applies everywhere)**: device
  resolution `auto` → `cuda` if `torch.cuda.is_available()` else `mps` if
  `torch.backends.mps.is_available()` else `cpu`. Precision resolution `auto` →
  `bf16` iff resolved device is `cuda` AND `torch.cuda.is_bf16_supported()`, else
  `fp32`. Never branch on platform names; only on these runtime capability checks.
  Tests monkeypatch the capability functions to cover all resolutions.

---

## 5. Target architecture

### 5.1 Module layout (new code in bold)

```
src/binary_classifier/
├── **metrics.py**                  PR-1  shared metric bundle + bootstrap CI (promoted)
├── data/
│   ├── load.py / sample.py / quality.py   (existing, unchanged)
│   └── **anchor.py**               PR-1  stage 05
├── annotate/                        (existing; aggregate.py extended in PR-7)
├── train/                          PR-2
│   ├── **data.py**                 training frame, soft targets, dev split, curve subsets
│   ├── **baselines.py**            tfidf_logreg, minilm_logreg
│   ├── **encoder.py**              one fine-tune run
│   ├── **crossfit.py**             K-fold OOF pred_probs over the silver pool
│   ├── **arms.py**                 gated arm runners (hard / pruned / class_weighted)
│   ├── **sweep.py**                run matrix + selection report
│   └── **trainer.py**              run_training entrypoint
├── evaluation/                     PR-3   (NOT `eval` — builtin; NOT `evaluate` — HF pkg)
│   ├── **calibration.py**          Platt/temperature, cross-fit
│   ├── **thresholds.py**           precision-floor + max-F1
│   ├── **subgroups.py**            per-NTEE / per-length error analysis
│   ├── **decision_curve.py**       net benefit (report-only)
│   └── **evaluate.py**             run_evaluation entrypoint
├── inference/                      PR-4
│   ├── **router.py**               pure tier/rule routing
│   └── **predict.py**              run_inference entrypoint (sharded, resumable)
├── prevalence/                     PR-5
│   ├── **weights.py**              design weights from anchor manifest
│   ├── **ppi.py**                  ppi_py wrappers
│   ├── **quantify.py**             EMQ/KDEy (+ vendored EMQ fallback)
│   ├── **composite.py**            HM ⊕ LOW stratified combination
│   └── **estimate.py**             run_prevalence entrypoint
├── viz/                            PR-6
│   ├── **ngrams.py** / **curves.py** / **prevalence_plots.py**
└── qc/
    ├── agreement.py                 (PR-1: switch to metrics.py imports)
    ├── preflight.py                 (PR-1: G4 + splits; PR-3: G3)
    └── **aggregation_compare.py**  PR-7

scripts/  **05_build_anchor.py  06_train.py  07_evaluate.py  08_infer.py
          09_prevalence.py  10_visualize.py  11_aggregation_compare.py**
config/   religious_missions.yaml (extended)   **smoke.yaml** (new, committed)
```

### 5.2 Orchestrator wiring

`_STAGE_MODULES` additions (each PR adds its row):
```python
"05": ("binary_classifier.data.anchor", "build_anchor"),          # PR-1
"06": ("binary_classifier.train.trainer", "run_training"),        # PR-2
"07": ("binary_classifier.evaluation.evaluate", "run_evaluation"),# PR-3
"08": ("binary_classifier.inference.predict", "run_inference"),   # PR-4
"09": ("binary_classifier.prevalence.estimate", "run_prevalence"),# PR-5
```
Stages 10 (viz) and 11 (aggregation compare) are script-only, not orchestrated.

### 5.3 Gate map (all checks in `qc/preflight.py`, reported via `_report_gate`, exit 2)

| Gate | Checks | Blocks |
|---|---|---|
| G1 (existing, extended) | strict 0/1 human labels in `gold_to_code.csv` for the stage's split. `_STAGE_SPLITS` gains `"06": "validation"`, `"07": "test"` | 02, 04, 06, 07 |
| G2 (existing) | confirmed `production_slate.json` | 03 |
| **G4** (PR-1) | `anchor_to_code.csv` exists and is fully coded strict 0/1 | 07, 09 |
| **G3** (PR-3) | `test_unlock.json` exists, `confirmed: true`, its `acceptance` snapshot equals current `cfg.evaluation.acceptance`, its `checkpoint_sha256` matches the selected checkpoint | 07 (the test-eval step only) |

### 5.4 Artifact registry (authoritative schemas)

| Artifact (PathRegistry property → path) | Producer | Schema |
|---|---|---|
| `anchor_manifest` → `interim/manifests/anchor_manifest.csv` | 05 | `EIN2, stratum (tier×ntee key), tier, ntee_major_group, sample_prob, split="anchor"` |
| `anchor_coding_template` → `gold/anchor_to_code.csv` (committed) | 05 | `EIN2, tier, text, human_label` (human fills 0/1) |
| `runs_dir` → `models/runs/<run_id>/` | 06 | `train.log`, `metrics.json`, `trainer_state.json` |
| `learning_curve_results` → `models/runs/results.jsonl` | 06 | one JSON object per completed run (schema below) |
| `selection_report` → `models/selection_report.json` | 06 | per-arm seed-mean±SD bundles, tie-rule verdicts, recommendation |
| `selected_model` → `gold/selected_model.json` (committed) | human after 06 | `{checkpoint_relpath, checkpoint_sha256, tokenizer_id, encoder_id, targets, recipe_snapshot, selected_at}` |
| `oof_pred_probs` → `interim/oof_pred_probs.parquet` | 06 | `EIN2, fold, p0, p1` (every silver EIN2 exactly once) |
| `test_unlock` → `gold/test_unlock.json` (committed) | human before 07 | `TestUnlock` model: `{confirmed, checkpoint, checkpoint_sha256, acceptance: {...}, rationale}` |
| `calibrator_path` → `processed/evaluation/calibrator.json` | 07 | `{method, params: {a,b}|{T}, threshold, threshold_policy, precision_floor, max_f1_threshold, fitted_on: "anchor", crossfit_folds, anchor_oof_scores_path}` |
| `anchor_oof_scores` → `processed/evaluation/anchor_oof_scores.parquet` | 07 | `EIN2, prob_raw, prob_calibrated_oof, human_label, tier, sample_prob` |
| `test_evaluation` → `processed/evaluation/test_evaluation.json` | 07 (one-shot) | metric bundle + CIs, subgroups, decision-curve points, calibration-on-anchor metrics, acceptance verdict, metadata (model id/sha, calibrator, config hash, git SHA, date) |
| `rule_validation` → `processed/evaluation/rule_validation.json` | 07 | rule sens/spec/precision/recall on anchor LOW cells + Wilson CIs + counts |
| `predictions_parquet` → `processed/predictions/predictions.parquet` (+ `shards/shard_{i:05d}.parquet`) | 08 | `EIN2, pred_label, prob_raw, prob_calibrated, decision_source ∈ {classifier, rule_strong_positive, rule_short_negative, low_via_classifier}, tier, Q, ntee_major_group, model_id, checkpoint_sha256, calibrator_method, threshold, inference_date, pipeline_version, config_hash` (rule rows: prob_* = NaN) |
| `monitor_scores` → `processed/predictions/monitor_scores.json` | 08 | per-monitor-row calibrated probs + run metadata (drift diffing) |
| `prevalence_report` → `processed/prevalence/prevalence_report.json` | 09 | estimands (HM, LOW, composite) with CIs, weighted+unweighted PPI, EMQ/KDEy cross-checks, sensitivity band, n per stratum |
| `prevalence_by_ntee` → `processed/prevalence/prevalence_by_ntee.csv` | 09 | `ntee_major_group, n_anchor, estimator, estimate, ci_lower, ci_upper, suppressed` |
| `figures_dir` → `processed/figures/` | 10 | PNG + SVG per figure |
| `aggregation_compare` → `interim/aggregation_compare.json` | 11 | per-arm metric bundles + CIs on human validation, adoption verdict |

`results.jsonl` row schema (stage 06):
```json
{"run_id": "...", "model": "baseline:tfidf_logreg" | "<hf-id>", "targets": "soft|hard",
 "arm": "default|hard|pruned|class_weighted", "train_fraction": 1.0, "n_train": 18000,
 "seed": 42, "dev": {<metric bundle>}, "validation": {<metric bundle + CIs>},
 "wall_seconds": 0.0, "precision": "bf16|fp32", "device": "cuda|mps|cpu",
 "git_sha": "...", "config_hash": "...", "timestamp": "..."}
```

### 5.5 Data-flow invariants (enforce in code + tests)

1. Human **test** split: read ONLY inside stage 07's one-shot step. The training data
   loader raises `ValueError` if asked for `split=="test"`.
2. Human **validation** split: model selection only (and stage-11 comparison). Never
   used to fit calibrators or thresholds.
3. **Anchor**: calibration/threshold/prior/prevalence/rule-validation only. Anchor
   rows are excluded from the training frame. Cross-fit discipline for any quantity
   later used in inference-about-the-corpus.
4. **Monitor** split: scored, never trained on, never selected on.
5. Every artifact carries EIN2; every EIN2 set comparison casts both sides
   `.astype(str)` (the `agreement.py:339` pattern); add `.str.strip()` when the
   source is a hand-edited CSV (coding templates).
6. Every stochastic operation seeds from `cfg.SEED` (or an explicit derived seed
   recorded in the artifact).

---

## 6. Config and registry target specification

### 6.1 New/changed pydantic models (`src/binary_classifier/config.py`)

```python
class AnchorConfig(BaseModel):
    """Stage 05 anchor sample over the FULL frame (incl. LOW)."""
    n: int = 500
    oversample_low_factor: float = 1.5      # LOW allocation multiplier; weights recorded
    min_stratum_frame: int = 200            # sectors with fewer frame rows get floor allocation

class EncoderArm(BaseModel):
    id: str
    arm: Literal["primary", "comparison"] = "comparison"
    max_length: int = 256

class TrainingConfig(BaseModel):            # REPLACES the stub at lines 238-254
    dev_fraction: float = 0.1
    targets: Literal["soft", "hard"] = "soft"
    arms: list[Literal["hard", "pruned", "class_weighted"]] = ["hard", "pruned", "class_weighted"]
    curve_fractions: list[float] = [0.25, 0.5, 1.0]
    sweep_seeds: list[int] = [42, 43, 44]
    final_seeds: list[int] = [42, 43, 44, 45, 46]
    crossfit_folds: int = 5
    encoders: list[EncoderArm] = Field(default_factory=lambda: [
        EncoderArm(id="microsoft/deberta-v3-base", arm="primary"),
        EncoderArm(id="answerdotai/ModernBERT-base", arm="comparison"),
    ])
    baselines: list[Literal["tfidf_logreg", "minilm_logreg"]] = ["tfidf_logreg", "minilm_logreg"]
    minilm_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 10
    weight_decay: float = 0.01
    warmup_fraction: float = 0.06           # passed as float warmup_steps (v5 semantics)
    early_stopping_patience: int = 4
    metric_for_best_model: str = "pr_auc"
    greater_is_better: bool = True
    save_total_limit: int = 2
    precision: Literal["auto", "bf16", "fp32"] = "auto"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    report_to: list[str] = []               # optional ["trackio"]

class AcceptanceCriteria(BaseModel):
    min_pr_auc: float = 0.90                # on frozen test
    min_minority_f1_ci_lower: float = 0.70  # on frozen test
    max_ece: float = 0.05                   # on anchor OOF (representative surface)

class EvaluationConfig(BaseModel):
    acceptance: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    calibration_methods: list[Literal["platt", "temperature"]] = ["platt", "temperature"]
    crossfit_folds: int = 5
    threshold_policy: Literal["precision_floor", "max_f1"] = "precision_floor"
    precision_floor: float = 0.80
    ece_bins: int = 10
    bootstrap_resamples: int = 2000
    length_bins: list[int] = [10, 25, 50]   # word-count bin edges

class InferenceConfig(BaseModel):
    batch_size: int = 512
    shard_size: int = 50_000
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    route_low_to_rules: bool = True
    rule_ambiguous_to_classifier: bool = True

class PrevalenceConfig(BaseModel):
    alpha: float = 0.05
    cross_checks: list[Literal["emq", "kdey"]] = ["emq", "kdey"]
    use_design_weights: bool = True
    per_ntee: bool = True
    ntee_min_n: int = 10
    low_tier_sensitivity: bool = True

class AggregationConfig(BaseModel):
    method: Literal["majority", "dawid_skene", "crowdlab"] = "majority"
    comparison_arms: list[str] = []

class TestUnlock(BaseModel):                # clone of the Slate pattern (lines 109-143)
    confirmed: bool = False
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    acceptance: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    rationale: str = ""

def load_test_unlock(path: Path | str) -> TestUnlock: ...   # mirrors load_slate
```
Root model: add fields `anchor: AnchorConfig`, `evaluation: EvaluationConfig`,
`inference: InferenceConfig`, `prevalence: PrevalenceConfig`,
`aggregation: AggregationConfig` — all `Field(default_factory=...)` so
`BinaryClassifierConfig()` stays valid (existing tests construct it bare).

### 6.2 PathRegistry additions (`src/binary_classifier/paths.py`)

New `@property` methods (style identical to existing ones):

```python
anchor_manifest          = interim_dir / "manifests" / "anchor_manifest.csv"
anchor_coding_template   = gold_dir / "anchor_to_code.csv"
silver_labels            = processed_dir / "silver_labels.csv"   # formalizes agreement.py's inline path
runs_dir                 = models_dir / "runs"
checkpoints_dir          = models_dir / "checkpoints"
learning_curve_results   = runs_dir / "results.jsonl"
selection_report         = models_dir / "selection_report.json"
selected_model           = gold_dir / "selected_model.json"
test_unlock              = gold_dir / "test_unlock.json"
oof_pred_probs           = interim_dir / "oof_pred_probs.parquet"
embeddings_dir           = interim_dir / "embeddings"
evaluation_dir           = processed_dir / "evaluation"
test_evaluation          = evaluation_dir / "test_evaluation.json"
calibrator_path          = evaluation_dir / "calibrator.json"
anchor_oof_scores        = evaluation_dir / "anchor_oof_scores.parquet"
rule_validation          = evaluation_dir / "rule_validation.json"
predictions_dir          = processed_dir / "predictions"
predictions_parquet      = predictions_dir / "predictions.parquet"
monitor_scores           = predictions_dir / "monitor_scores.json"
prevalence_dir           = processed_dir / "prevalence"
prevalence_report        = prevalence_dir / "prevalence_report.json"
prevalence_by_ntee       = prevalence_dir / "prevalence_by_ntee.csv"
figures_dir              = processed_dir / "figures"
aggregation_compare      = interim_dir / "aggregation_compare.json"
```
Extend `ensure_dirs()` with: `runs_dir, checkpoints_dir, embeddings_dir,
evaluation_dir, predictions_dir / "shards", prevalence_dir, figures_dir`.
NOTE: `agreement.py` currently builds the silver-labels path inline — switch it to
`registry.silver_labels` (PR-1, no behavior change).

### 6.3 YAML updates

`config/religious_missions.yaml`: remove `training.fp16`; rewrite the `training:`
block to the new fields; add `anchor:`, `evaluation:`, `inference:`, `prevalence:`,
`aggregation:` blocks mirroring §6.1 defaults (write values explicitly — the YAML is
the human-facing source of truth even where it repeats defaults).

`config/smoke.yaml` (NEW, committed — used by verification tiers 2–3):
```yaml
SEED: 42
entity: missions
field: LONGEST_MISSION
label_name: religious
data: { allow_synthetic: true }
anchor: { n: 60 }
sample_sizes: { silver: 400, gold: 120, prompt_dev: 20, monitor: 10 }
training:
  epochs: 1
  batch_size: 16
  precision: fp32          # deterministic everywhere; auto would pick bf16 on CUDA
  device: auto             # cuda / mps / cpu — resolved at runtime
  sweep_seeds: [42]
  final_seeds: [42]
  curve_fractions: [1.0]
  arms: []                 # smoke skips gated arms
  crossfit_folds: 2
  encoders: [{ id: prajjwal1/bert-tiny, arm: primary, max_length: 128 }]
evaluation:
  acceptance: { min_pr_auc: 0.50, min_minority_f1_ci_lower: 0.0, max_ece: 0.30 }
  bootstrap_resamples: 200
  crossfit_folds: 2
inference: { batch_size: 64, shard_size: 200 }
prevalence: { alpha: 0.05 }
```

### 6.4 pyproject changes

```toml
# PR-0 (core deps)
"transformers>=5.8.0,<6.0.0",
"torch>=2.7.0",            # fallback to >=2.6.0 ONLY if uv lock fails on a local platform;
                           # then document the 2.7/cu128 UCloud requirement instead
"sentencepiece>=0.2.0",    # DeBERTa-v3 tokenizer
# REMOVE vllm from core; add:
[project.optional-dependencies]
serve = ["vllm>=0.22.0"]
tracking = ["trackio"]     # optional dashboard knob

# PR-0 (pytest markers)
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "slow: real training loops / long E2E; excluded by default",
  "network: requires HF Hub downloads; excluded by default",
]

# PR-5 (core deps)
"ppi-py>=0.2.3",
"quapy>=0.2.0,<0.3",
# deptry map: ppi-py -> ppi_py ; quapy -> quapy
```

---

## 7. PR work orders

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

---

### PR-0 — `chore/deps-transformers-v5` (dependency modernization)

**Objective**: move the environment to the June-2026 stack with zero behavior change
to stages 01–04.

**T0.1 — pyproject + lock** (owns `pyproject.toml`, `uv.lock`)
Operations: (1) Pre-check: `grep -rn "import vllm" src/ scripts/` MUST return zero
matches (re-verifying §3.2's claim) before moving vllm to the extra. (2) Environment
arch check: `uv run python -c "import platform; print(platform.machine())"` — on
Apple Silicon this MUST print `arm64`; if it prints `x86_64`, the local uv/python is
a Rosetta x86_64 build and **torch ≥2.3 has no macOS x86_64 wheels — resolution will
fail** (observed on this machine on 2026-06-12: uv resolved platform
`macosx_26_0_x86_64` and torch errored). Fix the toolchain first (`uv python install`
a native arm64 CPython and recreate `.venv`) — do NOT paper over it with
`tool.uv.required-environments`. (3) Apply §6.4 PR-0 edits exactly (transformers,
torch, sentencepiece, vllm → `serve` extra, `tracking` extra, pytest markers). Run
`uv lock`, `uv sync`. If torch 2.7 still fails to resolve on a correctly-arch'd local
platform, apply the documented fallback (floor 2.6.0) and record it as a DEVIATION.
Acceptance: `uv lock` clean; `uv sync` clean; `uv sync --extra serve` resolves;
`uv run pytest -m "not slow and not network"` green (existing 13 test files);
`uv run python -c "import transformers, torch, sentencepiece;
print(transformers.__version__, torch.__version__)"` prints 5.8+ / 2.7+ (or fallback).

**T0.2 — tokenizer regression guard** (owns `tests/test_tokenizer_sanity.py`) [parallel-ok]
Operations: new test file, `@pytest.mark.network`, asserting for BOTH
`microsoft/deberta-v3-base` and `answerdotai/ModernBERT-base`:
`ids = tok("hello world")["input_ids"]; ids[0] == tok.cls_token_id and
ids[-1] == tok.sep_token_id` (guards the v5.0–5.3 DeBERTa regression, §4.5).
Acceptance: `uv run pytest -m network tests/test_tokenizer_sanity.py` green (network
required, run once locally).

**T0.3 — UCloud doc note** (owns `docs/RUNNING_ON_UCLOUD.md`) [parallel-ok]
Operations: add a "GPU environment" note: B200 = sm_100 ⇒ torch ≥2.7 from the cu128
index; verify with `python -c "import torch; print(torch.version.cuda,
torch.cuda.is_bf16_supported())"`; `uv sync` without `--extra serve` for training jobs;
`--extra serve` only when serving annotation models.
Acceptance: doc renders; no other file touched.

**PR-0 gate**: T0.1–T0.3 reports green; full Tier-1 (§8) green.

---

### PR-1 — `feature/05-anchor-sample` (shared metrics + anchor stage + gates foundation)

**Objective**: promote the metric bundle to a shared module; implement stage 05 so the
human coding session can start; lay all registry/config/gate plumbing later PRs need.

**T1.1 — `metrics.py` promotion** (owns `src/binary_classifier/metrics.py`,
`src/binary_classifier/qc/agreement.py`, `tests/test_metrics.py`)
Operations: create `metrics.py` exposing
`compute_metric_bundle(y_true, y_pred, *, y_score=None, minority_class=1, seed=42,
n_resamples=1000, confidence_level=0.95) -> dict` and
`bootstrap_ci(y_true, y_pred, minority_class, seed, n_resamples=1000,
confidence_level=0.95) -> dict` by MOVING the bodies of `_compute_metrics`
(agreement.py:345–416) and `_bootstrap_ci` (457–511). Add `roc_auc` to the bundle when
`y_score` is provided (sklearn `roc_auc_score`). Rewire `agreement.py` to import and
delegate (keep its private names as thin aliases so its internal call sites and the
13 existing tests pass UNCHANGED). Also switch agreement.py's inline silver-labels
path to `registry.silver_labels` (after T1.2 lands — sequence T1.2 → T1.1 or
coordinate). New `tests/test_metrics.py`: numeric parity test (same fixture through
old API path and new module gives identical numbers), roc_auc presence/absence.
Acceptance: full Tier-1 green, including the untouched `tests/test_agreement.py`.

**T1.2 — registry + config plumbing** (owns `src/binary_classifier/paths.py`,
`src/binary_classifier/config.py`, `config/religious_missions.yaml`, `config/smoke.yaml`)
Operations: add ALL §6.2 properties + `ensure_dirs()` entries (one PR-wide pass, so
later PRs never touch paths.py; nested dirs like `predictions_dir / "shards"` use
`mkdir(parents=True, exist_ok=True)` as the existing code does). Add `AnchorConfig`
to config.py + root field + YAML block, and set
`model_config = ConfigDict(extra="ignore")` explicitly on `BinaryClassifierConfig`
(makes the already-default behavior visible; see §3.2). Create `config/smoke.yaml`
exactly per §6.3 — its training/evaluation/… blocks reference fields added in later
PRs; with extra="ignore" this loads fine today, and the acceptance check proves it.
Acceptance: `uv run python -c "from binary_classifier.config import load_config;
load_config('config/religious_missions.yaml'); load_config('config/smoke.yaml')"`;
Tier-1 green.

**T1.3 — anchor stage** (owns `src/binary_classifier/data/anchor.py`,
`scripts/05_build_anchor.py`, `tests/test_anchor.py`)
Operations: implement `build_anchor(cfg, registry, force=False) -> None`:
1. `df = load_missions(cfg)`; compute `Q = compute_quality_score(text)` and
   `tier = assign_tier(Q, cfg.q_thresholds)` for every row (vectorize via `.map`).
2. Exclusion: drop EIN2s present in `registry.silver_manifest` or
   `registry.gold_manifest` (string-normalized comparison per §5.5#5). Log the
   excluded count (~3.7% of frame) — this is a deliberate, documented estimand note.
3. Stratify by `tier × ntee_major_group`. Allocation: proportionate to stratum frame
   share, times `oversample_low_factor` for LOW strata, floor of 1 for any stratum
   with ≥ `min_stratum_frame` rows; renormalize to `cfg.anchor.n`. Draw with
   `np.random.default_rng(cfg.SEED)`. Compute and store per-stratum
   `sample_prob = n_drawn_k / n_frame_k` (the DSL/PPI design weight input).
4. Write `anchor_manifest` (§5.4 schema) — anchor.py writes this itself (do NOT
   reuse `sample._write_manifest`, whose schema differs): `stratum` is the composite
   `f"{tier}|{ntee_major_group}"` key and `ntee_major_group` is deliberately kept as
   its own column (stage-09 groups by it). Also write the coding template
   `anchor_to_code.csv` (`EIN2, tier, text, human_label` empty) with clobber
   protection: refuse if the template exists with any `human_label` filled, unless
   `force=True` (mirror stage-01's behavior).
Script: thin wrapper (`--config`, `--force`). Tests: allocation sums to n;
`sample_prob` consistent with draw counts; every tier present; LOW oversampled;
exclusion works incl. dtype drift; determinism (two runs identical); clobber
protection (with/without force); synthetic path (`allow_synthetic=True`) end-to-end
on `tiny_registry`.
Acceptance: Tier-1 green; `uv run python scripts/05_build_anchor.py --config
config/smoke.yaml` produces both artifacts in a scratch checkout.

**T1.4 — G4 gate + orchestrator wiring** (owns `src/binary_classifier/qc/preflight.py`,
`scripts/run_pipeline.py`, `tests/test_preflight.py` (extend),
orchestrator test (extend))
Operations: in preflight.py add `_validate_anchor_labels(cfg, registry) ->
list[str]` (template exists; every row has strict 0/1 `human_label`) and wire it into
`validate_gates` for stages `{"07", "09"}` (G4). Add `_STAGE_SPLITS` entries
`"06": "validation"`, `"07": "test"`. In run_pipeline.py: add `"05"` to
`_STAGE_MODULES` (§5.2); pass `force=` to stage 05 like stage 01; extend the G1 check
set from `{"02","04"}` to `{"02","04","06","07"}`; add the G4 check before stages
07/09 (G3 arrives in PR-3 — leave a marked TODO hook). Update the module docstring
gate narrative. Tests: G4 missing/partial/non-binary/complete; `_STAGE_SPLITS`
additions; `_STAGE_MODULES["05"]` import-resolves.
Acceptance: Tier-1 green.

**PR-1 gate**: all tasks green; `run_pipeline.py --config config/smoke.yaml
--stages 01,05` runs end-to-end on synthetic data (manual orchestrator check).

---

### PR-2 — `feature/06-training`

**Objective**: full stage 06 — training frame with soft targets, baselines, encoder
fine-tuning, OOF cross-fit, gated arms, run matrix, selection report.
Read §4.3 (recipe + skip list) and §4.5 (v5 API + device/precision policy) first.

**T2.1 — TrainingConfig replacement** (owns `config.py`, both YAMLs)
Operations: replace the stub (config.py:238–254) with §6.1 `TrainingConfig` +
`EncoderArm`; update `religious_missions.yaml` training block (remove `fp16`);
smoke.yaml already carries its block (verify it validates against the new model).
Acceptance: Tier-1 green (existing tests construct `BinaryClassifierConfig()` bare —
defaults must keep them passing).

**T2.2 — training frame** (owns `src/binary_classifier/train/__init__.py`,
`train/data.py`, `tests/test_training_data.py`)
Operations: implement
- `build_training_frame(cfg, registry) -> pd.DataFrame` with columns
  `EIN2, text, ntee_major_group, p_pos (float), hard_label (int)`:
  read `registry.silver_labels`; drop NaN `silver_label`; **recompute `p_pos` from
  the long store** (`registry.annotation_store` via `AnnotationStore`): per EIN2,
  `p_pos = (# rows with label==1) / (# rows with label in {0,1})` (NaN votes are
  abstains, excluded from numerator and denominator). Edge rule: an EIN2 present in
  `silver_labels.csv` with a non-NaN `silver_label` but ZERO non-NaN votes in the
  store indicates artifact corruption — raise `ValueError` with the EIN2 count.
  `hard_label = silver_label`;
  rehydrate `text`/`ntee_major_group` from `load_missions(cfg)` joined on EIN2;
  assert ZERO overlap with gold-manifest AND anchor-manifest EIN2s (string-normalized;
  raise with counts on violation).
- `split_dev(frame, dev_fraction, seed) -> (train_df, dev_df)` stratified by
  `hard_label` via `np.random.default_rng(seed)`.
- `subset_fraction(frame, fraction, seed) -> frame` — label-stratified, **nested**
  (sort by a seeded stable hash of EIN2, take the prefix) so 25% ⊂ 50% ⊂ 100%.
- `load_human_split(cfg, registry, split) -> pd.DataFrame` (`EIN2, text, human_label`)
  reading `gold_to_code.csv` filtered by split — **raises `ValueError` on
  `split=="test"`** (§5.5#1; stage 07 uses its own internal reader).
Tests: p_pos math against a fabricated store (incl. abstain votes and unanimous
rows); NaN-silver drop; overlap guard (incl. dtype-drift case); dev-split determinism
+ stratification; nested-subset property; test-split guard raises.
Acceptance: Tier-1 green.

**T2.3 — baselines** (owns `train/baselines.py`, `tests/test_baselines.py`) [parallel-ok after T2.2]
Operations:
- `tfidf_logreg(train_df, eval_dfs, seed) -> dict`: sklearn `FeatureUnion` of
  `TfidfVectorizer(analyzer="word", ngram_range=(1,2))` and
  `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))` →
  `LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)`;
  trains on `hard_label`.
- `minilm_logreg(...)`: embeddings via plain transformers (`AutoModel` +
  attention-masked mean pooling, `torch.inference_mode()`, device per §4.5 policy);
  cache to `registry.embeddings_dir / f"{model_slug}.npy"` aligned to a saved EIN2
  index file; LogReg head as above. **No sentence-transformers dependency.**
- Both score silver-dev and human-validation with
  `metrics.compute_metric_bundle(..., y_score=predict_proba[:,1])` and return rows
  conforming to the `results.jsonl` schema (§5.4, `model="baseline:..."`).
Tests: tfidf end-to-end on ~60 synthetic rows (sanity: PR-AUC > prevalence); minilm
with the embedding function monkeypatched to seeded random vectors (no download);
cache hit path.
Acceptance: Tier-1 green.

**T2.4 — encoder run** (owns `train/encoder.py`, `tests/test_encoder_args.py`)
Operations: implement
- `resolve_device(knob) -> str` and `resolve_precision(knob, device) -> str`
  EXACTLY per §4.5 device/precision policy (capability checks only — never platform
  detection).
- `soft_ce(outputs, labels, num_items_in_batch)` per this spec (labels = float
  p_pos tensor; `logp = log_softmax(logits, -1)`;
  `loss = -(labels*logp[:,1] + (1-labels)*logp[:,0]).sum() / num_items_in_batch`).
  Hard arm reuses it with binarized labels; class-weighted arm multiplies the two
  terms by `w1, w0` from inverse class frequencies (single code path, closure-built).
- `finetune(cfg, registry, encoder: EncoderArm, train_df, dev_df, *, targets,
  arm, train_fraction, seed, run_root=None) -> dict` (returns a `results.jsonl` row):
  `transformers.set_seed(seed)`; tokenizer with startup [CLS]/[SEP] assert (§4.5);
  HF `Dataset` with float `labels`; `DataCollatorWithPadding`;
  `TrainingArguments(output_dir=checkpoints_dir/<slug>/<arm>/s<seed>,
  eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
  metric_for_best_model="pr_auc", greater_is_better=True,
  warmup_steps=cfg.training.warmup_fraction, learning_rate=…, num_train_epochs=…,
  per_device_train_batch_size=…, weight_decay=…, save_total_limit=…,
  bf16=(precision=="bf16"), report_to=cfg.training.report_to or "none",
  disable_tqdm=True, logging_strategy="steps", logging_steps=50, seed=seed,
  data_seed=seed)`; `Trainer(processing_class=tokenizer, compute_loss_func=…,
  compute_metrics=…)` where `compute_metrics(eval_pred: transformers.EvalPrediction)
  -> dict` (namedtuple signature: `.predictions` = logits, `.label_ids` = float
  targets) computes PR-AUC (`average_precision_score` on softmax positive probs
  against BINARIZED dev labels) + minority-F1 + MCC, returned with key `"pr_auc"` so
  `metric_for_best_model="pr_auc"` resolves to `eval_pr_auc`;
  `EarlyStoppingCallback(early_stopping_patience=…)`. Note `warmup_steps` receives
  the float `warmup_fraction` directly — v5 interprets floats in [0,1) as a fraction
  of total steps.
  Per-run logging: attach `logging.FileHandler(runs_dir/<run_id>/train.log)` to the
  `transformers` and `binary_classifier` loggers for the run duration; console stays
  WARNING. After training: score human validation (bundle + bootstrap CIs), write
  `metrics.json`, return the row.
Tests (no model instantiation, no network): device/precision resolution matrix
(monkeypatch `torch.cuda.is_available`, `torch.cuda.is_bf16_supported`,
`torch.backends.mps.is_available` → cover cuda+bf16, cuda−bf16, mps, cpu);
`soft_ce` vs hand-computed values incl. the num_items normalization and the
class-weighted variant; TrainingArguments construction (warmup float passthrough,
metric wiring); run-dir/log layout.
Acceptance: Tier-1 green.

**T2.5 — OOF cross-fit** (owns `train/crossfit.py`, `tests/test_crossfit.py`)
Operations: `compute_oof_pred_probs(cfg, registry, frame, *, finetune_fn=None) ->
Path`: stratified K-fold (`cfg.training.crossfit_folds`, seeded) over the training
frame; per fold, train on K−1 folds (via `finetune_fn`, default = T2.4 `finetune`
with the selected recipe), predict the held-out fold; write `oof_pred_probs`
(§5.4 schema: every EIN2 exactly once). Resume: skip folds whose shard exists.
Tests: with `finetune_fn` stubbed (returns a predictor stub) — partition property
(every EIN2 exactly once, fold ids correct), determinism, resume, schema.
Acceptance: Tier-1 green.

**T2.6 — gated arms** (owns `train/arms.py`, `tests/test_arms.py`) [parallel-ok after T2.4/T2.5]
Operations: arm runners returning modified (train_df, loss-spec) pairs:
- `hard`: targets binarized.
- `pruned`: cleanlab confident-learning on OOF probs
  (`cleanlab.filter.find_label_issues(labels=hard_label, pred_probs=oof[[p0,p1]])`),
  intersect flags with low-vote-share rows (`p_pos` in (0.34, 0.66) — the
  disagreement band), drop the intersection (log count), retrain.
- `class_weighted`: closure weights from inverse class frequency of `hard_label`.
Tests: prune intersection logic on synthetic frames (cleanlab runs fine on ~100 rows);
weights math; each arm returns schema-conformant rows.
Acceptance: Tier-1 green.

**T2.7 — run matrix + selection** (owns `train/sweep.py`, `train/trainer.py`,
`scripts/06_train.py`, `tests/test_sweep.py`, `tests/test_e2e_stages_05_11.py`)
Operations:
- `run_training(cfg, registry, *, baselines_only=False, sweep=True, final=False,
  encoder=None, limit=None) -> None` (stage entrypoint): orchestrates
  baselines → documentation curve (`curve_fractions` × encoders × 1 seed, soft) →
  arm matrix at 100% ({soft} ∪ cfg.training.arms × primary encoder ×
  `sweep_seeds`; {soft} × comparison encoder × `sweep_seeds`) → (with `final=True`)
  winning cell × `final_seeds`. Every completed run appends to
  `learning_curve_results`; **resume = skip any run whose `metrics.json` exists**.
- Selection: `write_selection_report(...)` — per-arm seed-mean ± SD of validation
  PR-AUC and minority-F1 (+ CIs), tie-rule verdicts (§4.2), recommended
  (encoder, targets, arm). The HUMAN then writes `selected_model.json` (§5.4) — the
  script prints the exact JSON skeleton to copy; sha256 computed over
  `model.safetensors`.
- `scripts/06_train.py`: flags `--config --baselines-only --sweep --final
  --encoder ID --subset FLOAT --epochs N --seeds CSV --limit N` (overrides for local
  tiers; `--subset 0.5 --epochs 1 --seeds 42` is the documented local real-data run).
- Wire `"06"` into `_STAGE_MODULES`.
Tests: matrix enumeration (counts per §4.3: 2 baselines + 6 curve + 12+3 arms + 2
finals on the default config); resume-by-metrics.json; tie rule (synthetic seed
results: clear win, tie→simpler); report schema; `finetune` stubbed throughout.
ALSO CREATE `tests/test_e2e_stages_05_11.py` (marked `slow`): fabricates synthetic
missions + annotation store + silver labels + coded gold/anchor templates +
confirmed slate (the conftest pattern) on `tiny_registry`, then drives stages 05→06
in-process with `finetune` stubbed. **Ownership handoff: each later PR's
stage-entrypoint task (T3.4, T4.2, T5.5, T7.2) EXTENDS this file through its own
stage** — by PR-7 it covers 05→11 and is the Tier-2 offline route (§8).
Acceptance: Tier-1 green; smoke run `scripts/06_train.py --config config/smoke.yaml
--baselines-only` then `--sweep` completes locally (bert-tiny; `network` once to
cache).

**T2.8 — docs superseding notes** (owns `.agents/plans/we-work-on-the-floofy-wreath.md`
(append memo), `docs/agents/configuration.md`, `README.md`) [parallel-ok]
Operations: append a dated "Superseded decisions (June 2026)" memo: sweep
{0.5k..16k} → full-pool + {25/50/100%} documentation curve (cite arXiv:2504.15432,
Card et al. 2020); encoder grid −RoBERTa/−DistilBERT (cite arXiv:2504.08716);
soft-label default (cite §4.3 evidence); label-smoothing/focal/confidence-weighted
skip list. Update configuration.md roadmap hooks + README roadmap section to point at
stages 05–11 and this plan.
Acceptance: docs updated; no code touched.

**PR-2 gate**: all tasks green; Tier-2 steps 1–4 (§8) pass.

---

### PR-3 — `feature/07-evaluation`

**Objective**: calibrator + threshold from the anchor (cross-fit), rule validation,
G3, one-shot frozen-test evaluation with acceptance verdict.

**T3.1 — config + G3 plumbing** (owns `config.py`, `qc/preflight.py`,
`scripts/run_pipeline.py`, both YAMLs, `tests/test_evaluation_gate.py`)
Operations: add `EvaluationConfig`, `AcceptanceCriteria`, `TestUnlock`,
`load_test_unlock` (§6.1) + root field + YAML blocks. Preflight: `_validate_test_unlock
(cfg, registry) -> list[str]` (file exists; `confirmed`; acceptance snapshot ==
current config acceptance — field-by-field; sha matches `selected_model.json`'s);
wire as G3 for stage 07 at the orchestrator TODO hook from T1.4; add `"07"` to
`_STAGE_MODULES`. Tests: G3 variants (missing / unconfirmed / acceptance drift /
sha mismatch / happy path).
Acceptance: Tier-1 green.

**T3.2 — calibration** (owns `evaluation/__init__.py`, `evaluation/calibration.py`,
`tests/test_calibration.py`)
Operations: implement
- `fit_platt(scores, labels) -> {a, b}` (logistic on logit(score); use
  sklearn LogisticRegression on the 1-D logit feature) and
  `fit_temperature(scores, labels) -> {T}` (scalar NLL minimization via
  `scipy.optimize.minimize_scalar` — scipy ships with sklearn; if unavailable, golden
  section by hand);
- `crossfit_calibrate(scores, labels, folds, methods, seed) -> (oof_calibrated,
  per_method_metrics)` — OOF discipline per §4.2;
- metrics: Brier, log-loss, ECE (equal-width bins, `cfg.evaluation.ece_bins`),
  reliability-curve points (serializable);
- winner = best mean OOF Brier (log-loss tiebreak); refit winner on ALL anchor rows
  → deployed params.
Tests: **Platt absorbs a synthetic prior shift, temperature does not** (construct
scores calibrated under 30% prior, evaluate against 13%-prior labels — assert Platt
OOF Brier < temperature OOF Brier); ECE/Brier hand-checked values; OOF discipline
(no in-fold fitting); round-trip serialize/deserialize.
Acceptance: Tier-1 green.

**T3.3 — thresholds + subgroups + decision curve** (owns `evaluation/thresholds.py`,
`evaluation/subgroups.py`, `evaluation/decision_curve.py`, `tests/test_thresholds.py`,
`tests/test_subgroups.py`, `tests/test_decision_curve.py`) [parallel-ok with T3.2]
Operations:
- `pick_threshold(probs, labels, policy, precision_floor) -> {threshold, policy,
  achieved_precision, achieved_recall, max_f1_threshold, pr_curve_points}` —
  precision floor: sweep PR curve, among thresholds with precision ≥ floor take max
  recall; documented fallback if floor unattainable: threshold at max precision, flag
  `floor_unattainable: true`.
- `subgroup_report(df, y_true, y_pred, y_prob, *, by, length_bins, min_n)` — per
  NTEE major group and word-count bin: n, minority-F1, FPR, FNR; suppress (report
  `suppressed: true`) below `min_n`.
- `net_benefit(y_true, y_prob, thresholds) -> points` — NB(t) = TP/n − FP/n·t/(1−t),
  plus treat-all/treat-none reference lines (Vickers & Elkin 2006). Report-only.
Tests: hand-computed PR points / floor policy / fallback; subgroup suppression;
net-benefit values vs manual computation.
Acceptance: Tier-1 green.

**T3.4 — stage entrypoint** (owns `evaluation/evaluate.py`, `scripts/07_evaluate.py`,
`tests/test_evaluate_stage.py`)
Operations: `run_evaluation(cfg, registry, *, predictor=None) -> None` in this exact
order:
1. Load + verify `selected_model.json` (sha256 of checkpoint file matches; hard error
   with guidance if absent/mismatched). `predictor` kwarg (default: load the
   checkpoint; tests inject a stub with `predict_proba(texts) -> (n,2)`).
2. G4 re-check (anchor coded).
3. Predict raw probs on all anchor rows (join text via `load_missions`); run T3.2
   cross-fit; write `calibrator_path` + `anchor_oof_scores` (§5.4).
4. Threshold via T3.3 on the OOF-calibrated anchor scores; store in calibrator.json.
5. Rule validation: anchor LOW cells → `apply_rule_label` vs human labels →
   sens/spec/precision/recall + Wilson CIs → `rule_validation` (§5.4).
6. G3 re-check. **One-shot guard**: if `test_evaluation.json` exists → raise with
   "delete it explicitly to re-run" (loud, auditable; single-researcher acceptable).
7. Frozen-test eval (test split read HERE only, via an internal reader — not the
   T2.2 loader): discrimination bundle from `metrics.compute_metric_bundle` with
   `bootstrap_resamples`; subgroups; net-benefit points; calibration metrics on the
   anchor OOF (NOT on test — test is boundary-enriched, §3.3).
8. Acceptance verdict (min_pr_auc + min_minority_f1_ci_lower on test; max_ece on
   anchor OOF) — on failure raise like stage-04's freeze gate. Write
   `test_evaluation.json` (§5.4 incl. metadata).
Tests: full happy path with injected predictor stub + fabricated anchor/gold
artifacts on `tiny_registry`; one-shot refusal; acceptance failure raises; ordering
(calibrator written before any test read — assert via file mtimes or call recording).
Extend `tests/test_e2e_stages_05_11.py` through stage 07 (writes a confirmed
`test_unlock.json` fixture).
Acceptance: Tier-1 green; Tier-2 step 5 (§8) passes.

**PR-3 gate**: all green; smoke E2E through stage 07.

---

### PR-4 — `feature/08-inference`

**T4.1 — config + router** (owns `config.py` (InferenceConfig + root + YAMLs),
`inference/__init__.py`, `inference/router.py`, `tests/test_inference_router.py`)
Operations: `route(text, tier, cfg) -> tuple[str, int | None]` pure function
implementing: tier in {HIGH, MEDIUM} → `("classifier", None)`; tier LOW and
`route_low_to_rules`: `apply_rule_label(text)` → 1 ⇒ `("rule_strong_positive", 1)`,
0 ⇒ `("rule_short_negative", 0)`, None ⇒ `("low_via_classifier", None)` if
`rule_ambiguous_to_classifier` else `("rule_abstain", None)`. Truth-table tests for
every branch.
Acceptance: Tier-1 green.

**T4.2 — predictor + sharding** (owns `inference/predict.py`, `scripts/08_infer.py`,
`tests/test_predict_stage.py`)
Operations: `run_inference(cfg, registry, *, predictor=None, limit=None) -> None`:
1. `load_missions(cfg)` (FULL corpus); compute Q/tier; apply T4.1 router.
2. Load `calibrator.json`; resolve device per §4.5 policy.
3. Shard into `cfg.inference.shard_size` chunks (stable order by EIN2). Per shard:
   classifier-routed rows → tokenize/batch (`cfg.inference.batch_size`),
   `torch.inference_mode()`, autocast bf16 only when resolved device is cuda+bf16;
   apply calibrator + threshold; rule rows get `prob_raw/prob_calibrated = NaN` and
   the router's label. Write `predictions_dir/shards/shard_{i:05d}.parquet`
   (§5.4 schema, all metadata columns stamped). **Resume = skip existing shards.**
4. Merge shards → `predictions_parquet`.
5. **Monitor scoring**: predict the monitor split (EIN2s from `monitor_manifest`),
   write `monitor_scores.json` (per-row calibrated probs + run metadata) for
   run-over-run drift diffing (§4.4).
Also wire `"08"` into `_STAGE_MODULES` (this task owns the run_pipeline.py edit for
PR-4; pass `limit=` through like stage 03 does).
Script flags: `--config --limit N`. Tests (predictor stub): schema incl. NaN rule
rows + decision_source mix; shard resume (pre-write a shard, assert skipped); EIN2
completeness (merged == input); monitor scores written; metadata stamped. Extend
`tests/test_e2e_stages_05_11.py` through stage 08.
Acceptance: Tier-1 green; Tier-2 step 6 first command.

**PR-4 gate**: all green; smoke inference produces rule-routed LOW rows + monitor file.

---

### PR-5 — `feature/09-prevalence`

**T5.1 — deps** (owns `pyproject.toml`, `uv.lock`)
Operations: add `ppi-py>=0.2.3`, `quapy>=0.2.0,<0.3`; extend
`[tool.deptry.package_module_name_map]` with `ppi-py = ["ppi_py"]` (quapy imports as
`quapy`, no entry needed); `uv lock && uv sync`; smoke imports:
`uv run python -c "import ppi_py, quapy"` (numba and abstention are the known risks,
§4.5). If quapy fails: record DEVIATION, skip the dep, and rely on T5.4's vendored
EMQ (KDEy becomes `importorskip`-optional).
Acceptance: lock clean; imports OK (or documented fallback engaged); Tier-1 green.

**T5.2 — config + weights** (owns `config.py` (PrevalenceConfig + root + YAMLs),
`prevalence/__init__.py`, `prevalence/weights.py`, `tests/test_prevalence_weights.py`)
Operations: `design_weights(anchor_manifest_df, *, normalize=True) -> pd.Series`
(`w = 1/sample_prob`, normalized to mean 1 over the labeled set); join helper to align
weights/labels/predictions by EIN2 (string-normalized). Tests: weight math, alignment,
missing-EIN2 handling (drop + warn count).
Acceptance: Tier-1 green.

**T5.3 — PPI wrapper** (owns `prevalence/ppi.py`, `tests/test_ppi.py`)
Operations: `ppi_prevalence(y_labeled, yhat_labeled, yhat_unlabeled, *, alpha, w=None)
-> {estimate, ci_lower, ci_upper, lam, n_labeled, n_unlabeled, weighted}` calling
`ppi_py.ppi_mean_pointestimate` / `ppi_mean_ci` with `alpha=cfg.prevalence.alpha`
**explicitly** (library default is 0.1, §4.5) and `w=` when provided; `yhat` are
calibrated probabilities (floats — PPI on the mean of Y via prob predictions is the
standard prevalence use). The wrapper's returned dict MUST include the `alpha`
actually used. Tests: synthetic population with known prevalence — CI covers truth
across seeds; weighted vs unweighted differ when weights are skewed;
**assert `result["alpha"] == cfg.prevalence.alpha`** (guards the library's 0.1
default silently producing 90% CIs); `importorskip("ppi_py")`.
Acceptance: Tier-1 green.

**T5.4 — quantification cross-checks** (owns `prevalence/quantify.py`,
`tests/test_quantify.py`) [parallel-ok]
Operations: `emq_prevalence(val_posteriors, val_labels, corpus_posteriors,
max_iter=1000, tol=1e-6) -> float` — **vendor the SLD/EMQ EM loop directly**
(~40 lines: iterate prior re-weighting of posteriors until convergence; Saerens et
al. 2002) so the cross-check never depends on quapy resolving; ALSO provide
`quapy_emq_prevalence(...)` and `kdey_prevalence(...)` behind
`importorskip`-style guards using the 0.2.0 API (`EMQ(clf, fit_classifier=False)`,
`aggregate(posteriors)`; wrap our precomputed posteriors in a minimal sklearn-style
shim). Tests: vendored EMQ recovers a known prior shift on synthetic posteriors;
quapy parity test guarded by importorskip.
Acceptance: Tier-1 green.

**T5.5 — composite + entrypoint** (owns `prevalence/composite.py`,
`prevalence/estimate.py`, `scripts/09_prevalence.py`, `tests/test_prevalence_stage.py`)
Operations:
- `rogan_gladen(p_obs, sens, spec) -> float` + variance propagation (delta method) +
  clipping to [0,1];
- `composite(prev_by_stratum: dict[str, (est, var, share)]) -> (est, var)`
  (§4.2 formulas);
- `run_prevalence(cfg, registry) -> None`: G4 assumed (orchestrator); inputs =
  `predictions_parquet`, `anchor_oof_scores`, `rule_validation`, `anchor_manifest`;
  compute: HM stratum via T5.3 (labeled = anchor HM rows with OOF-calibrated scores;
  unlabeled = corpus HM calibrated probs; weighted AND unweighted variants); LOW
  stratum via rule labels + Rogan–Gladen with T3.4's sens/spec (+ sensitivity band
  over their Wilson CIs when `low_tier_sensitivity`); composite with corpus tier
  shares; cross-checks (T5.4) on the same inputs; per-NTEE loop with
  `ntee_min_n` suppression (EMQ point-estimate fallback for suppressed groups,
  flagged). Write `prevalence_report` + `prevalence_by_ntee` (§5.4) with explicit
  estimand statements.
Tests: Rogan–Gladen vs hand-computed; composite variance; full stage on fabricated
artifacts (known truth within CI); suppression behavior; report schema. Extend
`tests/test_e2e_stages_05_11.py` through stage 09.
Acceptance: Tier-1 green; Tier-2 step 6 second command. Wire `"09"` into
`_STAGE_MODULES` (this task owns the run_pipeline.py edit for PR-5).

**PR-5 gate**: all green; smoke E2E through stage 09.

---

### PR-6 — `feature/10-viz`

**T6.1 — figures** (owns `viz/__init__.py`, `viz/ngrams.py`, `viz/curves.py`,
`viz/prevalence_plots.py`, `tests/test_viz.py`)
Operations: pure functions, each `(data, ax: matplotlib.axes.Axes) -> None`, reading
ONLY stage artifacts (§5.4) — no model loading, no torch import:
- `ngram_log_odds(silver_df_with_text, top_k=30)`: CountVectorizer (1–2 grams,
  min_df=5) on religious vs non silver rows; log-odds with +1 smoothing; horizontal
  bar chart (substitutes the roadmap's word clouds — see §4.4 docs note).
- `documentation_curve(results_jsonl_rows)`: validation PR-AUC vs train_fraction per
  encoder, seed bands where >1 seed.
- `pr_curve(points)`, `reliability_diagram(points, ece)`: from
  `test_evaluation.json` / `calibrator.json` serialized points.
- `prevalence_forest(prevalence_by_ntee_df)`: estimates + CIs per NTEE group,
  suppressed groups greyed.
Tests: each renders to a tmp PNG under the `Agg` backend from fabricated inputs.
Acceptance: Tier-1 green.

**T6.2 — script + docs** (owns `scripts/10_visualize.py`,
`docs/agents/pipeline.md` (viz note), README viz note) [parallel-ok]
Operations: script renders every figure whose input artifact exists (skip + log
otherwise) into `figures_dir` as PNG and SVG; `--config` flag only. Docs: record the
word-cloud → log-odds substitution rationale.
Acceptance: Tier-1 green; smoke run renders ≥1 figure after Tier-2.

**PR-6 gate**: all green.

---

### PR-7 — `feature/11-aggregation-compare`

**T7.1 — config + unlock implementations** (owns `config.py` (AggregationConfig +
root + YAMLs), `src/binary_classifier/annotate/aggregate.py`,
`tests/test_aggregate_unlock.py`)
Operations: implement the quarantined arms (replacing the `NotImplementedError`
bodies at lines 85–135; PRESERVE the quarantine behavior when prerequisites are
missing):
- `aggregate_crowdlab(df, pred_probs: pd.DataFrame | None = None)`: if
  `pred_probs is None` → raise the existing quarantine error. Else: pivot the long
  store to an (EIN2 × source_id) label DataFrame (NaN = abstain/missing — cleanlab
  handles it, §4.5); align `pred_probs[["p0","p1"]]` by EIN2 (drop + warn on rows
  missing predictions); call `get_label_quality_multiannotator(labels, probs,
  quality_method="crowdlab")`; map the returned consensus + quality back to the
  EXACT majority_vote wide schema (§3.2: `EIN2, silver_label, silver_confidence
  (=consensus quality), num_votes, num_abstain, agreement, tie(=False)`).
- `aggregate_dawid_skene(df)`: long → `(task=EIN2, worker=source_id, label)` frame →
  `crowdkit DawidSkene(n_iter=100).fit_predict_proba`; same wide mapping; KEEP the
  correlated-annotators caveat in the docstring.
- `aggregate_labels(df, method, pred_probs=None)` dispatch grows the optional kwarg.
Tests: pivot shape/NaN handling on a fabricated store; drop-in schema equality vs
`majority_vote` columns; quarantine preserved (`crowdlab` without pred_probs raises);
small-N end-to-end for both arms (cleanlab/crowdkit run on ~20 rows).
Acceptance: Tier-1 green.

**T7.2 — comparison report** (owns `qc/aggregation_compare.py`,
`scripts/11_aggregation_compare.py`, `tests/test_aggregation_compare.py`)
Operations: `run_aggregation_compare(cfg, registry) -> None`: load the long store +
`oof_pred_probs` (PR-2 artifact — the out-of-sample probs cleanlab requires); for
majority + each arm in `cfg.aggregation.comparison_arms`: aggregate, join human
**validation** labels (template, `split=="validation"`), score with
`metrics.compute_metric_bundle` + bootstrap CIs; write `aggregation_compare`
(§5.4) including the **adoption rule verdict**: an arm may replace majority ONLY if
its minority-F1 CI lower bound > majority's point estimate (standing rule,
configuration.md), and the report MUST state that adoption invalidates the frozen
`silver_labels.csv` ⇒ re-run stages 04→06. Never auto-switch
`cfg.aggregation.method`.
Tests: fabricated store + OOF probs + coded validation → report schema, verdict
logic both ways. Extend `tests/test_e2e_stages_05_11.py` through stages 10–11
(viz renders + comparison report — completing the offline Tier-2 route).
Acceptance: Tier-1 green; smoke run after Tier-2 produces the report.

**T7.3 — docs** (owns configuration.md weak-supervision hooks, README) [parallel-ok]
Operations: update the "future weak-supervision arms" hook: CROWDLAB/Dawid-Skene now
implemented as gated comparison arms; adoption rule unchanged.
Acceptance: docs only.

**PR-7 gate**: all green.

---

## 8. Verification ladder (LOCAL, platform-agnostic — device/precision always resolved at runtime per §4.5; UCloud is production, not a test tier)

**Tier 1 — offline unit tests (every commit, seconds):**
```
uv run pytest -m "not slow and not network"
uv run ruff check . && uv run ruff format --check . && uv run ty check
```
No downloads, no accelerator assumptions, no API keys.

**Tier 2 — local synthetic E2E (~5–10 min, per PR from PR-2 on):**
1. One-time per machine: `uv run pytest -m network` (caches bert-tiny + tokenizer asserts).
2. Stages 01–04 artifacts: EITHER the offline route — run the integration test
   `tests/test_e2e_stages_05_11.py` (marked `slow`; fabricates synthetic missions,
   annotation store, silver labels, coded gold + anchor templates, confirmed
   slate/unlock JSONs via the conftest pattern, then drives stages 05→11 in-process
   with a stubbed predictor) — OR the live route with `OPENAI_API_KEY`:
   `run_pipeline.py --config config/smoke.yaml --stages 01` → fill
   `gold_to_code.csv` programmatically → `--stages 02,03,04 --annotate-limit 10`.
3. `run_pipeline.py --config config/smoke.yaml --stages 05` → fill
   `anchor_to_code.csv` programmatically.
4. `scripts/06_train.py --config config/smoke.yaml --baselines-only` then `--sweep`
   (bert-tiny, 1 epoch, fp32, on whatever accelerator resolves) → write
   `selected_model.json` from the printed skeleton.
5. Write confirmed `test_unlock.json` → `scripts/07_evaluate.py --config
   config/smoke.yaml`; immediately re-run and confirm the one-shot refusal.
6. `scripts/08_infer.py --config config/smoke.yaml --limit 500` (assert rule-routed
   LOW rows + `monitor_scores.json` exist) → `scripts/09_prevalence.py` →
   `scripts/10_visualize.py` → `scripts/11_aggregation_compare.py`.
7. `uv run pytest -m slow`.

**Tier 3 — local real-data subset (~1 h, pre-merge bar for PR-2/3/4):**
- Baselines on the real ~20k silver pool (TF-IDF minutes; MiniLM embed+LR ~10–15 min
  CPU, faster with a local accelerator).
- `scripts/06_train.py --encoder microsoft/deberta-v3-base --subset 0.5 --epochs 1
  --seeds 42` — validates the real tokenizer, the soft-target path, and training
  stability on the locally resolved device (CUDA bf16, MPS fp32, or CPU fp32; wall
  time varies accordingly — that is expected and fine).
- `scripts/08_infer.py --limit 5000` with the tier-3 checkpoint.

**Tier 4 — production (UCloud B200, after merges; not testing):**
1. `uv sync` (no `serve` extra); check `torch.version.cuda` ≥ cu128 and
   `torch.cuda.is_bf16_supported()`.
2. Stage 05 → human codes the 500-row anchor (in parallel with step 3).
3. Stage 06 full matrix (~25–30 runs; tail `data/models/runs/*/train.log`); review
   `selection_report.json`; `--final`; commit `selected_model.json`.
4. Declare + commit acceptance criteria; commit `test_unlock.json`; run stage 07 once.
5. Stage 08 over 560k (sharded, resumable — `/work` persists); stages 09–11; review
   the aggregation report (majority retained unless beaten per the adoption rule).

---

## 9. Risks and mitigations

1. **ppi_py (numba) / quapy (abstention) on Py3.13 + numpy 2** — import smoke tests
   at PR-5 lock; vendored EMQ makes the cross-check independent of quapy; KDEy
   optional behind importorskip.
2. **torch ≥2.7 resolution on a given local platform** — verified risk-low; fallback
   documented in §6.4/PR-0 T0.1.
3. **DeBERTa tokenizer regression class of bugs** — guarded twice (network test +
   runtime assert in encoder.py).
4. **Anchor is single-coder** — same limitation as gold; flagged in the prevalence
   report; codebook boundary rules mitigate.
5. **Per-NTEE CIs at n=500 are wide** — suppression below n=10 + EMQ fallback; user
   accepted the trade-off.
6. **One-shot test discipline is socially enforced** — G3 + refuse-if-exists; a
   delete is possible but loud and auditable.
7. **Soft-target reconstruction depends on store schema** — pinned by tests against
   the §3.2 `AnnotationStore.COLUMNS` fixture.
8. **Cross-PR file contention** — prevented by per-task file ownership (§2); shared
   files have exactly one owner per PR.
9. **Local toolchain arch mismatch (OBSERVED on this machine, 2026-06-12)** — uv
   resolved the platform as `macosx_*_x86_64` (Rosetta x86_64 uv/python on Apple
   Silicon), for which torch ships no wheels ⇒ `uv sync` fails before any code runs.
   PR-0 T0.1 step (2) is the mandatory fix (native arm64 CPython + fresh `.venv`).

## 10. Reference index

In-repo: `.agents/docs/20260606-tech-short-text-model-alternatives.md` (grids,
baselines) · `…-imbalanced-text-evaluation.md` (metric bundle) ·
`…-calibration-quantification-prevalence.md` (calibration/quantification) ·
`…-llm-weak-supervision-noisy-labels.md` (noisy-label gating) ·
`.agents/plans/we-work-on-the-floofy-wreath.md` (+ annex; original locked decisions)
· `docs/agents/{configuration,pipeline}.md` · `docs/RUNNING_ON_UCLOUD.md`.

Literature (key): PPI Science 2023 arXiv:2301.09633 · PPI++ arXiv:2311.01453 ·
Stratified PPI arXiv:2406.04291 · DSL arXiv:2306.04746 · Alexandari ICML 2020
(calibrate-then-EM) · Saerens et al. 2002 (SLD/EMQ) · Rogan & Gladen 1978 ·
Kumar/Liang/Ma NeurIPS 2019 (calibration sample complexity) · Card et al. EMNLP 2020
(dev-set power) · Mosbach ICLR 2021 (stable fine-tuning) · arXiv:2504.15432
(LLM-label plateaus/risks) · arXiv:2511.14117 + Davani TACL (soft labels) ·
arXiv:2403.14715 (label smoothing harms ranking) · Henning EACL 2023 (imbalance) ·
Northcutt JAIR 2021 (confident learning) · CROWDLAB arXiv:2210.06812 ·
arXiv:2504.08716 (DeBERTa-v3 vs ModernBERT) · Chicco & Jurman 2020/2023 (metric
choice) · Vickers & Elkin 2006 (net benefit).
