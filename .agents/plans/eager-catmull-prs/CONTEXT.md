# Shared Context Pack — Pipeline Roadmap Completion (Stages 05–11)

> Extracted 1:1 (mechanically, by line range) from the frozen source plan
> `.agents/plans/we-are-still-working-eager-catmull.md` on 2026-06-12. Sections keep
> their ORIGINAL numbering (§1, §3–§6, §8–§10) so every §-reference in the PR work
> orders resolves here unchanged. §2 lives verbatim in `ORCHESTRATOR.md` (Part 1);
> §7 is split verbatim into `pr-0.md` … `pr-7.md` (same directory). T-numbers
> (T1.2, T5.5, …) resolve in those PR documents.
>
> **Audience**: per-PR orchestrating agents and their specialized subagents, all
> starting with FRESH CONTEXT. Every code fact cited here was verified against the
> repo on 2026-06-12 (branch `refactor/harmonize-pipeline`). Read §1–§5 before
> executing any PR. Each PR work order is independently executable given §1–§5
> (plus the §6 blocks it implements).
>
> **Living overlay**: if executing an earlier PR changed any fact below, the change
> is recorded in `.agents/ralph/state/DEVIATIONS.md`. Read it together with this
> pack; on conflict, DEVIATIONS.md wins.

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

> Relocated verbatim to `ORCHESTRATOR.md` (same directory), Part 1 — together with
> the binding opencode/Ralph execution adaptations (Part 2). Every "§2" reference in
> this pack or in a PR work order resolves there.

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
  `.agents/architecture/configuration.md` (roadmap hooks), and `README.md`; PR-6
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

> Split verbatim into the standalone documents `pr-0.md` … `pr-7.md` (same
> directory) — one per PR. Every reference of the form "§7 PR-N TN.x" resolves to
> `pr-N.md`. The task-conventions preamble that opened §7 is reproduced verbatim at
> the top of every PR document.

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
· `.agents/architecture/{configuration,pipeline}.md` · `docs/RUNNING_ON_UCLOUD.md`.

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
