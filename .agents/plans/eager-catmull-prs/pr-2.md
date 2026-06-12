# PR-2 — `feature/06-training`

> Standalone work order for the per-PR orchestrating agent, extracted VERBATIM from
> §7 of `.agents/plans/we-are-still-working-eager-catmull.md` (2026-06-12). All
> §-references resolve in `CONTEXT.md`; T-numbers of other PRs resolve in their
> `pr-N.md`. The binding orchestration protocol — subagent input/report contracts,
> Ralph iteration rules, state/status files, smoke-vs-production guardrails — is
> `ORCHESTRATOR.md`. Read `CONTEXT.md` §1–§5 (plus the §6 blocks listed below) and
> `ORCHESTRATOR.md` in full before spawning any subagent.

| | |
|---|---|
| Branch | `feature/06-training` |
| Depends on | PR-1 — sentinels: `src/binary_classifier/metrics.py` exists; `PathRegistry` has `anchor_manifest`, `runs_dir`, `oof_pred_probs`, `embeddings_dir` properties; `config/smoke.yaml` loads via `load_config` |
| Blocks | PR-3 (and PR-6, PR-7, which consume its artifacts) |
| Spec blocks implemented | §6.1 `TrainingConfig` + `EncoderArm`; §5.4 `results.jsonl` / `selection_report` / `selected_model` / `oof_pred_probs` schemas |
| Required first read | §4.3 (recipe + skip list) and §4.5 (v5 API + device/precision policy) — per the objective below |
| Ralph state | `.agents/ralph/state/pr-2.md` + `pr-2.status` |

**Pre-flight (first iteration):** verify the PR-1 sentinels above; switch to / create
the branch.

**Cross-PR ownership note:** T2.7 CREATES `tests/test_e2e_stages_05_11.py`; the
stage-entrypoint tasks of later PRs (T3.4, T4.2, T5.5, T7.2) each EXTEND it through
their own stage (stated in T2.7).

**Human checkpoints:** Tier-2 step 1 (`uv run pytest -m network`, once per machine)
needs HF Hub access. The smoke `selected_model.json` written from the printed
skeleton during Tier-2 step 4 is SMOKE-ONLY (ORCHESTRATOR.md A.5) — the production
one is human-authored after reviewing `selection_report.json`. §8 Tier-3 (real-data
subset, ~1 h) is part of the pre-merge bar for this PR — human runs or waives it at
the boundary (A.5). Review the T2.8 superseding-docs memo and any DEVIATIONS rows
when gating.

**Task conventions (§7 preamble, verbatim):**

Conventions for every task below: follow §3.1; docstrings Google style; type hints
required (ty must pass); new tests use `tiny_config`/`tiny_registry` and fabricate
inputs per-test; stage entrypoints are `run_x(cfg, registry, **kw)`.

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
(append memo), `.agents/architecture/configuration.md`, `README.md`) [parallel-ok]
Operations: append a dated "Superseded decisions (June 2026)" memo: sweep
{0.5k..16k} → full-pool + {25/50/100%} documentation curve (cite arXiv:2504.15432,
Card et al. 2020); encoder grid −RoBERTa/−DistilBERT (cite arXiv:2504.08716);
soft-label default (cite §4.3 evidence); label-smoothing/focal/confidence-weighted
skip list. Update configuration.md roadmap hooks + README roadmap section to point at
stages 05–11 and this plan.
Acceptance: docs updated; no code touched.

**PR-2 gate**: all tasks green; Tier-2 steps 1–4 (§8) pass.

