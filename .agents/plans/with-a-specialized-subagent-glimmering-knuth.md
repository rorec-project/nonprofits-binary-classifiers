---
created: 2026-06-10
---

# Sprint: harmonize the weak-supervision pipeline + DVC-ready data flow

This document is BOTH the full plan (reference sections) AND an executable sprint (ordered task
cards). It is designed to be run by an **orchestrator** that dispatches one **sequential subagent
per task card**. Tasks are serialized because several touch the same files.

---

## Orchestrator protocol (how to run this sprint)

- **Sequential.** Execute task cards strictly in order `T1 → T12`. Do not parallelize — shared files
  (`run_annotation.py`, `agreement.py`, `sample.py`) are touched by multiple tasks and must be serialized.
- **Each task = one fresh subagent.** When dispatching task `Tk`, give the subagent, in its fresh context:
  1. **This entire document** (all reference sections + all task cards) — so it has full context.
  2. A directive that it must implement **only** task `Tk`'s card — nothing else.
  3. The **report-back contract** below.
- **Subagent procedure (every task):**
  1. Read this whole document first; locate its task card `Tk`.
  2. Read the exact files named in the card (and their tests) before editing.
  3. Implement **only** the card's operations. Touch no files outside the card's "Files" list.
  4. Add the card's "Tests to add"; run `pytest -q` and any task-specific verification in the card.
  5. Apply the **commenting & citation convention** (next section) to all new/changed code.
  6. Report back per the contract.
- **Orchestrator gating:** after each report, confirm the contract is satisfied (tests green, files match,
  no scope creep). On any failure/BLOCKED, **stop and surface to the human** — do not dispatch the next task.
  On PASS, dispatch the next card.
- **Report-back contract (every subagent returns):**
  - Task ID + `PASS` or `BLOCKED`.
  - Files changed (exact paths) + one line per change.
  - New tests (names) + the full `pytest -q` summary line (passed/failed counts).
  - Task-specific verification commands run + their output.
  - Any deviation from the card + why; any newly-discovered risk/follow-up for the orchestrator.
  - Confirmation that the commenting/citation convention was applied and that nothing outside the card's
    "Files" list was modified.
- **Line numbers are indicative (pre-sprint baseline = current branch).** As sequential tasks edit shared
  files, line numbers shift. Always locate an edit site by its **function/symbol name** (stable), not the exact
  line; re-read the file before editing (already required). Re-run `pytest -q` to confirm the baseline (69
  passed) before T1.
- **Follow existing repo conventions.** Before writing code, read `docs/agents/conventions/comments.md`
  and `docs/agents/conventions/python-standards.md`, and match the surrounding style (Pydantic
  configs, the `PathRegistry` pathlib pattern, Google-style docstrings, seeded determinism).
- **Citations are locked in the References appendix** at the end of this document — use those exact
  identifiers in code comments / docs; do not invent or rely on outside memory.
- **Supporting evidence (optional, NOT required to implement — cards are self-contained).** For deeper
  rationale a subagent MAY consult the durable audit findings committed in this repo under `.agents/plans/`:
  `*-agent-a7984f9188a75eba4.md` (skeptical review), `*-agent-a2383f9b609ae4b96.md` and
  `*-agent-ab65966ce63b3fc30.md` (sampling/prevalence), `*-agent-a9fe574aeb844e218.md` (weak-supervision
  currency), `*-agent-a599b7180c7125ada.md` (modeling currency), `*-agent-a96ed53168b7a4933.md` (drift check);
  and the design notes `docs/research/20260605-replication-*.md`.

## Cross-cutting convention: generous commenting + method citations (applies to EVERY task)

- **Comment generously** — this is a research codebase reviewed by a human. Every new or changed function
  gets a docstring stating **what it does and why**; non-obvious logic gets inline comments.
- **Cite the authoritative source for any statistical/ML method**, inline, next to the implementing code, so
  the human reviewer can verify currency — e.g.
  `# Horvitz–Thompson design weight = 1/π (Horvitz & Thompson 1952)` or
  `# κ ≥ 0.70 "substantial" agreement (Landis & Koch 1977)` or
  `# PPI++ (Angelopoulos et al. 2023, arXiv:2311.01453)`.
- Err on the side of **more** explanation. These comments are a deliverable, not noise; the reviewer will
  prune if needed.

---

## Context

The `refactor/harmonize-pipeline` branch renamed the data layout and reshaped config in code, but the change
only half-landed (docs, `.gitignore`, and the on-disk data dir still reflect the old world), and several
weak-supervision features are wired but not operational. This sprint was driven by: an audit (3 Explore
agents), an independent skeptical review, authoritative DVC / label-aggregation research, a sampling-design
review, and a 2026 literature-currency audit (durable findings in `.agents/plans/*-agent-*.md`). All claims
were re-verified against the code on this branch (baseline `pytest -q` = 69 passed).

**Blocking pre-existing bug (fixed by T1–T3).** The human-coded **validation set is never LLM-annotated**
(annotation runs on the silver manifest only, run_annotation.py:117), yet the stage-04 freeze gate
**inner-joins silver labels against validation and raises if the overlap is empty** (agreement.py:111-122).
`build_silver_pool` (sample.py:362) and `build_gold_set` (sample.py:372) are independent draws; validation is
split out of gold (sample.py:379). With silver=20k vs validation≈175 from a large pool the overlap is ~0, so
the gate freezes nothing on a real run — masked today only by tiny-pool synthetic smoke.

## Target architecture (end-to-end, incl. future stages)

A reproducible, config-driven binary text classifier labelling US nonprofit records as `religious` (1) vs not
(0) via **LLM-as-primary weak supervision**, then a fine-tuned encoder. Entity-agnostic (a new YAML swaps
`entity`/`field`/`label_name`). Stages 01–04 are **DONE**; 05–08 are **FUTURE/roadmap**.

**Data layout (cookiecutter-data-science):** `data/raw/` (immutable upstream parquets, gitignored/cloud) →
`data/interim/` (manifests, bakeoff, `annotation_store.csv`; cloud-symlinked) → `data/processed/`
(`silver_labels.csv`, gitignored; `gold/` git-committed: `gold_to_code.csv`, `production_slate.json`) →
`data/models/` (future checkpoints). `EIN2` is the join key carried through every artifact.

**Quality rubric Q & tiers** (`quality.py`, max 6.0): HIGH ≥5.0 / MEDIUM 3.0–<5.0 / LOW <3.0. The sampling
**frame is HIGH+MEDIUM** (Q≥3.0); LOW is excluded from sampling and handled by a high-precision **rule layer**
at inference (`apply_rule_label`): strong-tradition lexicon → 1; very short + no religious lexicon → 0; else
abstain → classifier.

| Stage | Name               | Status | In → Out                                                                                            | Gate                                                                         |
| ----- | ------------------ | ------ | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 01    | Build sample       | DONE   | upstream parquets → silver/gold/prompt_dev/validation/test(/monitor) manifests + `gold_to_code.csv` | —                                                                            |
| 02    | Bake-off prompts   | DONE   | coded `prompt_dev` + candidate models×prompts → `bakeoff_results.json`, `proposed_slate.json`       | **G1** prompt_dev coded                                                      |
| 03    | Annotate matrix    | DONE   | confirmed `production_slate.json` + silver (∪ gold, post-T2) → `annotation_store.csv`               | **G2** slate confirmed                                                       |
| 04    | QC & freeze        | DONE   | annotation store + coded `validation` → `silver_labels.csv` (silver-scoped, post-T3)                | **QC** chance-corrected agreement + minority-F1 CI floor (T9; was raw ≥0.85) |
| 05    | Train              | FUTURE | `silver_labels.csv` → fine-tuned encoder checkpoints                                                | —                                                                            |
| 06    | Evaluate           | FUTURE | model + frozen `test` → metric bundle, per-stratum fairness, calibration                            | acceptance criteria (TBD)                                                    |
| 07    | Inference at scale | FUTURE | model + rule layer over all records → `predictions_at_scale.csv` (label, prob, version)             | —                                                                            |
| 08    | Visualization      | FUTURE | predictions/embeddings → dashboards, word clouds, per-stratum summaries                             | —                                                                            |

**Gates:** **G1** — `gold_to_code.csv` must carry strict 0/1 for the required split before stages 02/04 (else
graceful exit). **G2** — a human-confirmed `production_slate.json` (`"confirmed": true`) before stage 03.
**QC** — LLM-vs-human agreement on `validation` must clear the freeze gate (post-T9: chance-corrected κ/α +
minority-F1 CI floor; raw ≥0.85 today) or stage 04 blocks and freezes nothing.

**Prevalence/calibration intent (cross-cutting).** A stated downstream goal
(`docs/research/20260605-replication-calibration-prevalence.md`) is estimating the **population share** of the
positive class over **all nonprofits** (locked estimand) — which is why the frame (Q≥3.0, excludes LOW) and
the design weight must be correct now (T4), and why a representative anchor covering LOW is on the roadmap.
2026 estimator stack: **PPI++** (primary, CI-valid) with **SLD/EMQ + KDEy/DyS via QuaPy** cross-check and
**per-NTEE-stratum calibration**.

## Roadmap refinements (2026-aligned, future stages — documented in T12, built later; NO code this sprint)

- **Prevalence / population share (estimand = ALL nonprofits).** Representative **anchor** sample at the true
  prior over the _full_ frame **including LOW (Q<3.0)**, rule-layer LOW label rate folded into the estimate
  (enriched silver/gold are training/eval only). Estimator **PPI++** (Angelopoulos 2023 → PPI++ arXiv:2311.01453,
  the `ppi_py` default) primary; **SLD/EMQ + KDEy/DyS via QuaPy** cross-check; **per-NTEE-stratum calibration**
  (prior-shift methods break under covariate shift). Open sub-questions: estimand granularity (national vs
  trend vs per-NTEE → PPI++ vs stratified-PPI vs MRP) and prevalence-sample size.
- **Stage 05 fine-tuning:** base model **DeBERTa-v3-base** (ModernBERT-base only if 560k-inference throughput
  is a hard constraint; may be a sweep arm). Soft-label / confidence-weighted loss using per-EIN2 silver
  agreement + label smoothing, each gated by "adopt only if it beats plain majority-vote labels on the human
  held-out." **bf16 not fp16** on the Blackwell B200 node. Learning-curve sweep for N. LoRA/PEFT + LLM
  distillation as optional comparison arms.
- **Stage 06 evaluation:** keep minority P/R/F1 + MCC + balanced-acc + PR-AUC + bootstrap-CI + per-stratum
  fairness; **add decision-curve (net-benefit) analysis** and report **ECE**. Conformal kept as
  selective-prediction/abstention add-on (not a calibration replacement).
- **Stages 07–08:** rule-layer + calibrated-classifier cascade; record model_version/checkpoint_hash;
  population-share correction runs on the stage-07 output using the anchor.
- **Gated annotation/aggregation arms:** uncertainty-weighted aggregation (use silver confidence in the vote)
  and NLI/calibrated-abstention evidence verification (vs exact-substring) — adopt each only if it beats the
  current default on the human held-out set.

## Decisions locked (drive the task cards)

- Validation overlap → annotate **silver ∪ gold** (T1–T2); scope the frozen artifact to silver-only (T3).
- Canary → dedicated **monitor** sub-slice from gold, excluded from the freeze gate (T1, T6).
- DVC → layout + gitignore + docs only; **no `dvc init`**; symlinks are the current stand-in. On-disk data
  move is the **user's** action (documented in T12).
- Aggregators → quarantine Dawid-Skene & CROWDLAB; majority vote stays default (T7).
- Q rubric → thread `cfg.q_thresholds`; keep the `quality.py:593` fallback; fix the docstring (T5).
- `guided_json` honesty + AGENTS.md/README reconcile + gitignore parquet guard kept; **drop** `load_missions`
  de-dup; **drop** Neyman/variance-aware allocation (deferred).
- QC gate → add minority-F1 **CI floor** + chance-corrected reporting; `kappa_threshold` set to today's
  operating point (T9).
- Sampling weight → fix `inclusion_prob` to per-cell (stratum × pos/neg) rate; document the Q≥3.0 frame (T4).
- Prevalence estimand = **all nonprofits**; encoder **DeBERTa-v3-base**; adopt the 4 upgrade families
  (quantification modernization, training/eval extras incl. **bf16**, drift-monitoring upgrade,
  gated aggregation/annotation arms) — roadmap only this sprint, except the drift-monitoring plumbing in T6.

---

## SPRINT — ordered task cards

> Shared-file serialization: `sample.py` → T1, T4, T5 (in that order). `run_annotation.py` → T2, T6.
> `agreement.py` → T3, T8, T9. Respect the order.

### T1 — Monitor split + manifest + incremental gold size [Workstream 0a; foundational]

**Depends on:** none (run first).
**Files:** `src/binary_classifier/data/sample.py`, `src/binary_classifier/paths.py`,
`src/binary_classifier/config.py`, `config/religious_missions.yaml`.
**Why:** create the held-out drift-monitor slice the canary (T6) needs, excluded from the freeze gate, sized
as _incremental_ gold so validation stays ≈120–175 (T9's CI floor is fragile on a 4-way split of ~400).
**Operations:**

1. Add `monitor: int` to `SampleSizesConfig` (config.py:157-168) and to `config/religious_missions.yaml`
   `sample_sizes:` (e.g. `monitor: 50`); **bump `gold`** by the monitor size (e.g. 400 → 450) so prompt_dev /
   validation / test keep their sizes. Document the rationale in the field docstring.
2. Extend `split_human_sets` (sample.py:~258-306) to carve a `monitor` slice (stratified, disjoint from
   prompt_dev / validation / test), parameterized by `monitor_size`; tag `split="monitor"`; return it.
3. Add `monitor_manifest` property to `PathRegistry` (paths.py, alongside the other manifests, under
   `interim_dir/manifests/`). Write it in `build_sample` (sample.py:388-390) via `_write_split_manifest`.
   Include monitor in the `gold_all` concat (sample.py:392) so it lands in `gold_to_code.csv`.
4. **Re-split caveat (put a comment + a one-line note in the code/docstring):** adding the monitor split
   reshuffles gold split tags; clobber-protection (sample.py:421) keeps an already-coded template, so manifests
   and template could disagree. The human must re-split **before** coding, or re-derive the template after.
   **Tests to add:** monitor slice is disjoint from prompt_dev/validation/test; `monitor_manifest` is written and
   carries an `EIN2` column; bumping `monitor` does not shrink validation/test sizes.
   **Acceptance:** tests pass; `pytest -q` green; report any split-size assertions in `tests/test_sample.py` you
   had to update.

### T2 — Annotate silver ∪ gold [Workstream 0b]

**Depends on:** T1.
**Files:** `src/binary_classifier/annotate/run_annotation.py`.
**Why:** the freeze gate and canary need the human-coded gold rows (validation + monitor) to actually carry
LLM labels; today only silver is annotated.
**Operations:**

1. In `run_annotation` (run_annotation.py ~117), build the EIN2 set from **both** `registry.silver_manifest`
   and `registry.gold_manifest`; join text from missions as today (run_annotation.py:125-127); annotate the
   union. Resume keying by (EIN2, source_id) is unchanged.
2. Comment why (gate + canary depend on validation/monitor being annotated).
   **Tests to add:** a fixture where validation EIN2s are NOT in silver pre-fix → after the union, the stage-04
   gate join (silver ⋈ validation) is non-empty (and was empty before).
   **Acceptance:** tests pass; `pytest -q` green.

### T3 — Scope the frozen artifact to silver only (leak guard) [Workstream 0c]

**Depends on:** T2.
**Files:** `src/binary_classifier/qc/agreement.py`.
**Why:** `run_quality_check` aggregates the **whole** store (agreement.py:99) and writes that full `aggregated`
to `silver_labels.csv` (agreement.py:178) — the stage-05 training pool. Post-T2 the store also holds gold
(validation/**test**/monitor/prompt_dev), so freezing the full frame would leak frozen test rows into training.
**Operations:**

1. Before the `to_csv` (agreement.py:177-178), **exclude all `gold_manifest` EIN2s
   (prompt_dev/validation/test/monitor) from `aggregated`** before writing. Frame it as _exclusion_, not
   "keep silver": silver and gold are independent draws and may incidentally overlap, and a `test` row must be
   dropped from the training pool **even if it is also in silver**. Comment this clearly.
2. Leave the agreement gate computation unchanged — it still computes on the validation overlap (now non-empty).
   **Tests to add:** **leak guard** — assert no gold/validation/test/monitor EIN2 appears in the written
   `silver_labels.csv`; the gate still computes on the validation overlap.
   **Acceptance:** tests pass; `pytest -q` green.

### T4 — Fix `inclusion_prob` granularity (design weight) [Workstream G-code]

**Depends on:** T1 (same file `sample.py`).
**Files:** `src/binary_classifier/data/sample.py`.
**Why:** today `inclusion_prob` stores one per-stratum marginal fraction per row (`n_target/len(stratum_df)`,
sample.py:160 silver / :249 gold), but positives and negatives are drawn at different rates within a stratum,
so it cannot reverse the enrichment and is unusable as a design weight (zero consumers today).
**Operations:**

1. Store the **per-cell** rate: `n_pos_target/len(pos_df)` for enriched-positive rows and
   `n_neg_target/len(neg_df)` for the rest (and per boundary-cell in `build_gold_set`). Manifest schema
   unchanged — only the values become a usable `1/π` for Horvitz–Thompson / PPI++. Cite the method in a comment
   (`# design weight = 1/π, per-cell inclusion probability; Horvitz & Thompson 1952`).
2. Comment that gold's boundary-quota weights are **diagnostic**, not for population estimation.
   **Tests to add:** in a fixture stratum with different pos/neg sampling rates, `inclusion_prob` differs between
   positive and negative rows and equals the per-cell rate (not a single per-stratum value).
   **Acceptance:** tests pass; `pytest -q` green.

### T5 — Thread `cfg.q_thresholds` + fix tier docstring [Workstream D]

**Depends on:** T4 (same file `sample.py`).
**Files:** `src/binary_classifier/data/sample.py`, `src/binary_classifier/data/quality.py`.
**Why:** `build_silver_pool` (sample.py:51, `QThresholdsConfig()` at :91) and `build_gold_set` (sample.py:169,
at :193) construct fresh defaults and silently ignore the YAML; only `build_sample` (:349) respects it.
**Operations:**

1. Add a `thresholds: QThresholdsConfig` param to `build_silver_pool` and `build_gold_set`; drop the fresh
   `QThresholdsConfig()` at :91 and :193; pass `cfg.q_thresholds` from `build_sample` (:349) and thread it to
   `assign_tier`. **Keep** the `assign_tier` default fallback at quality.py:593 (legit).
2. Fix the stale tier-band docstring in `compute_quality_score` (quality.py:431): config makes MEDIUM =
   `[3.0, 5.0)`, not "3.0–4.5"; clarify the `4.5` rescue band is a separate enrichment sub-rule.
   **Tests to add:** a non-default `q_thresholds` config changes tier boundaries at all threaded sites (shipped
   YAML equals defaults, so only a non-default test proves the wiring).
   **Acceptance:** tests pass; `pytest -q` green.

### T6 — Canary loader (monitor slice) + drift-monitoring upgrade [Workstream B]

**Depends on:** T1 (monitor_manifest) and T2 (same file `run_annotation.py`).
**Files:** `src/binary_classifier/annotate/run_annotation.py`, `scripts/03_annotate.py`.
**Why:** replace synthetic placeholder canary with the real held-out monitor slice; add 2026 drift monitoring.
**Operations:**

1. Replace the synthetic `CANARY_EIN2` constant (run_annotation.py:35) with
   `load_canary_ein2s(registry) -> set[str]` reading the `EIN2` column from `registry.monitor_manifest`, with
   an explicit `FileNotFoundError` guard ("run stage 01 first; monitor manifest lives under the cloud-symlinked
   interim tree"). Wire it through **both** `run_annotation` and `run_annotation_matrix` (both receive
   registry/cfg); the `canary_only` filter at run_annotation.py:195-197 lives inside `run_annotation_matrix`.
2. Hard-fail if `canary ∩ pool` is empty (no silent no-op). Update docstrings + `scripts/03_annotate.py
--canary` help: monitoring-only (drift + unbiased agreement), sourced from the held-out monitor slice,
   **excluded from the freeze gate**; prompts must not be tuned on it; test set untouched.
3. **Drift-monitoring upgrade (2026):** record the model fingerprint (pinned snapshot id, seed, temperature)
   with each canary run, hash/version the canary audit set, and add a **κ/α-over-time change test** vs the
   baseline canary run (the fingerprint/seed plumbing exists but is unused). Cite the monitoring rationale
   (model-snapshot/provider drift) in comments.
   **Tests to add:** fixture monitor manifest → `run_annotation(..., canary_only=True)` runs exactly those EIN2s;
   missing manifest → `FileNotFoundError`; empty `canary∩pool` → hard error.
   **Acceptance:** tests pass; `pytest -q` green.

### T7 — Quarantine dormant aggregators [Workstream C]

**Depends on:** none (independent file).
**Files:** `src/binary_classifier/annotate/aggregate.py`.
**Why:** majority vote stays the default; the alternative arms are not safe to run (D-S is unverified for a
correlated LLM ensemble; CROWDLAB needs a trained classifier).
**Operations:**

1. Replace the bodies of `aggregate_dawid_skene` (aggregate.py:87) and `aggregate_crowdlab` (aggregate.py:125)
   with a clear `raise NotImplementedError(...)` — D-S: "unverified for correlated LLM ensembles; majority vote
   is the default"; CROWDLAB: "requires pred_probs from a trained classifier (fine-tuning stage), not yet
   available." Keep both in the `aggregate_labels` dispatch so selection raises explicitly (not silent empty).
2. Update module + function docstrings with the _why_ and cite the rationale (correlated-ensemble limitation;
   CROWDLAB needs `pred_probs` — cleanlab multiannotator docs). Confirmed safe: only caller is agreement.py:99
   with `method="majority"`; no `__init__` re-exports.
   **Tests to add:** `aggregate_labels(method="dawid_skene"|"crowdlab")` raises `NotImplementedError`; `"majority"`
   unchanged.
   **Acceptance:** tests pass; `pytest -q` green.

### T8 — Abstain / evidence orchestrator parity [Workstream F]

**Depends on:** T3 (same file `agreement.py`).
**Files:** `src/binary_classifier/qc/agreement.py`, `scripts/04_quality_check.py`.
**Why:** the evidence-verification + abstain step (`qc/evidence.py:abstain_fabricated_positives`) runs only in
the standalone `scripts/04_quality_check.py:88-94`, NOT in the package `run_quality_check` that
`run_pipeline.py` stage 04 calls — so the orchestrator silently skips it.
**Operations:**

1. Lift the abstain/evidence step into `run_quality_check`, gated by `cfg.qc.abstain_on_fabricated_positive`
   (default False ⇒ no behavior change with the shipped config). Apply it to the annotation-store df **before**
   `aggregate_labels` (agreement.py:99). Comment the hallucination-guard rationale.
2. Preferred: make `scripts/04` a thin wrapper over `run_quality_check` to prevent future drift.
   **Tests to add:** with `abstain_on_fabricated_positive: true`, `run_quality_check` abstains a fabricated
   positive before aggregation (parity with scripts/04).
   **Acceptance:** tests pass; `pytest -q` green.

### T9 — QC gate: minority-F1 CI floor + chance-corrected reporting [Workstream H]

**Depends on:** T8 (same file `agreement.py`).
**Files:** `src/binary_classifier/qc/agreement.py`, `src/binary_classifier/config.py`,
`config/religious_missions.yaml`.
**Why & rationale correction (don't oversell):** the gate computes agreement on `valid` = silver ⋈
**validation**, and validation ⊂ gold, which is built with boundary/balanced quotas (~25% clear-pos / 25%
clear-neg / ~35% boundary) — so the gate set is roughly **balanced, NOT at the rare ~13% prior**. Raw agreement
here is _not_ imbalance-inflated, and on a balanced binary set chance ≈ 0.5 ⇒ κ = 2·acc − 1, i.e. κ ≥ 0.70 ⟺
acc ≥ 0.85. So chance-correction roughly _reproduces_ today's operating point; **the operative new lever is the
CI floor.**
**Operations:**

1. Change the gate (agreement.py:162) to ALSO require **the bootstrap-CI lower bound on minority-class F1 ≥
   `f1_ci_floor`** (bootstrap CIs for accuracy and minority-F1 already computed) and to **report κ /
   Krippendorff α** (κ already in `_compute_metrics`). Keep raw agreement logged as reported-only.
2. Add `kappa_threshold` (set to **match today's operating point**, ≈0.70 ≡ acc 0.85 on the balanced gate set,
   so it is NOT a stealth tightening) and `f1_ci_floor` (the deliberate new constraint) to `QCConfig`
   (config.py) + the YAML, config-driven with sensible defaults. Update the failure message to cite κ/α,
   minority-F1, and its CI lower bound. Cite Landis & Koch 1977 (κ bands) and the SILICON / variance-aware
   gating rationale in comments. (Krippendorff α handles abstain/missing cells cleanly if added.)
   **Tests to add:** a fixture that clears the point estimate but whose minority-F1 CI lower bound is below
   `f1_ci_floor` is **blocked**; a fixture clearing both κ and the CI floor **freezes**; thresholds read from
   config; raw agreement still logged.
   **Acceptance:** tests pass; `pytest -q` green.

### T10 — Defect cleanup: dead code, `guided_json`, test rename [Workstream E]

**Depends on:** none (independent files).
**Files:** `src/binary_classifier/annotate/annotators/base.py`, the two annotators
(`vllm_annotator.py`, `openai_annotator.py`), `tests/test_foundation.py` (rename), `tests/test_schema.py` (add
assertion).
**Operations:**

1. Remove dead `_read_prompt_file` (annotators/base.py:80) **and** the now-orphaned `from pathlib import Path`
   (base.py:10).
2. `guided_json` honest: wire the flag (`cfg.annotation.guided_json`) into both annotators so `false` disables
   guided/structured decoding. Keep `tests/test_schema.py:192` (true-branch) passing; add a false-branch
   assertion. Comment the flag's effect.
3. Rename `tests/test_foundation.py:77` `test_ensure_dirs_creates_gold_and_silver` → `..._gold_and_interim`
   (body already asserts `interim_dir`).
   **Tests to add:** `guided_json: false` disables guided decoding (true-branch test still passes).
   **Acceptance:** tests pass; `pytest -q` green.

### T11 — `.gitignore` gold-tracking fix [Workstream A1]

**Depends on:** none.
**Files:** `.gitignore`.
**Why:** verified `git ls-files data/` = 0 — gold (and `production_slate.json`) are NOT tracked; line 28 `data`
excludes the tree and the `!data/processed/gold/` negation on line 31 is void (git can't re-include a child of
an excluded parent).
**Operations:** replace the broken `data` + void negation with:

```
# Data: heavy/cloud artifacts ignored; symlinks stand in for DVC until we adopt it.
/data/*
!/data/processed/
/data/processed/*
!/data/processed/gold/
# Note: line-4 *.parquet is global — any future .parquet under gold/ is still ignored.
```

**Acceptance / verification (note ordering precondition):** the verification passes only **after** the user
has placed gold at `data/processed/gold/` (today that dir doesn't exist; legacy `train_test_datasets/gold/` is
empty). Run `git check-ignore -v data/processed/gold/gold_to_code.csv` (expect **no match**) and
`git add -n data/processed/gold/` (expect files listed); confirm `data/raw/*.parquet`, `data/interim/`,
`silver_labels.csv` stay ignored. If gold isn't present yet, report this precondition rather than failing.

### T12 — Docs sync: layout, frame, DVC note, roadmap [Workstreams A2/A3 + G-docs + E-docs + roadmap]

**Depends on:** run **last** (docs describe the final state of all prior tasks).
**Files:** `README.md`, `AGENTS.md`, `docs/agents/configuration.md`,
`docs/agents/conventions/python-standards.md`. (Leave dated `docs/research/*` and `.agents/plans/*` as
historical records.)
**Operations:**

1. Describe the cookiecutter `raw/interim/processed/models` layout; state plainly that **cloud symlinks
   currently substitute for DVC** for everything not committed (raw, interim, silver pool, models) and that
   **gold + `production_slate.json` are the committed pointers**. Add a short "future DVC migration" note (when
   adopting DVC, `dvc add` the `data/raw/*.parquet` + configure a `dvc remote`; avoid intermediate directory
   symlinks because DVC manages its own cache links).
2. **Setup note:** user creates real `data/{raw,interim,models}` dirs and moves gold to `data/processed/gold/`;
   `ensure_dirs()` does **not** create `raw_dir` (paths.py:162-172) and `load_missions` hard-fails on a missing
   raw parquet unless `allow_synthetic:true` (load.py:77-86) — surface a friendly hint. Tell the user to
   **re-point the cloud silver symlink** from `data/processed/train_test_datasets/silver` to the new location
   (e.g. under `data/interim/`), since moving gold alone leaves the heavy silver pool on the old path.
3. **Frame documentation (from T4):** the sampling frame is HIGH+MEDIUM (Q≥3.0), **not** the population; LOW is
   excluded and handled by the rule layer; any population claim must fold in the LOW tier (rule-layer rate ×
   LOW count). This is the hook for the all-nonprofits prevalence estimand (roadmap).
4. **Reviewer corrections (must apply):** there is **no `results/`** string in `pipeline.md` — do NOT edit it.
   `configuration.md:13-14` documents **both `gold_dir` and `silver_dir`** as `paths` YAML keys; neither exists
   (config.py:40-43 has only raw/interim/processed/models) and `gold_dir` is a derived registry property — fix
   **both**. `AGENTS.md:43` ("`data/` is a symlink … all gitignored") is false on two counts after T11 (`data/`
   is a real dir; only `…/silver` is the symlink; gold IS committed) — rewrite it. `README` line 83 (stale
   `train_test_datasets/...`) contradicts line 126 (new path) — reconcile the whole file.
    `docs/agents/conventions/python-standards.md:37,40` example path
   `data/processed/train_test_datasets/manifests/silver_manifest.csv` → `data/interim/manifests/silver_manifest.csv`.
5. **Roadmap section:** record the 2026-aligned future stages (prevalence anchor + PPI++/QuaPy + per-stratum
   calibration; DeBERTa-v3-base; soft-label/label-smoothing + bf16; decision-curve + ECE; gated aggregation
   arms) in the architecture docs, citing the sources named in the "Roadmap refinements" section above.
   **Acceptance:** docs build/read consistently; no remaining `train_test_datasets`/`silver_dir`/`gold_dir`-as-key
   references in the living docs (grep to confirm); report the grep result.

---

## References (locked — use these exact identifiers in code comments / docs per the citation convention)

**Implementation-stage (cite inline in code comments):**

- Horvitz–Thompson design weight (1/π): Horvitz & Thompson 1952, _JASA_ 47(260):663–685. — **T4**
- Rare-events / case-control enrichment + prior correction: King & Zeng 2001, _Political Analysis_ 9:137–163
  (SSRN abstract_id=1083726). — **T4** rationale
- Calibration after undersampling/enrichment: Dal Pozzolo et al. 2015, IEEE SSCI. — **T4** rationale
- Majority/EM label aggregation: Dawid & Skene 1979, _JRSS-C_ 28(1):20–28. — **T7**
- CROWDLAB (requires classifier `pred_probs`): Goh, Mueller et al. 2022; cleanlab `multiannotator` docs. — **T7**
- Chance-corrected agreement bands: Landis & Koch 1977, _Biometrics_ 33(1):159–174 (κ); Krippendorff's α. — **T9**
- LLM-annotation gating / variance: SILICON (arXiv:2412.14461); Variance-Aware protocol (arXiv:2601.02370);
  "LLM Hacking" model×prompt variance (arXiv:2509.08825). — **T9, T6**
- Human-validation-first for LLM annotation: Pangakis & Wolken 2025 (ICWSM); Halterman & Keith 2025
  "Codebook LLMs" (_Political Analysis_). — context

**Roadmap-stage (cite in the T12 docs):**

- PPI → PPI++ (the `ppi_py` default): Angelopoulos et al. 2023; PPI++ arXiv:2311.01453; stratified PPI
  (NeurIPS 2024); active inference Zrnic & Candès 2024.
- Quantification: Saerens, Latinne & Decaestecker 2002 (EM/SLD prior adjustment); CC/ACC/PACC, DyS/HDy, and
  KDEy (_Machine Learning_ 2024) via **QuaPy**; King & Hopkins ReadMe / readme2.
- Small-area / weighting: MRP (poststratification superset; _Statistics in Medicine_ 2024).
- Encoder: DeBERTa-v3 vs ModernBERT controlled study arXiv:2504.08716; **bf16 over fp16** on Blackwell B200 (NVIDIA).
- Eval: decision-curve / net-benefit analysis (Vickers & Elkin 2006); ECE calibration.
- Data engineering: DVC docs (dvc.org — `add`, cache link types, remotes, external data); cookiecutter-data-science
  (drivendata) for the raw/interim/processed layout.

_Caveat:_ a few benchmark **rankings** in the source audits were flagged as not independently verified (e.g.
LeQua year/leaderboard specifics) — cite the **methods** above, not contested leaderboard claims.

---

## Out of scope / deferred (intentional)

- **No model/prevalence code this sprint** — stages 05–08 and the prevalence stack are documented roadmap only
  (T12). Also deferred: `dvc init`/remotes; surfacing rubric magic numbers to config; fixing Dawid-Skene into a
  real arm; `load_missions` de-dup; Neyman/variance-aware allocation. The physical on-disk data move stays the
  user's action.

## Final verification (orchestrator runs after T12)

1. **Full suite:** `pytest -q` green (baseline 69 + all new T1–T10 tests).
2. **Gitignore (after user places gold):** `git check-ignore -v data/processed/gold/gold_to_code.csv` → no
   match; `git add -n data/processed/gold/` lists files; `data/raw/*.parquet`, `data/interim/`,
   `silver_labels.csv` still ignored.
3. **W0 gate + leak guard:** on a realistic fixture (validation drawn disjoint from silver), the gate join is
   non-empty after the union annotation and was empty before; the frozen `silver_labels.csv` contains **only**
   silver-manifest EIN2s.
4. **Canary:** monitor-slice EIN2s drive `--canary`; gate metrics computed on validation only (monitor excluded).
5. **Q thresholds:** non-default config shifts tiers at every threaded site.
6. **Design weight:** per-cell `inclusion_prob` differs for pos vs neg rows within a stratum.
7. **QC gate:** a point-pass / CI-floor-fail fixture blocks; a both-pass fixture freezes; thresholds from config.
8. **Smoke:** `python scripts/run_pipeline.py` stages 01–04 on synthetic data (`allow_synthetic:true`) runs
   end-to-end, exercises the abstain step (flag true in a smoke config), and writes `silver_labels.csv` only
   when the (chance-corrected + CI-floor) gate passes.
9. **Docs:** grep confirms no stale `train_test_datasets` / `silver_dir`-as-key references remain in living docs.

```

```
