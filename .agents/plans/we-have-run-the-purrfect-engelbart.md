---
created: 2026-07-02
---

# Sprint Plan — Harmonize & Strengthen the Religious-Mission Pipeline

> **For the orchestrator agent.** This file is the shared brief. Sub-agents start with **no prior context** except the codebase, the docs, and this file. Read §0–§2 (Context, Invariants, Reference Map) first; then execute your assigned **Wave** or **Task Card** (§3) exactly. Each card lists its files/functions, ordered operations, reuse, acceptance criteria, and a **Report-back contract**. Commit once per wave (§4). Waves are dependency-ordered — do not start a wave until its predecessor is committed and verified. Task cards **within** a wave may be parallelized unless a card says "depends on" a sibling.
>
> **Sequencing law:** The **entire sprint (Waves 1–6) is LOCAL** — code + verification only. Verify on `config/smoke.yaml`, on unit tests, and (where noted) on a local CPU dry-run against the **existing first-run artifacts**. **No sub-agent runs UCloud, re-scores the model, or re-opens the real frozen test.** The single controlled UCloud re-evaluation and the production of the final canonical artifacts happen **after the sprint is done** — see **§7 Post-Sprint Handoff (UCloud)** — and are executed by the user, not by sprint sub-agents.

---

## §0. Context & Deliverables

First full UCloud run of the pipeline completed; audited in `docs/audits/religious_evaluation_report.md`. An independent review (`.agents/plans/we-have-run-the-purrfect-engelbart-agent-a87b32148d5a6939e.md`) confirmed the load-bearing code claims and found the LOW-tier prevalence bug (§Decision 4). The model passes its gates (F1 0.894, recall 0.987, PR-AUC 0.9014, ECE 0.007). This sprint fixes inconsistencies and adds SOTA methodology + figures + documentation, all implemented and verified **locally**; a controlled frozen-test re-evaluation and the final production run follow on UCloud after the sprint.

**Deliverables (code + local verification in the sprint; final artifacts post-sprint):** (a) a released **per-organization** labeled dataset (`predictions_full.parquet`); (b) an aggregate **prevalence** estimate with figures; (c) refreshed **documentation** for any audience.

**Confirmed decisions (do not relitigate):**
1. **Estimand = per-organization**, all raw EIN2 (N=560,354). `data/load.py:117` dedups on `cfg.field` (=`LONGEST_MISSION`) to 531,660 rows before inference. Duplicate text = same mission = same score → expand predictions back to every raw EIN2.
2. **Deliverable** includes a released per-org labeled dataset + prevalence + docs.
3. **Triple label:** keep `pred_label` (recall-first, threshold 0.0577) as the **prevalence** label; for the released dataset ship **both** `pred_label_maxf1` (0.608) **and** `pred_label_baserate` (Wave 1), plus `prob_calibrated`.
4. **LOW-tier prevalence is misspecified — fix it.** `estimate.py:_estimate_low` (line 549) averages `pred_label` over *all* LOW rows, sweeping the 59,704 `low_via_classifier` classifier decisions (`router.py:53`) into a Rogan-Gladen rule-correction driven by rule-only sens/spec from 67 anchor rows. LOW is 20.7% of the corpus; this can move the 14.45% headline. Split LOW: classifier-decided → PPI; pure-rule (~50.5k) → Rogan-Gladen.
5. **Do not enlarge the gold/test set** (deterministic split; would reshuffle & break the one-shot frozen test). Accept the model; document limits; report label precision at the ~14% population base rate (test is ~50% positive enriched → its 0.817 precision is optimistic).
6. **Controlled frozen-test re-evaluation on UCloud, post-sprint.** `test_evaluation.json` stores only aggregate metrics (scalar PR-AUC/ROC-AUC, one confusion matrix at 0.0577) — no per-row test scores — so real frozen-test PR/ROC curves and release-threshold confusion matrices require re-opening stage 07. This is done **once**, code-frozen, on UCloud (environment parity), with a reproduce-assertion, persisting per-row test scores so it is one-and-done. **Not part of the sprint — see §7.**
7. **Scope:** all 8 figures (completeness); simplify archiving (manifest + git tag + env lock) and stage-10 wiring (just register `"10"`); full documentation pass.

## §1. Governing Invariants (every sub-agent must respect)

- **The model checkpoint (SHA-verified) and the operating threshold (selected on anchor OOF, deterministic) are FIXED.** No retraining. No threshold re-selection informed by test results.
- **No sprint sub-agent re-scores the frozen test or runs the model.** The frozen test is re-scored exactly once, **post-sprint, on UCloud (§7)**, then re-sealed. Because model + threshold cannot change, that re-run is a *reproducibility check that also enriches the report* — never a re-selection.
- Every sprint artifact is a re-derivation from **already-saved probabilities**: `prob_calibrated` in `predictions*.parquet`, `prob_calibrated_oof` + `human_label` in `anchor_oof_scores.parquet` (both exist on disk from the first run).
- **Preserve** the deduped `predictions.parquet` (consumed by stage 09 + calibrator) unchanged; add `predictions_full.parquet` alongside it.
- **Determinism:** reuse `cfg.SEED` (42) and existing seeded RNG patterns; never introduce unseeded randomness.

**Rerun boundary during the sprint (all local):** 01–06 frozen (model kept). 07 = **not re-run in the sprint**; its code changes (Wave 1) are smoke-verified only. 08/09/10 = local CPU derivations from existing frozen probabilities (used both for smoke verification and to report the real corrected numbers). The canonical artifacts are regenerated on UCloud in §7.

## §2. Reference Map (key code — verified locations)

| Area | Path | Anchor points |
|---|---|---|
| Orchestrator | `scripts/run_pipeline.py` | `_STAGE_MODULES` (~35); stages 10/11 NOT registered; gates via `qc/preflight.py:validate_gates` |
| Stage 07 evaluate | `src/binary_classifier/evaluation/evaluate.py` | `run_evaluation` (81–223): writes anchor_oof (141), calibrator (153), rule_validation (160); **guard 166–171 fires AFTER those writes**; `_calibrator_payload` (410–428) **drops** `threshold_report["pr_curve_points"]`; `_test_report` (537–582); `_read_frozen_test_labels` (500–530) |
| Threshold | `src/binary_classifier/evaluation/thresholds.py` | `pick_threshold` (46–106) returns `threshold`, `max_f1_threshold`, `pr_curve_points` (line 104); `_pr_curve_points` (130–152) |
| Calibration | `src/binary_classifier/evaluation/calibration.py` | `crossfit_calibrate` (191–305) → `deployed`; metrics `expected_calibration_error`, `reliability_curve` |
| Prevalence | `src/binary_classifier/prevalence/estimate.py` | `run_prevalence` (240–348); `_estimate_hm` (416–476); `_estimate_low` (519–595, `p_obs` line 549); tier shares (264–267); per-NTEE `_ntee_emq_fallback` (821–870); `_report` (882–961). PPI in `ppi.py` (weights labeled anchor only; corpus enters as a **mean**); `composite.py` (Rogan-Gladen); `weights.py:design_weights`; `quantify.py` (EMQ) |
| Routing/tiers | `src/binary_classifier/inference/router.py` | `route` (18–54): `low_via_classifier` (53), `rule_strong_positive`/`rule_short_negative` (49/51). Tiers in `data/quality.py:assign_tier` |
| Stage 08 inference | `src/binary_classifier/inference/predict.py` | `run_inference`; `_PREDICTION_COLUMNS` (~35); `_predict_shard` (~439); `_existing_shard_matches` (~397); `_RESUME_METADATA_COLUMNS` (~62); `_prediction_metadata` (~640); `_validate_ein2_completeness` (~620); `_load_calibrator` (~252); `_write_monitor_scores` (~566) |
| Data load | `src/binary_classifier/data/load.py` | `load_missions` (46–134); dedup on `cfg.field` at **line 117**; rename→`mission_text` (120) |
| Sampling/split | `src/binary_classifier/data/sample.py` | `build_gold_set` (180–289); `split_human_sets` (291–339) assigns `split=="test"` |
| Stage 10 viz | `scripts/10_visualize.py` | `_maybe_render_production_summary` (132–166, **sizing bug 162–166**); `_maybe_render_pr_curve` (244); `_PR_POINT_KEYS` (46–52); `_save_plot` (419) |
| Viz helpers | `src/binary_classifier/viz/` | `style.py` (`figure_size`, `standardize_figsize`, `PAGE_WIDTH=7.0`, `COLUMN_WIDTH=3.5`, `style_context`); `curves.py`, `bakeoff.py`, `prevalence_plots.py`, `ngrams.py` |
| Paths | `src/binary_classifier/paths.py` | `PathRegistry` property accessors for every artifact |
| Config | `config/religious_missions.yaml` + `src/binary_classifier/config.py` | `threshold_policy: precision_floor`, `precision_floor: 0.80`; `prevalence` block; `config_hash` stamped into outputs |
| Smoke harness | `config/smoke.yaml`, `tests/test_e2e_stages_05_11.py` | gold 120 / test 45 / anchor 60, tiny model; monkeypatch pattern |
| Current artifacts | `data/processed/{evaluation,predictions,prevalence,figures}/` | frozen first-run inputs for local dry-runs / verification |

Existing test files to extend live under `tests/` (e.g. `test_evaluate_stage.py`, `test_predict_stage.py`, `test_prevalence_stage.py`, `test_thresholds.py`, `test_viz.py`).

---

## §3. Sprint Waves & Task Cards (all local)

### Wave 1 — Stage-07 code hardening & evaluation enrichment (code; smoke-verified)
**Goal:** Land every stage-07-side change and freeze it, so the §7 controlled re-run produces enriched, correct artifacts. **Nothing here re-opens the real frozen test; verify on `config/smoke.yaml` only.**

**T1.1 — Reorder the frozen-test guard.**
- File/fn: `evaluation/evaluate.py:run_evaluation` (140–171).
- Ops: Move the G3 gate (`_raise_gate_problems("G3", _validate_test_unlock(...))`) and the `if registry.test_evaluation.exists(): raise RuntimeError(...)` block to the top, immediately after `_load_and_verify_selected_model` and the G4 check, before any artifact write. Confirm no write occurs before the guard.
- Acceptance: with an existing `test_evaluation.json`, `run_evaluation` raises **before** writing; the three calibration artifacts are byte-unchanged. Add/extend a test.
- Report-back: files changed, test name, confirmation the artifacts are untouched on a blocked run.

**T1.2 — Enforce label-set disjointness.**
- Files/fn: `data/sample.py` (build path) and/or `qc/preflight.py`.
- Ops: add an anti-join check that `silver ∩ (gold ∪ anchor) = ∅` on `EIN2`; raise a clear error listing offending EINs if not.
- Acceptance: unit test where an injected overlap raises; clean sets pass. Run the check on **current** on-disk sets and report the result/count.
- Report-back: implementation location, test name, disjointness result on current data.

**T1.3 — Persist PR-curve points in the calibrator payload (+ regen utility).**
- Files/fn: `evaluation/evaluate.py:_calibrator_payload` (410–428); new `scripts/regen_calibrator.py`.
- Ops: (1) Add `"pr_curve_points": threshold_report["pr_curve_points"]`, `"achieved_precision"`, `"achieved_recall"`. (2) `scripts/regen_calibrator.py` reads `anchor_oof_scores.parquet`, re-derives `pick_threshold(prob_calibrated_oof, human_label, policy, precision_floor)` to recover `pr_curve_points`/`max_f1_threshold`/achieved P-R, rewrites **only** `calibrator.json` (never touches `test_evaluation.json`). Reuse `thresholds.pick_threshold`.
- Acceptance: regen reproduces `pr_curve_points` + `max_f1_threshold` without opening the test; unit test. May be run locally against the **existing** `anchor_oof_scores.parquet` to add PR points to the real `calibrator.json` (frozen-safe).
- Report-back: confirmation regen touches only `calibrator.json`.

**T1.4 — Persist per-row frozen-test scores, multi-threshold confusion matrices, PR/ROC points.**
- Files/fn: `evaluation/evaluate.py:_test_report` (537–582) and the test-scoring block (178–206).
- Ops: (1) Add a `test_scores` block to `test_evaluation.json` with per-row frozen-test `y_true` + `prob_calibrated` (EIN2-keyed). (2) Store confusion matrices at **0.0577, 0.608, base-rate** (from T1.5). (3) Store frozen-test PR + ROC curve points. Reuse `metrics.py`; keep the headline `metric_bundle` (at 0.0577) unchanged.
- Acceptance: schema round-trips; headline metrics identical at 0.0577; smoke run produces the new blocks. Unit test. (These real blocks are only populated by the §7 UCloud re-run; the sprint verifies the code on smoke.)
- Report-back: new schema keys; confirmation headline metrics unchanged. **Depends on T1.5.**

**T1.5 — Base-rate precision module + base-rate-targeted threshold.**
- Files/fn: new `evaluation/base_rate.py`; config knobs (`evaluation.base_rate_precision_target` default 0.90; `evaluation.population_base_rate` or derive from prevalence composite).
- Ops: (1) `precision_at_base_rate(tpr, fpr, pi) = tpr*pi/(tpr*pi+fpr*(1-pi))`. (2) TPR/FPR from **anchor classifier-routed** rows in `anchor_oof_scores.parquet` at each threshold (0.0577, 0.608), **unweighted and `design_weights`-weighted** (reuse `weights.py:design_weights`). (3) **Seeded bootstrap CI** (reuse `evaluate.py` pattern). (4) `pred_label_baserate` threshold = smallest threshold with base-rate precision ≥ target; else best achievable + `unattainable` flag. (5) Emit `evaluation/base_rate_precision.json`; add a `paths.py` property.
- Acceptance: `precision(π)` matches a hand-computed case; weighted+unweighted+CI present; target met/flagged. `tests/test_base_rate.py`. Runs locally against the **existing** `anchor_oof_scores.parquet` — report the real derived threshold.
- Report-back: derived base-rate threshold + whether the 0.90 target is attainable on current anchor data.

**Wave 1 verification:** `pytest` new/changed tests; run stage 07 on `config/smoke.yaml`, confirm enriched `test_evaluation.json` blocks + `base_rate_precision.json` appear and smoke acceptance still passes. Optionally run T1.3/T1.5 locally against existing artifacts to produce the real `calibrator.json` PR points + base-rate threshold. **Do not re-score the real frozen test.**
**Wave 1 commit:** `feat(evaluation): harden stage-07 guard and enrich frozen-test reporting`.

---

### Wave 2 — Inference: triple labels, expand-back, shard hygiene (code + real CPU backfill)
**Goal:** Produce the released per-org dataset with three labels; keep the deduped artifact intact. **Backfill runs on CPU off existing `predictions.parquet` — no model, no UCloud.**

**T2.1 — Triple-label schema in stage 08.**
- Files/fn: `inference/predict.py` — `_PREDICTION_COLUMNS` (~35), `_predict_shard` (~439), `_load_calibrator` (~252), `_prediction_metadata` (~640), `_RESUME_METADATA_COLUMNS` (~62), `_existing_shard_matches` (~397), `_write_monitor_scores` (~566).
- Ops: (1) Add `pred_label_maxf1`, `pred_label_baserate`, `threshold_maxf1`, `threshold_baserate`; keep `pred_label` (0.0577) + `prob_calibrated`. (2) `_load_calibrator` exposes `max_f1_threshold` (present) + base-rate threshold (from `base_rate_precision.json`). (3) `_predict_shard`: release label = `(calibrated >= threshold)` for classifier rows, `= pred_label` for rule/abstain rows (document). (4) Add both thresholds to metadata + `_RESUME_METADATA_COLUMNS` + `_existing_shard_matches` (`np.isclose` branches) so a changed threshold forces re-write. (5) Include release labels in monitor scores.
- Acceptance: unit tests in `tests/test_predict_stage.py` (label == threshold rule for classifier rows, == `pred_label` for rule rows; shard-match rejects on changed release threshold).
- Report-back: final schema + the two threshold values.

**T2.2 — Stale-shard cleanup.**
- Files/fn: `inference/predict.py:run_inference` before `_process_shards`.
- Ops: delete `predictions/shards/shard_*.parquet` not in the about-to-be-written set; keep resume via `_existing_shard_matches`; guard so a `--infer-limit` run cannot delete production shards.
- Acceptance: unit test — stale removed, matching resumed, limited run safe. **Depends on T2.1.**

**T2.3 — Expand-back to raw EIN2 → `predictions_full.parquet`.**
- Files/fn: `data/load.py:load_missions` (add `deduplicate: bool = True`); new `inference/predict.py:_write_predictions_full`; new `paths.py:predictions_full_parquet`.
- Ops: (1) `load_missions(deduplicate=False)` → all 560,354 raw rows (default True preserves current callers). (2) Join key uses **`cfg.field`** (not hardcoded); normalize both sides with a NaN sentinel that is **not** `""`. (3) Map key→prediction from the deduped merged frame; left-join onto the full raw frame with `validate="many_to_one"`; carry all prediction columns onto every raw EIN2, keyed on raw EIN2. (4) Completeness assertion mirroring `_validate_ein2_completeness`: 560,354 raw EIN2 present exactly once, zero null `pred_label`.
- Acceptance: `tests/test_expand_back.py` (sentinel≠""; NaN and empty-text rows labeled; `validate="many_to_one"` collision raises; completeness assertion fires on a dropped EIN). Deduped `predictions.parquet` unchanged.
- Report-back: raw row reconciliation (560,354), null-label count (0), collision result. **Depends on T2.1.**

**Wave 2 verification:** unit tests green; smoke stage 08 → `predictions_full.parquet` one row per raw synthetic EIN2, zero nulls, three labels. Local CPU dry-run on **real** existing `predictions.parquet` (re-threshold + expand-back) to confirm it produces a well-formed real `predictions_full.parquet`; assert `test_evaluation.json` + `anchor_oof_scores.parquet` byte-unchanged. (Canonical artifact regenerated on UCloud in §7.)
**Wave 2 commit:** `feat(inference): triple labels, per-organization expand-back, shard hygiene`.

---

### Wave 3 — Prevalence: LOW decomposition & per-organization weighting (code + real local run)
**Goal:** Correct the LOW estimator and make the estimand per-organization. Runs locally on existing/backfilled artifacts (frozen probs) — reports the **real** corrected headline without UCloud.

**T3.1 — LOW decomposition (classifier→PPI, rule→Rogan-Gladen).**
- Files/fn: `prevalence/estimate.py:_estimate_low` (519–595); `run_prevalence` (240–348); tier-share accounting (264–267).
- Ops: split tier==LOW by `decision_source`: `low_via_classifier` (~59.7k, carry `prob_calibrated`) → **PPI** (HM machinery/`ppi.py`); `rule_*` (~50.5k) → **Rogan-Gladen** with rule sens/spec. Recombine (share-weighted) into composite. Keep `pred_label` (0.0577) as the prevalence label.
- Acceptance: unit test — classifier-LOW→PPI, rule-LOW→RG; shares sum correctly; LOW estimate changes vs old all-RG (report both).
- Report-back: old vs new LOW estimate + new composite headline.

**T3.2 — Per-organization multiplicity weighting.**
- Files/fn: `prevalence/estimate.py:_estimate_hm` (+ LOW-classifier sub-stratum), tier shares (264–267).
- Ops: expand each **unlabeled** classifier prob by its raw-EIN multiplicity (count of raw EIN2 sharing that `cfg.field`, from the Wave 2 raw frame) before the corpus mean (no `ppi.py` change). Do **not** expand the labeled anchor residual. Tier shares + sub-stratum denominators over **raw EIN2 counts**. Record per-org estimand in `_report`'s `estimand_statements`. Add a one-line sensitivity (anchor residual multiplicity-weighted vs not).
- Acceptance: unit test — multiplicity-weighted corpus mean == explicitly-expanded-row mean; tier shares over raw counts.
- Report-back: per-org vs per-unique-text composite + sensitivity delta. **Depends on T3.1.**

**T3.3 — Per-NTEE fallback relabel.**
- Files/fn: `prevalence/estimate.py:_ntee_emq_fallback` (821–870) + CSV writer.
- Ops: mark suppressed low-support NTEE groups as `not estimated` rather than emitting unstable ~77–83% points. Acceptance: unit test.

**Wave 3 verification:** unit tests green; run stage 09 locally on real existing data → report has LOW sub-strata, per-org tier shares summing to 1 over raw counts, sensitivity line. **Record the real corrected headline shift vs the audit's 14.45%.** (Canonical numbers reconfirmed on UCloud in §7.)
**Wave 3 commit:** `fix(prevalence): correct LOW-tier estimator and adopt per-organization estimand`.

---

### Wave 4 — Paper figures (code; smoke-verified; real non-test figures rendered locally)
**Goal:** Fix broken figures and add the CSS-grade set. All renderers route through `_save_plot` + `standardize_figsize` + `style_context`; register each new renderer in the `run_visualization` tuple. **Figures that consume frozen-test per-row scores (T4.2/T4.4) are validated against smoke-generated `test_scores`; their real versions are produced post-sprint (§7) after the UCloud re-eval. All other figures render on real local data.**

- **T4.1 — Fix `production_annotation_summary` sizing** (`scripts/10_visualize.py` 132–166): aggregate `_production_summary_frame` to per-category rows before sizing. Do **not** touch `_maybe_render_prevalence_forest`. Acceptance: PNG < ~1 MB.
- **T4.2 — PR curve** (`_maybe_render_pr_curve` 244; `viz/curves.py`): render **frozen-test** PR (+ ROC) from `test_evaluation.json` `test_scores`; annotate operating points. (Real version pending §7.)
- **T4.3 — Score distribution by tier/label** (new `viz/curves.py` helper): calibrated-score histogram/KDE faceted by tier from `predictions.parquet`; mark all three thresholds; shade the disagreement band.
- **T4.4 — Confusion matrices** (new helper): frozen-test CMs at 0.0577 / 0.608 / base-rate from the persisted multi-threshold CMs. (Real version pending §7.)
- **T4.5 — Prevalence decomposition** (new `viz/prevalence_plots.py` helper): waterfall of HM-PPI / LOW-PPI / LOW-RG contributions with CIs.
- **T4.6 — Rule-validation intervals** (new helper): Wilson-CI forest from `rule_validation.json`.
- **T4.7 — Quantification sensitivity** (new helper): PPI vs EMQ vs weighted/unweighted point-range.
- **T4.8 — Subgroup performance** (new helper): per-NTEE / data_source / length-bin dot plot from `subgroups` (present in the existing `test_evaluation.json`).

**Wave 4 verification:** run stage 10 on smoke; all 8 renderers succeed (no skips), the previously-skipped `precision_recall_curve` + the fixed production summary now render, every PNG < ~1 MB. Render the non-test figures (T4.1, T4.3, T4.5–T4.8) on real existing data as a dry-run. Extend `tests/test_viz.py`.
**Wave 4 commit:** `feat(viz): fix figure sizing and add CSS-grade evaluation & prevalence figures`.

---

### Wave 5 — Reproducibility & orchestration (code)
- **T5.1 — Run manifest + git tag + env lock:** emit one `run_manifest.json` (config_hash, git sha, thresholds, input row counts, Wave-2 completeness result, and a placeholder field for the §7 reproduce-assertion) + a git tag; include an environment lock (`uv.lock` + CUDA/driver versions). New `paths.py` property.
- **T5.2 — Register stage 10:** add `"10"` to `scripts/run_pipeline.py:_STAGE_MODULES`, wired after stage 09; expose the visualization entry point as importable (thin CLI stays). No module relocation.
- **T5.3 — Opt-in canary docs:** document `scripts/03_annotate.py --canary` (produces `data/interim/canary_drift_audit.jsonl`); keep out of the default 01–09 chain.
- **Wave 5 verification:** `run_pipeline.py --stages 10` works via the registry on smoke; manifest emitted.
- **Wave 5 commit:** `chore(pipeline): add run manifest, env lock, and stage-10 orchestration`.

---

### Wave 6 — Documentation (full pass, Docus/markdown style under `docs/`)
Match existing styling; informative to any audience. Consider the `docus-writer` agent and `review-docs` skill. Use the **real local numbers** from Waves 2–4 (corrected prevalence, released-dataset schema, base-rate precision). Mark the frozen-test-specific figures/metrics (PR/ROC, multi-threshold CMs) as **"finalized after the post-sprint UCloud re-evaluation (§7)"** so no invented numbers appear.
- **T6.1 — Technical docs:** update `docs/agents/pipeline/*`, `AGENTS.md`, `CONTEXT.md`, `config/README.md` for the per-org estimand + expand-back, triple labels, LOW decomposition, base-rate precision, the §7 controlled re-eval + one-shot semantics, new figures, simplified archiving.
- **T6.2 — Refreshed evaluation report:** a new `docs/audits/`-style report reflecting the corrected prevalence + base-rate precision now, with the frozen-test PR/ROC + multi-threshold CM sections stubbed for §7 finalization; supersedes/annotates `religious_evaluation_report.md`.
- **T6.3 — Plain-language overview:** results/methods narrative for non-technical readers — what the classifier claims, what prevalence means, how to read the figures, caveats.
- **T6.4 — Released-dataset data dictionary:** columns of `predictions_full.parquet` — the three labels (recall/max-F1/base-rate), `prob_calibrated`, `decision_source`, `tier`, how to join on raw EIN2, the duplicate-text = same-mission convention.
- **Wave 6 verification:** `review-docs` pass; links resolve; numbers match the Wave 2–4 local artifacts; §7-pending sections clearly flagged.
- **Wave 6 commit:** `docs: refresh pipeline, evaluation, and released-dataset documentation`.

---

## §4. Commit protocol (per wave)

- Commit **once at the end of each wave**, only after that wave's verification passes.
- Work on the current feature branch (`refactor/harmonize-pipeline`); do not commit to `master`.
- **Conventional commit** subject (`type(scope): summary`) as specified per wave, followed by a body explaining **the reason** and **how** it was implemented.
- **Do NOT include a `Co-Authored-By` trailer** (explicit user instruction).
- Include sub-agents' report-back highlights (old-vs-new headline, disjointness result, completeness reconciliation, derived base-rate threshold) in the body where relevant.

## §5. Sprint verification (local, end of sprint)

1. Smoke end-to-end: `run_pipeline.py --config config/smoke.yaml --stages 05,06,07,08,09` + stage 10 — all green, all 8 figures render, no artifact over size guards.
2. Frozen-test integrity: `test_evaluation.json` and `anchor_oof_scores.parquet` are byte-unchanged across all local CPU-backfill/dry-run activity (except the frozen-safe `calibrator.json` PR-points regen).
3. Released dataset (local dry-run): `predictions_full.parquet` covers all 560,354 raw EIN2, zero null `pred_label`, three label columns + `prob_calibrated`.
4. Prevalence (local): report is per-organization, LOW split into PPI + RG sub-strata, tier shares over raw counts; real corrected headline shift recorded.
5. Docs: refreshed, consistent with local numbers, with §7-pending sections flagged.

---

## §7. Post-Sprint Handoff (UCloud — run by the user AFTER the sprint)

Not executed by sprint sub-agents. After Waves 1–6 are committed and locally verified, the user runs one controlled UCloud session to produce the canonical artifacts:

1. **Archive** the current `data/processed/evaluation/{test_evaluation,calibrator,rule_validation}.json` + `anchor_oof_scores.parquet` with the original git sha/date, and capture the original headline metrics.
2. **Controlled stage-07 re-run on UCloud** (environment parity): with the sprint code frozen and tagged, unlock (move/remove existing `test_evaluation.json`) and run stage 07 exactly once; model checkpoint + threshold policy unchanged. Then run stages 08→10 to regenerate the canonical released dataset, prevalence, and all figures (including the real frozen-test PR/ROC and multi-threshold confusion matrices).
3. **Reproduce-assertion:** confirm primary metrics (F1, PR-AUC, precision, recall, MCC, `acceptance.passed`) match the archive within a documented tolerance; a divergence halts and is investigated as nondeterminism. Record the result into `run_manifest.json` (fills the Wave-5 placeholder).
4. **Finalize docs:** populate the §7-pending sections of the refreshed evaluation report and overview with the real frozen-test figures/numbers.
5. **Commit:** `chore(evaluation): controlled one-shot frozen-test re-evaluation on UCloud` (regenerated artifacts + manifest), then `docs: finalize evaluation report with frozen-test results`.

## §8. Out of scope (documented, not done)

- Enlarging gold/test (breaks the one-shot frozen test — Decision 5). Future option: append-only augmentation preserving the existing 175 test rows byte-for-byte.
- Retraining / encoder comparison — not justified by current evidence; revisit only if robustness figures expose a real weakness.
