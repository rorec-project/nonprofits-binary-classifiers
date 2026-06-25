# Final Pipeline Review — Full Harmonization (Stages 01–11)

**Reviewed:** 2026-06-16
**Branch:** `refactor/harmonize-pipeline` (`9768142`)
**Scope:** The whole built pipeline now that the modeling-and-measurement half
exists — stages 01–09 wired into `scripts/run_pipeline.py`, plus the two
script-only extensions (10 visualize, 11 aggregation-compare) — read against
(a) the prior audits (`docs/audits/20260609-…`, `docs/audits/20260610-…`),
(b) the research synthesis in `.agents/docs/`, (c) the roadmap and decision log
(`.agents/plans/we-work-on-the-floofy-wreath.md`, `.agents/ralph/state/DEVIATIONS.md`),
and (d) the actual producer→consumer artifact contracts between stages.
**Type:** Read-only review. The only artifact produced is this document; no
source or data was modified.
**Health at review time:**
- `uv run pytest -m "not slow and not network"` → **202 passed, 4 deselected** (5m18s).
- `uv run pytest tests/test_e2e_stages_05_11.py` (the `slow` 05→11 E2E) → **1 passed** (9.6s, offline) — executed separately because it is one of the 4 deselected above.
- `uv run ruff check .` → **clean**.
- Pure-logic subset (metrics/aggregate/preflight/thresholds) → 29 passed in 6s.
- `ty` not re-run this pass (prior audit reported clean).
- ⚠️ Plain `uv run pytest` (the command documented in `python-standards.md`)
  pulls in the `slow` E2E and `network` HF-download tests because there is no
  `addopts` filter — see DOC-5. Use the `-m` filter above for an offline run.

---

## 1. Headline

**The half the 2026-06-10 review said "does not exist yet" now exists, and it is
built to the same standard as the data-production half.** Stages 05–09 (anchor →
train → evaluate → infer → prevalence) are wired into the orchestrator behind two
new human gates (G3 test-unlock, G4 anchor-labels); the calibrated-prevalence
estimand the project exists to deliver (PPI++ over HIGH/MEDIUM + Rogan–Gladen over
LOW, recombined to a population composite) is implemented, not stubbed; the prior
review's headline config defect (F1: `fp16` vs `bf16`, no encoder, no soft-label)
is **resolved** in the current `TrainingConfig`/YAML. The system is config-driven,
typed, seeded, `EIN2`-keyed, path-centralized, and covered by a real offline
end-to-end test that walks 05→11.

The remaining issues are not in the algorithms — they are at the **seams**: two
real producer→consumer breaks where a genuine implementation exists but the
pipeline wires a different (or no) path, one carried-over metric incoherence
between the two model-choosing gates, and a documentation set that now
under-describes and internally contradicts what the code actually does. None of
them break the default majority-vote path, but two of them silently disable
documented capabilities.

**Finding index**

| ID  | Sev    | One-liner                                                                        |
| --- | ------ | -------------------------------------------------------------------------------- |
| H1  | High   | Cross-fit OOF is dead code → CROWDLAB arm can't run; pruned arm is a permanent proxy |
| H2  | Medium | Bake-off still selects the slate on raw accuracy (carryover F2), inconsistent with every other gate |
| H3  | Medium | Stage 07 emits `value: null` rule metrics that stage 09 cannot consume (latent crash) |
| H4  | Low-Med| Orchestrated stage 06 only sweeps; the deployable final checkpoint needs `06_train.py --final` |
| D1  | High*  | README claims "01–09 built" yet describes 05–11 as roadmap; operator guide stops at 04 |
| D2  | Medium | `pipeline.md` / `configuration.md` still say "01–04 exist" / "roadmap" — stale vs code |
| D3  | High*  | No operator instructions for stages 05–09, the G3/G4 human steps, `selected_model.json`, or per-stage CLI options |
| D4  | Low    | README documents only G1/G2; G3/G4 are undocumented for users                    |
| D5  | Low    | pytest markers say "excluded by default" but no `addopts` enforces it            |
| D6  | Low    | README architecture tree omits the train/eval/inference/prevalence/viz packages + scripts 05–10 |
| S1  | Low    | PPI++ reaches into `ppi_py` private internals (`_calc_lam_glm`) — version-fragile |
| S2  | Low    | EMQ documented "via QuaPy" but the wired path is the vendored SLD/EMQ; QuaPy-EMQ wrapper is dead code |

\* High *relative to the user's explicit ask* ("documentation explaining how to
run each stage with the different options"); not a code-correctness defect.

> **Resolution (2026-06-16, same day).** All findings above were implemented in a
> follow-up pass and verified (offline suite green, ruff clean, 05→11 E2E passes).
> - **H1** — cross-fit OOF wired into stage 06 via a new
>   `encoder.finetune_predictor`; gated on the pruned arm / CROWDLAB, idempotent,
>   with an injectable seam. `crossfit._default_finetune` now resolves to the
>   predictor (not the row-returning `finetune`). The pruned arm now consumes true
>   OOF and `oof_pred_probs.parquet` is produced.
> - **H2** — the bake-off scores the imbalanced bundle and selects the slate on
>   κ + minority-F1 CI (the freeze-gate criteria); `agreement_threshold` is
>   reported only.
> - **H3** — stage 09 falls back to the uncorrected observed LOW rate (with a
>   `correction_applied: false` flag) when rule sensitivity/specificity is null,
>   instead of crashing on `float(None)`.
> - **H4** — orchestrated stage 06 now runs sweep→final automatically.
> - **S1** — PPI++ relies on `ppi_py`'s public `lam=None` auto-tuning; the private
>   `_calc_lam_glm` dependency and `_auto_lam` are removed.
> - **S2** — the vendored SLD/EMQ is the single canonical EMQ; the dead
>   `quapy_emq_prevalence` wrapper was deleted (KDEy still uses QuaPy).
> - **D1–D6, D5** — README + `pipeline.md` + `configuration.md` reconciled
>   (status, operator steps 05–09, G3/G4, per-stage CLI table, architecture tree,
>   EMQ prose); pytest `addopts` added.
>
> **Verification caveat:** the real-encoder `finetune_predictor` shares all
> primitives with the verified `finetune` but, like it, only runs under a
> GPU/`slow` test — its stage-06 wiring is covered offline by a stubbed
> control-flow test, not a real fine-tune. **Residual (new follow-up):** stage-11
> CROWDLAB *scoring on the validation split* additionally needs classifier
> probabilities for the gold/validation rows; the training OOF covers silver rows
> only, so producing `oof_pred_probs` removes the hard `FileNotFoundError` but
> full CROWDLAB-vs-majority validation scoring needs gold-row probabilities too.

---

## 2. What the pipeline does well (verified this pass)

- **The harmonization spine is real.** One Pydantic `BinaryClassifierConfig`
  (`config.py`) → one `PathRegistry` (`paths.py`) exposing *every* artifact as a
  resolved `pathlib.Path`; stages consume only `(cfg, registry)`. I traced no
  CWD-relative or hand-built artifact paths in stages 05–11.
- **Gates G1–G4 are coherent and graceful** (`qc/preflight.py`,
  `run_pipeline.py`): G1 strict-`0/1` labels per dependent split, G2 confirmed
  slate, G3 test-unlock with **checkpoint-SHA + acceptance-snapshot matching**,
  G4 fully-coded anchor. Any failure exits `2` with a fix hint and touches no
  GPU/API. The ordering (run 02 to produce the *proposed* slate, then stop at G2)
  is deliberately wasteless.
- **The QC freeze gate blocks** (`qc/agreement.py:186-218`): raises and writes
  nothing unless Cohen's κ ≥ threshold **and** bootstrap minority-F1 CI-lower ≥
  floor; the evidence-span hallucination guard runs *before* aggregation; gold
  EIN2s are removed from the frozen artifact by an *exclusion* guard.
- **Prevalence is genuine SOTA, not a stub.** PPI++ is the real thing — a
  power-tuned λ computed via `ppi_py`'s GLM optimizer and passed to the mean
  estimator/CI (`prevalence/ppi.py:63-95`); LOW tier gets a Rogan–Gladen
  sensitivity/specificity correction with a Wilson-interval sensitivity band;
  the two strata recombine into a variance-propagated composite; per-NTEE
  estimates fall back to EMQ and suppress small cells (`prevalence/estimate.py`).
- **Aggregation arms are honestly wired** to their libraries — `crowdkit`'s
  `DawidSkene` and `cleanlab`'s `get_label_quality_multiannotator(... "crowdlab")`
  (`annotate/aggregate.py`), not hand-rolled (subject to H1 for CROWDLAB's input).
- **Artifact contracts I verified hold:** 07→09 (`anchor_oof_scores` carries the
  exact `prob_calibrated_oof, human_label, tier, sample_prob` columns stage 09
  requires) and 08→09 (`predictions.parquet` carries `pred_label, prob_calibrated,
  tier, ntee_major_group`). The inference router keeps **all** HIGH/MEDIUM rows on
  the classifier (`inference/router.py:41-42`), so the PPI corpus mean is not
  silently biased by rule-routed HM rows.
- **Reproducibility/provenance** stamped through stage 08 (`config_hash`,
  `git_sha`, `checkpoint_sha256`, `pipeline_version`, `inference_date`) and the
  one-shot frozen-test report (`evaluation/evaluate.py`).
- **Prior R1 (frame/prevalence gap) is now addressed by construction:** the
  `inclusion_prob`/`sample_prob` design weights that had "zero consumers" in the
  06-10 review are now consumed — anchor design weights feed the weighted-PPI HM
  estimate, and LOW mass is folded back via Rogan–Gladen. The consumer exists.
- **The 05→11 E2E test passes offline (verified, not just read).** It walks
  anchor → train (stubbed `finetune`) → evaluate → infer → prevalence → viz →
  aggregation-compare with an injected predictor, and asserts the merged
  predictions cover every input `EIN2`. Its passing also proves `build_anchor`
  actually writes `sample_prob` (otherwise weighted-PPI would drop every anchor
  row and stage 09 would raise). **Coverage caveat:** the fixture sets
  `q_thresholds.MEDIUM = 0.0`, so **no row is LOW** and `comparison_arms` stays
  empty — i.e. the Rogan–Gladen/LOW path (the H3 crash region) and the CROWDLAB
  arm (H1) are never exercised. That is precisely why H1 and H3 escaped the
  suite; the green E2E covers the HIGH/MEDIUM + classifier path only.

---

## 3. Harmonization findings (cross-stage seams)

### H1 — High · Cross-fit OOF is dead code; CROWDLAB can't run and the pruned arm is a permanent proxy

`train/crossfit.py:compute_oof_pred_probs` is a correct, validated, *unit-tested*
stratified-K-fold OOF routine that writes `registry.oof_pred_probs` with
`[EIN2, fold, p0, p1]`. **It has no production callers** — the only references are
in `tests/test_crossfit.py`. Nothing in `trainer.py`, `sweep.py`, or any
`scripts/` invokes it, so `oof_pred_probs.parquet` is **never produced by a real
run**. Two documented capabilities silently degrade:

1. **Stage 11 CROWDLAB cannot run.** `qc/aggregation_compare.py:159-162` →
   `_load_oof_pred_probs` raises `FileNotFoundError("…Run stage 06 first")` because
   the file is never written; `aggregate.py:248-252` then refuses without
   `pred_probs`. So if a user sets `aggregation.comparison_arms: [crowdlab]`, the
   stage fails. One of the three advertised aggregation arms is non-functional
   end-to-end (the E2E test only exercises `majority`, so this is untested).
2. **The stage-06 pruned arm is a permanent vote-share proxy.**
   `sweep.py:_oof_probs_for_pruned` loads true OOF *if the file exists*, else falls
   back to `p1 = p_pos` (vote share). Since the file never exists, the fallback is
   always taken. `DEVIATIONS.md` (PR-2 T2.7) framed the proxy as transitional
   pending "true-OOF integration"; that integration was built (`compute_oof_pred_probs`)
   but never connected, so the proxy is in fact permanent.

**Fix:** call `compute_oof_pred_probs` once in stage 06 (e.g. after the selected
cell is known, or as a dedicated step writing `registry.oof_pred_probs`) so both
the pruned arm and CROWDLAB get genuine OOF probabilities; or, if CROWDLAB and the
true-OOF pruned arm are out of scope, delete the dead `compute_oof_pred_probs`,
drop `crowdlab` from the documented `comparison_arms`, and re-label the pruned arm
as proxy-only in the README. Either way, code and docs should agree.

### H2 — Medium · The two model-choosing gates use inconsistent metrics (carryover of prior F2)

The freeze gate (04) and the frozen-test acceptance gate (07) judge on κ +
minority-F1 CI-lower + ECE — the imbalanced bundle the project's own synthesis
mandates. But the **bake-off (02) still proposes the slate on raw accuracy**:
`_build_proposed_slate` keeps every combo with `accuracy >= threshold` where
`threshold = cfg.qc.agreement_threshold` (`annotate/bakeoff_prompts.py:98,253`),
and `_score_vs_human` still computes only accuracy/P/R/F1 — its own docstring says
the imbalanced bundle "is Phase 2 (T2.1)" (`:277-326`). So the field the config
and `configuration.md` document as *"legacy … no longer the freeze gate"* is the
field that gates which models a human promotes at G2, using the one metric the
literature most warns against on a rare-positive task. A slate can clear 02 and
then be unfreezable at 04. This was F2/L3 in the 2026-06-10 review and is
unchanged. **Fix:** report MCC + minority-F1 (+ bootstrap CI) in
`bakeoff_results.json` via `metrics.compute_metric_bundle` (already used
elsewhere) and select the proposed slate on the same quantities the freeze gate
will demand.

### H3 — Medium · Stage 07 emits `null` rule metrics that stage 09 cannot read (latent crash)

`evaluate.py:_rate_with_ci` returns `{"value": None, "ci": {"lower": None,
"upper": None}, …}` when the denominator is 0 — which happens whenever the anchor
sample contains LOW rows but the rule layer covers none of them (`evaluate.py:413-425`).
Stage 09 then reads `metrics.sensitivity/specificity` through `_extract_rule_metric`
→ `_unit_interval` → `_finite_float(None)`, which raises (`prevalence/estimate.py:707-724,886-897`).
Because `_estimate_low` is reached whenever the corpus has any LOW predictions,
a perfectly valid run with a sparse anchor will crash stage 09 with a
`TypeError`/`ValueError` instead of degrading. **Fix:** have stage 09 treat a
`None`/absent rule sensitivity-or-specificity as "rule layer unvalidated for LOW"
and skip the Rogan–Gladen correction (or fall back to the observed LOW rate) with
a logged warning, rather than letting `float(None)` propagate.

### H4 — Low-Medium · The orchestrated single command can't actually produce a deployable checkpoint

`run_pipeline.py:_run_stage` calls stage 06 as the bare `run_training(cfg, registry)`,
i.e. with defaults `sweep=True, final=False` (`trainer.py:36`). So
`run_pipeline.py --stages 06` runs only the documentation curve + selection sweep
and **never the final-seed refit** that produces the checkpoint whose SHA the
`selected_model.json` / G3 gate / stages 07–08 require. To get there you must run
`scripts/06_train.py --final` (and review the printed skeleton) directly; the
orchestrator exposes none of stage 06's eight options (`--final`,
`--baselines-only`, `--encoder`, `--subset`, …). The "end-to-end single command"
in the README is therefore real only for 01–04. **Fix:** either teach the
orchestrator a two-phase stage 06 (sweep → human selection → `--final`) or
document plainly that 06's final refit + `selected_model.json` is a separate
manual step (it is inherently human-gated anyway).

---

## 4. Documentation findings (the user's explicit ask)

The build outran the docs. The README is the closest-to-truth document but is now
internally inconsistent, and the architecture notes lag the code.

- **D1 (High for this ask) — README contradicts itself on status.** The Status
  banner says *"Stages 01–09 are built …"* yet the later **"Roadmap and
  script-only extensions"** section describes 05–11 in the future tense
  ("**Stage 05 — anchor sample:** draw a representative anchor sample …"), and the
  **Operator Guide** ("Two-Checkpoint Pipeline Loop") stops at stage 04. A reader
  cannot tell whether 05–09 are shipped or planned.
- **D3 (High for this ask) — no operator instructions for 05–09.** There is no
  documented procedure for: running stage 05 and coding `anchor_to_code.csv`
  (the **G4** gate); the **`selected_model.json`** creation step that sits between
  06 and 07; creating `test_unlock.json` (the **G3** gate); the fact that the
  deployable checkpoint needs `06_train.py --final` (H4); or the per-stage CLI
  options (`--infer-limit`, 06's flags, etc.). The orchestrator's *docstring*
  documents G1–G4, but nothing user-facing does. Given the request explicitly
  names "how to run each stage with the different options," this is the central
  gap to close — ideally extend the Operator Guide with a Step 7–11 section and a
  per-stage CLI table.
- **D2 (Medium) — architecture docs are stale.** `docs/agents/pipeline.md`
  still says *"Stages 01–04 exist. Training, evaluation, and inference-at-scale are
  roadmap"* and *"runs 01→04"*; `configuration.md` lists 05–11 under "Roadmap hooks."
  Both contradict `run_pipeline.py`'s `_STAGE_MODULES` (01–09 wired). These are
  explicitly "shallow, code is source of truth" docs, but they currently assert a
  *false* status rather than just being shallow.
- **D4 (Low) — gate docs incomplete.** README's Operator Guide names only Gate 1
  and Gate 2; G3 (test unlock) and G4 (anchor labels) are real and load-bearing.
- **D5 (Low) — pytest marker claim is unenforced.** `pyproject.toml` declares
  `slow`/`network` as "excluded by default," but there is no `addopts`, so the
  documented `uv run pytest` actually runs them (≈5 min + HF network). Add
  `addopts = "-m 'not slow and not network'"` (or fix the comment).
- **D6 (Low) — README architecture tree is out of date.** It lists only
  `data/ annotate/ qc/` packages and scripts `01–04` + `11`; it omits `train/`,
  `evaluation/`, `inference/`, `prevalence/`, `viz/` and scripts `05–10`.

---

## 5. Drift vs SOTA / research

### Documented & intentional — *not* defects (verified against the decision log)

These all match `DEVIATIONS.md` and the README/roadmap and should not be "fixed":

- **One-seed documentation curve** vs the synthesis's "≥5 seeds." Note the curve
  is 1-seed *by design*; model **selection** still uses 3 sweep seeds and the
  **final** refit uses 5 seeds (`config.py`/YAML), so the seed budget is not as
  thin as the phrase suggests.
- **Dropped encoder grid** (RoBERTa/DistilBERT/ELECTRA) in favor of DeBERTa-v3-base
  (primary) + ModernBERT-base (comparison); dropped label-smoothing / focal /
  resampling / confidence-weighted loss.
- **n-gram log-odds bars** instead of word clouds (stage 10).
- **Pruned-arm vote-share proxy** (`DEVIATIONS` PR-2 T2.7) and **`ppi-python`
  package name** (PR-5) are logged — though see H1: the proxy is now permanent
  rather than transitional, which the log does not reflect.

### Genuine but minor drift worth recording

- **S1 (Low) — PPI++ depends on `ppi_py` private internals.** `ppi.py:_auto_lam`
  imports `ppi_py.ppi._calc_lam_glm` (a leading-underscore function) to compute the
  power-tuning λ. Correct today, but a `ppi_py` refactor would break it silently;
  the public `ppi_mean_ci(..., lhat=...)` path or a pinned `ppi-python` version
  would be safer.
- **S2 (Low) — EMQ "via QuaPy" is documented but not wired.** The roadmap says
  cross-checks are "SLD/EMQ and KDEy/DyS **via QuaPy**." In fact `kdey_prevalence`
  uses QuaPy's `KDEyML`, but the EMQ cross-check stage 09 calls is the **vendored**
  `emq_prevalence` (a faithful Saerens–Latinne–Decaestecker loop); the QuaPy
  wrapper `quapy_emq_prevalence` exists but is **dead in production** (tests only).
  Same "two implementations, wrong one wired / other one orphaned" pattern as H1.
  Decide which EMQ is canonical and delete or wire the other; update the doc claim.
- **S3 / S4** are the research-facing faces of **H2** (accuracy-based slate
  selection contradicts the project's own imbalanced-metric synthesis) and **H1**
  (the CROWDLAB weak-supervision comparison the synthesis calls for is only
  half-deliverable). Listed here so the SOTA picture is complete; fixes are in §3.

---

## 6. Phase-by-phase status

| Stage | Verdict | Notes |
| ----- | ------- | ----- |
| 01 sample | Solid | Assertions, `inclusion_prob`, EIN2 manifests. **Confirmed still open:** the gold template is binary-only (`preflight._TEMPLATE_COLS = {EIN2, split, text, human_label}`; anchor template likewise) — prior L1's cheap multi-label-codebook insurance was not taken, and the human-coding step is irreversible. Lexicon circularity in gold/test selection (prior R2/F4) was **not re-verified this pass**. |
| 02 bake-off | Works, but **H2** | Parallelized; resumable store; routes by provider. Selection metric drift is the open issue. |
| 03 annotate | Solid | Resumable by `(EIN2, source_id)`; `guided_json` on both providers; drift/canary monitor. |
| 04 QC freeze | Solid | Blocks on κ + minority-F1 CI; evidence guard; gold-exclusion. |
| 05 anchor | Solid | Full-frame incl. LOW, LOW oversampling, design weights, `sample_prob` written and consumed downstream. |
| 06 train | Solid + **H1/H4** | Matches documented plan (one-seed curve, 3 sweep / 5 final seeds, PR-AUC selection, DeBERTa primary). Pruned-arm proxy + dead cross-fit OOF (H1); orchestrated sweep-only (H4). |
| 07 evaluate | Strong + **H3** | Cross-fit calibration, one-shot frozen-test with SHA + acceptance gate, decision curve, subgroups. `null` rule-metric edge (H3). |
| 08 infer | Strong | Sharded/resumable, capability-based device/precision, routing, calibration applied, EIN2-completeness check, full provenance stamping. |
| 09 prevalence | Sophisticated + **H3/S1/S2** | PPI++ + Rogan–Gladen + composite + EMQ/KDEy cross-checks + per-NTEE w/ EMQ fallback. |
| 10 visualize | Script-only, fine | n-gram log-odds + metric/calibration/forest plots; skips missing inputs. Not in the orchestrator (by design). |
| 11 aggregation-compare | Script-only; **H1** for CROWDLAB | Majority + DS run; adoption rule (minority-F1 CI-lower > majority point) is sound; CROWDLAB input never produced. |

---

## 7. Prioritized next steps

1. **Wire or retire the cross-fit OOF (H1).** Highest leverage: it fixes both the
   non-functional CROWDLAB arm and the permanent pruned-arm proxy, and removes the
   dead `compute_oof_pred_probs`. Pick "wire" or "retire," then make docs agree.
2. **Harden stage 09 against `null` rule metrics (H3).** Cheap, prevents a crash
   on valid sparse-anchor data.
3. **Align the bake-off selection metric with the freeze/acceptance gates (H2).**
   Report and select on the imbalanced bundle the project already computes.
4. **Close the documentation gap (D1–D4).** Extend the Operator Guide with stages
   05–11, the G3/G4 human steps, the `selected_model.json` + `06_train.py --final`
   flow, and a per-stage CLI-options table; reconcile the Status banner, fix the
   stale `pipeline.md`/`configuration.md` status lines, and refresh the
   architecture tree.
5. **Add the pytest `addopts` filter (D5)** so the documented test command is
   offline by default.
6. **Decide the canonical EMQ (S2)** and pin/guard the PPI++ λ access (S1).

## 8. Bottom line

The modeling-and-measurement half landed, and it landed well: the prevalence
stack is real SOTA, the gates are coherent, the artifact contracts I traced hold,
and the offline suite is green. The work left is at the seams and in the docs, not
the math. Two seam bugs (H1, H3) silently disable or crash documented paths and
should be fixed before a real population run; the bake-off metric incoherence (H2)
is a known carryover; and the documentation needs to catch up to a pipeline that
now does far more than it says it does.
