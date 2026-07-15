# Independent Architecture Review

**Reviewed:** 2026-06-10
**Branch:** `refactor/harmonize-pipeline` (`7af0ba5`)
**Scope:** The whole repository as it stands after the two implemented sprints —
PR1/PR2 (`let-s-design-a-plan-idempotent-harbor.md`) and the harmonize sprint
(`with-a-specialized-subagent-glimmering-knuth.md`, commit `7af0ba5`) — read
against (a) the implemented plans in `.agents/plans/`, (b) the research synthesis
in `docs/research/`, (c) the roadmap in `.agents/stubs/pipeline-roadmap.md`, and
(d) the prior best-practices review (`docs/audits/20260609-pipeline-review-best-practices.md`).
**Type:** Read-only review. The only artifact produced is this document; no
source or data was modified. (`skills-lock.json` was already dirty in the working
tree and is unrelated.)
**Health at review time:** `uv run pytest` → **92 passed**; `uv run ruff check .`
→ clean; `uv run ty check` → clean.

---

## 1. Headline

**The data-production half of the project is built and genuinely solid; the
modeling-and-measurement half — which is the project's actual reason to exist —
does not exist yet.**

The 2026-06-09 review found the pipeline's three QC mechanisms silently inert and
the validation loop open. **Every High/Medium defect from that review is now
closed in code** (traceability table in §3), the validation loop is closed with
two blocking human gates, and the weak-supervision spine (stages 01–04) is
well-engineered, tested, and internally consistent. That is real and creditable
progress.

But stages 01–04 only produce `silver_labels.csv` — a frozen weak-supervision
training set. **Nothing consumes it.** Training (05), evaluation (06),
inference-at-scale (07), and visualization (08) are unbuilt, and so is the entire
prevalence/calibration stack that the roadmap and
`docs/research/20260605-replication-calibration-prevalence.md` name as the
_locked estimand_ — the population share of religious nonprofits over **all**
nonprofits. The two roadmap hooks that exist for that goal —
`apply_rule_label` (LOW-tier rule layer) and the per-cell `inclusion_prob`
design weight — are implemented but have **zero consumers**.

So the honest status is: **~50% of the way to the goal.** The half that is done
is done well. The findings below separate (a) _unexpected_ latent flaws and
incoherences worth fixing now from (b) _expected_ roadmap gaps that are
documented and not defects.

---

## 2. What the pipeline does well

These are real strengths and they match the plans and the research synthesis:

- **The validation loop is genuinely closed.** Two graceful human gates
  (`qc/preflight.py`): **G1** rejects a freeze/bake-off unless `gold_to_code.csv`
  carries strict `0/1` for the required split; **G2** requires a human-confirmed
  `production_slate.json`. The orchestrator exits `2` (no GPU/API touched) on any
  gate failure (`scripts/run_pipeline.py:130-163`).
- **The QC freeze gate actually blocks** (`qc/agreement.py:194-217`): it raises
  and writes nothing unless Cohen's κ ≥ `kappa_threshold` **and** the bootstrap-CI
  lower bound on minority-class F1 ≥ `f1_ci_floor`. The full sklearn bundle
  (confusion matrix, minority P/R/F1, MCC, balanced accuracy, κ, Krippendorff α,
  PR-AUC, bootstrap CIs) is computed, not just advertised.
- **Leak guard before freeze** (`agreement.py:297-342`): all gold-manifest EIN2s
  (prompt_dev/validation/test/monitor) are excluded from `silver_labels.csv` as an
  _exclusion_ (not a silver keep-list), so a held-out test row is dropped even when
  it also appears in silver.
- **Evidence-span hallucination guard** (`qc/evidence.py`): every stored
  `evidence_span` is checked as a verbatim substring of the source text; fabricated
  positives can be abstained pre-aggregation (config-gated).
- **Drift monitoring is real** (`run_annotation.py:342-407`): a held-out `monitor`
  slice, a hashed canary set, model fingerprints, and a κ/α-over-time change test
  against the first baseline run.
- **Config-driven, typed, entity-agnostic, seeded throughout.** One Pydantic
  `BinaryClassifierConfig` → `PathRegistry`; retasking is a YAML copy; `EIN2`
  carried through every artifact; resume keyed by `(EIN2, source_id)`.
- **Honest quarantine of dormant arms** (`aggregate.py:85-135`): Dawid–Skene and
  CROWDLAB raise `NotImplementedError` with cited reasons rather than silently
  returning empty labels.
- **Tested.** 12 test files, 92 cases, covering the gates, leak guard, metric
  bundle, resume, sampling, evidence, and the orchestrator — the regression
  coverage whose absence let the prior D1 land.

---

## 3. Prior defects — traceability (all High/Medium closed)

Every defect from the 2026-06-09 review, mapped to the code that closes it.

| ID  | Prior defect                              | Status                               | Evidence                                                                                                                                                                                                                       |
| --- | ----------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D1  | Stage 03 drops `prompt_id`, breaks resume | **Closed**                           | real prompt stem threaded via `make_annotator`; `source_id=f"{id}__{prompt_id}"` matches resume key (`run_annotation.py:290-297`, `base.py:81-83`)                                                                             |
| D2  | Validation + agreement gate inert         | **Closed**                           | G1/G2 (`preflight.py`); gate raises (`agreement.py:194-217`); human artifact `gold_to_code.csv` defined + emitted (`sample.py:423-470`)                                                                                        |
| D3  | Silent synthetic-data fallback            | **Closed**                           | `allow_synthetic` (default `False`→`FileNotFoundError`); `data_source` stamp; `atexit` cleanup (`load.py:76-91,33-40`)                                                                                                         |
| D4  | Raw % agreement; "Krippendorff" overclaim | **Closed**                           | full sklearn bundle + bootstrap CIs + κ + Krippendorff α all computed (`agreement.py:345-413`)                                                                                                                                 |
| D5  | OpenAI ignores `guided_json`              | **Closed**                           | `response_format={"type":"json_schema",…,"strict":True}` from the single `build_json_schema()`; vLLM uses same via `guided_json` (`openai_annotator.py:110-117`, `vllm_annotator.py:105-107`)                                  |
| D6  | O(n²) annotation store                    | **Closed (CSV); residual (parquet)** | CSV is append-mode + vectorized `done_set` (`schema.py:379-397,334-346`); the `.parquet` branch still loads+rewrites (`schema.py:385-388`) — non-default, minor                                                                |
| D7  | Hardcoded CWD paths                       | **Closed**                           | all stores route through registry (`annotation_store`, `bakeoff_results`, `bakeoff_store`, `proposed_slate`); no CWD-relative writes remain                                                                                    |
| D8  | Reproducibility overstated                | **Closed (caveat)**                  | `system_fingerprint` captured (`openai_annotator.py:136`); docstrings softened to "best-effort". _Model IDs are still floating aliases_ — pinning is a documented user step (`config.py:66-83`)                                |
| D9  | No tests                                  | **Closed**                           | 92 tests                                                                                                                                                                                                                       |
| D10 | Dead fields / minor                       | **Mostly closed**                    | `_read_prompt_file` + orphaned import removed (`base.py`); canary uses the real monitor slice; `is_positive_enriched` divergence documented. _Carry-over:_ bake-off `_score_vs_human` still omits MCC/balanced-acc — see §4 F2 |
| D11 | Evidence spans unenforced                 | **Closed (off by default)**          | `qc/evidence.py` wired into `run_quality_check` (`agreement.py:111-112,250-294`); ships gated behind `abstain_on_fabricated_positive: False` (deliberate T8 choice) — built and wired, but not active in the shipped config    |

---

## 4. Unexpected gaps & latent flaws (new findings)

Tiered by severity. These are _not_ on the roadmap; they are incoherences or
landmines that the two sprints left behind.

### High

**F1 — `TrainingConfig` is a generic HF stub disconnected from the researched
stage-05 decisions; `fp16:true` directly contradicts the locked "bf16 on B200"
decision.**
`config.py:250` and `config/religious_missions.yaml:82` set `fp16: true`. The
roadmap, both plans, and `configuration.md` repeatedly lock the opposite:
_"bf16 not fp16 on the Blackwell B200 node"_ (harmonize plan "Decisions locked";
README/configuration roadmap). There is **no `bf16` field anywhere**
(`grep -rni bf16` → none), no encoder model name (the roadmap locks
DeBERTa-v3-base), and no soft-label / label-smoothing / class-weight knob — all
of which the research and roadmap specify. This is not a live bug (nothing
consumes `TrainingConfig` yet), which is exactly why it is dangerous: when stage
05 is wired, the config will silently steer training to fp16 on hardware where
bf16 is the documented choice, and the researched training decisions will have to
be rediscovered. **Fix now (cheap):** replace `fp16` with `bf16: true`, add
`encoder_model: microsoft/deberta-v3-base`, and stub the soft-label/label-smoothing
knobs so the config encodes the decisions the docs already made. Treat it as a
documentation-coherence fix, independent of any benchmark claim.

### Medium

**F2 — The two human decision points use inconsistent criteria; the
"legacy / reported-only" `agreement_threshold` is silently load-bearing for the
slate the human confirms.**
Stage 04's freeze gate was upgraded to κ + minority-F1 CI floor. But stage 02's
bake-off still selects the **proposed slate** purely on raw `accuracy`
(`bakeoff_prompts.py:_score_vs_human` reports accuracy/P/R/F1 only;
`_build_proposed_slate` keeps combos with `accuracy >= threshold`,
`bakeoff_prompts.py:196-229`), and that `threshold` is
`cfg.qc.agreement_threshold` (`bakeoff_prompts.py:96`) — the very field that
`config.py:213` and `configuration.md` document as _"legacy … logged for
continuity, no longer the freeze gate."_ So the metric the literature most warns
against on an imbalanced task (accuracy) is the one that picks the models a human
promotes at G2, while a _different_, stricter metric later decides whether their
labels can freeze. A slate can clear stage 02 and then be unfreezable at stage 04.
**Fix:** report the imbalanced bundle (MCC, minority-F1) in `bakeoff_results.json`
and base `_build_proposed_slate` on the same quantities the QC gate will demand;
at minimum, stop calling `agreement_threshold` "reported-only" while it gates
model selection00-642`). With
~60 minority positives, the bootstrap CI on minority-F1 is wide, and the gate
demands its _lower bound_ ≥ 0.70 on the hardest cases the project could assemble.
The harmonize plan itself flagged the CI floor as "fragile on a 4-way split of
~400." **Recommendation:** before the first real freeze, run the gate against the
actual coded gold and sanity-check that a genuinely good ensemble can pass; if
not, recalibrate, or stratify the gate (report boundary vs clear-case slices
separately) so adversarial gold doesn't block a deployable ensemble.

**F4 — Lexicon circularity in gold/test selection is unaddressed.** (Risk; this
was R2 in the prior review and neither sprint touched it.)
The same `RELIGIOUS_LEXICON` that enriches the positive pool also drives
`apply_rule_label` and shapes which positives reach the gold set
Design") prescribes a multi-label coding layer — `religious_purpose_explicit`,
`religious_identity_or_affiliation`, `religious_service_content`,
`faith_inspired_ambiguous` — with the binary derived later, specifically so
"faith-inspired-but-ambiguous" can be held out for sensitivity analysis. The gold
template captures only a binary `human_label` (`sample.py:460-467`). Human gold
coding is the **one expensive, irreversible human-in-the-loop step**; coding only
a binary now means every future audit/sensitivity analysis the literature calls
for requires **re-coding from scratch**. The cheap insurance is to enrich the
coding template's columns _before_ the human codes — even if only the binary
feeds downstream initially. Defer the downstream use, but capture the richer
labels now; the cost asymmetry is the whole point.

**L2 — Gold size.** Total gold is 450, split four ways (prompt_dev 50 / monitor
50 / validation ~175 / test ~175). The synthesis recommends 300–800 hand-labeled
examples; 450 is in range but on the low side once split four ways, and it is what
makes F3's CI floor fragile. Consider whether validation/test warrant more rows
before the gate becomes load-bearing.

**L3 — Bake-off / metric alignment.** Ties into F2: the synthesis makes PR-AUC,
minority P/R/F1, and MCC primary for the imbalanced task and accuracy explicitly
secondary; the bake-off still selects on accuracy.

---

## 6. Roadmap gap analysis (expected — documented, not flaws)

These are the _known_ unbuilt stages. They are not criticized as defects; they
are listed so the "what's left" picture is honest and so the unbuilt work is
checked against current state-of-the-art.

| Stage                         | State                             | SOTA check against the roadmap's own plan                                                                                                                                                                                                                           |
| ----------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 05 Train                      | Unbuilt; config stub only         | Roadmap locks DeBERTa-v3-base + bf16 + soft-label/label-smoothing. Config encodes **none** of this (see F1). ModernBERT as a throughput arm remains current.                                                                                                        |
| 06 Evaluate                   | Unbuilt                           | Metric bundle exists in QC and is reusable; roadmap adds decision-curve/net-benefit + ECE — both current and correct. No calibration split is defined yet (the literature insists calibration precedes prevalence).                                                 |
| 07 Inference at scale         | Unbuilt                           | `apply_rule_label` (the LOW-tier cascade) exists but is unconsumed; needs model_version/checkpoint_hash stamping per the roadmap.                                                                                                                                   |
| 08 Visualization              | Unbuilt                           | n-gram word clouds + metric dashboards — no dependencies yet.                                                                                                                                                                                                       |
| Prevalence / population share | Unbuilt (the **locked estimand**) | PPI++ primary + SLD/EMQ + KDEy/DyS via QuaPy + per-NTEE calibration is a current, well-chosen stack. The hooks (`inclusion_prob` per-cell weight, LOW rule-layer mass) are in place but unwired. This is the project's _raison d'être_ and is entirely future work. |

**The critical-path observation:** the project's stated purpose is a _prevalence
estimate over all nonprofits_, but the sampling frame is HIGH+MEDIUM only (Q≥3.0)
and positive-enriched to 35%. The estimate therefore depends on machinery that
does not exist yet (anchor sample over the full frame incl. LOW, design-weight
reweighting, quantification correction). Until that is built, no population number
can be reported — and the frame/enrichment decisions being correct _now_ (which
they are: per-cell `inclusion_prob` is right) only pays off once a consumer exists.

---

## 7. Prioritized next steps

Ordered to (a) protect the work already done, (b) close the cheap incoherences,
then (c) start the modeling half.

1. **Fix `TrainingConfig` to encode the researched decisions (F1).** `bf16` not
   `fp16`; add `encoder_model: microsoft/deberta-v3-base`; stub the
   soft-label/label-smoothing/class-weight knobs. Pure config + docs; do it before
   anyone writes stage 05 against the current stub.
2. **Harmonize the two human decision points (F2/L3).** Report MCC + minority-F1
   in `bakeoff_results.json` and select the proposed slate on the same quantities
   the freeze gate enforces; rename or re-document `agreement_threshold` so it is
   not "reported-only" while gating model selection.
3. **Enrich the gold coding template before the human codes (L1).** Add the
   multi-label columns now; the human-coding step is the irreversible bottleneck.
4. **Validate the freeze-gate thresholds against real coded gold (F3)** before the
   first production freeze; recalibrate or stratify if adversarial gold blocks a
   good ensemble. Revisit gold size (L2) at the same time.
5. **De-circularize the frozen test set (F4)** and **emit a conflict/abstain
   review queue (F5)** — both are cheap given the existing long/tidy store.
6. **Then start the modeling half:** stage 05 (DeBERTa-v3-base + bf16, fixed
   seeds, early-stop on _human_ validation F1/MCC, plain baseline first) → stage 06
   (reuse the QC bundle; add a calibration split + ECE before any prevalence work)
   → wire `apply_rule_label` and `inclusion_prob` into stage 07 → the PPI++/QuaPy
   prevalence estimator the project exists to produce.
7. **Add a one-line index for `.agents/plans/`** so the obfuscated durable audit
   files are navigable (F6).

---

## 8. Bottom line

The team did exactly what the prior review and the two plans asked: the
weak-supervision spine is closed-loop, gated, tested, and clean. The unexpected
issues are small and cheap — a config stub that contradicts a locked decision
(F1), an accuracy-vs-κ incoherence between the two human gates (F2), and a freeze
gate calibrated to a benchmark that never ran (F3). Fix those three config/coherence
items, capture the richer human labels once (L1), and the project is well
positioned to start the half it was actually built to deliver: a fine-tuned
classifier and a calibrated population-prevalence estimate over all US nonprofits.
</content>
</invoke>

# Independent Architecture Review

**Reviewed:** 2026-06-10
**Branch:** `refactor/harmonize-pipeline` (`7af0ba5`)
**Scope:** The whole repository as it stands after the two implemented sprints —
PR1/PR2 (`let-s-design-a-plan-idempotent-harbor.md`) and the harmonize sprint
(`with-a-specialized-subagent-glimmering-knuth.md`, commit `7af0ba5`) — read
against (a) the implemented plans in `.agents/plans/`, (b) the research synthesis
in `docs/research/`, (c) the roadmap in `.agents/stubs/pipeline-roadmap.md`, and
(d) the prior best-practices review (`docs/audits/20260609-pipeline-review-best-practices.md`).
</invoke>

# Independent Architecture Review

**Reviewed:** 2026-06-10
**Branch:** `refactor/harmonize-pipeline` (`7af0ba5`)
**Scope:** The whole repository as it stands after the two implemented sprints —
PR1/PR2 (`let-s-design-a-plan-idempotent-harbor.md`) and the harmonize sprint
(`with-a-specialized-subagent-glimmering-knuth.md`, commit `7af0ba5`) — read
against (a) the implemented plans in `.agents/plans/`, (b) the research synthesis
in `docs/research/`, (c) the roadmap in `.agents/stubs/pipeline-roadmap.md`, and
</invoke>

# Independent Architecture Review

**Reviewed:** 2026-06-10
**Branch:** `refactor/harmonize-pipeline` (`7af0ba5`)
**Scope:** The whole repository as it stands after the two implemented sprints —
PR1/PR2 (`let-s-design-a-plan-idempotent-harbor.md`) and the harmonize sprint
(`with-a-specialized-subagent-glimmering-knuth.md`, commit `7af0ba5`) — read
against (a) the implemented plans in `.agents/plans/`, (b) the research synthesis
in `docs/research/`, (c) the roadmap in `.agents/stubs/pipeline-roadmap.md`, and
</invoke>
