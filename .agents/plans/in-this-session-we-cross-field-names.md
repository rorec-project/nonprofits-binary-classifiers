---
created: 2026-07-28
status: design agreed, not implemented
supersedes: .agents/stubs/names_with_missions_vs_missions-idea.md (partially — see §0)
---

# Cross-field transfer: scoring the mission-trained encoder on organization names

Design record from a grilling session. Every decision below was made by the user;
the measurements are reproducible from the commands in §8.

## 0. What this is, and what changed under it

Apply the **existing** fine-tuned DeBERTa-v3-base checkpoint (trained on
`LONGEST_MISSION`) to **organization names**, with no retraining and no
target-field adaptation.

**This is not "zero-shot."** The label space is unchanged and the model was
task-trained. The defensible framing is **cross-field transfer under a change of
input view** (covariate shift; Shimodaira 2000, Moreno-Torres et al. 2012). Using
"zero-shot" in a manuscript would draw a reviewer objection. See
`docs/research/20260728-literature-cross-field-transfer-missions-to-names.md`.

### Where the religious mass sits — measure within scope, not pooled

The June stub argues names matter because churches are Form-990-exempt, so the
mission corpus structurally misses a disproportionately religious slice. Measured
against two external IRS flags (`NTEE_IRS` major group `X`, and
`BMF_FOUNDATION_CODE == 10` = church under 170(b)(1)(A)(i)), **within the 501(c)(3)
public-charity scope the pipeline deliberately targets**:

| 501C3 CHARITY | n | religious-flagged |
|---|---|---|
| has mission | 560,354 | 7.7% |
| **name, no mission** | **257,623** | **10.4%** (26,759 orgs) |
| no name, no mission | 14,707 | 9.7% |
| **BMF registry, not in panel** (any subsector) | **2,004,353** | **19.8%** (396,379 orgs) |

The stub's premise **holds**: the name-only stratum is 1.35× enriched over the
mission-having stratum, and 26,759 flagged organizations are unreachable today.

**Recorded so it is not repeated:** an earlier pass of this analysis pooled 501CX
nonprofits and private foundations into the "no mission" bucket and reported 4.4%,
concluding the premise was refuted. That was wrong — those subsectors are
**intentionally out of scope** (`src/corpus/06_collapse_missions.R:64` is a
deliberate filter, not an oversight). Always compute these rates within the 501(c)(3)
frame.

**Consequence for how this is written up:** phase 1 is both a validation vehicle
*and* a real, modest coverage win. Phase 2 (BMF-only, 19.8%) remains the larger
prize. Both claims are defensible; do not inflate phase 1 into the phase-2 number.

## 1. Hard constraints

- **Missions stay first-class and additive-only.** No missions artifact, threshold,
  or prevalence number is altered or reinterpreted. If the arms conflict, missions win.
- **The frozen test is not reopened.** `data/processed/evaluation/test_evaluation.json`
  is untouched. Validation uses `prompt_dev + validation`, the external flags, and a
  new hand-coded sample.
- **No fine-tuning of a names model** this round.
- **This round terminates at classification.** No name-based prevalence figure is
  published; stage 09 is untouched.

## 2. Decisions

| # | Decision | Choice |
|---|---|---|
| D1 | What runs | Encoder cross-field transfer. No LLM-on-names. |
| D2 | Estimand | **Same construct, held fixed** — observable religious *purpose* as a core driver, identical to `prompts/v1.txt`. "St Mary's Hospital" and "Trinity Health" are true **negatives**. |
| D3 | Deliverable | Classification + measured characterization. No prevalence. |
| D4 | Validation | Rungs 1–4 (all, incl. hand-coded sample). |
| D5 | Rung-4 draw | From the **BMF-only** population, stratified on IRS flags + conflict cases. |
| D6 | Target population | Panel **and** BMF-only, both this round — the base-rate-shift test needs both halves. |
| D7 | Cleaner | `cleanco`/`ftfy` + an explicit **religious-token guard**, applied to **both** populations from raw name fields. |
| D8 | Input variant | Bare-cased primary + cased-full ablation. **DBA demoted to diagnostic** (see §5). |
| D9 | Layout | `scripts/names/N1–N5` + new `src/binary_classifier/names/` subpackage. |
| D10 | Artifacts | `data/interim/names/`, `data/processed/names/`. |

D2 is the load-bearing one and deserves an ADR: it is hard to reverse (it governs
how the rung-4 gold is coded), surprising without context, and was chosen against a
real alternative (an auspice/identity construct, which names carry more naturally
but which would make the two arms non-comparable).

## 3. Why transfer might fail — three mechanisms, descending confidence

1. **Construct, not statistics.** The encoder learned from silver labels generated
   under a rubric that demands religious purpose "as a core driver of the
   organization's work" and routes `saint_name_only` → abstain. Names carry
   *identity*. Under D2, a chunk of apparent model failure is the model **agreeing
   with the construct**.
2. **The training frame excluded name-shaped text by design.** Stage 01 sampled
   `Q >= 3.0` (HIGH+MEDIUM). **93.2% of names score LOW** (median name Q = 0.5, 4
   words; median mission Q = 4.5, 22 words). The model has never seen an input of
   this shape — not because names are merely shorter, but because the frame filtered
   them out.
3. **Operating points are distribution-bound.** `calibrator.json` is Platt/temperature
   fit on mission OOF scores; all three labels (`pred_label`, `pred_label_maxf1`,
   `pred_label_baserate`) are cut points on a mission score distribution. Ranking may
   survive the shift; the thresholds will not.

Related: running the existing router over names, **67.7% are labeled negative by
`rule_short_negative`** (the `<6 words and no religious lexicon` branch). On a
mission, shortness is evidence of insufficiency; on a name it is a property of all
names. Only 28.2% would reach the classifier under current routing; 4.1% hit the
strong-tradition rule. **The names arm must not reuse the existing router.**

## 4. Expected results — predictions, pre-registered (INFERRED, not measured)

1. Score distribution collapses toward the negative pole rather than spreading.
2. High precision, poor recall at any mission-derived threshold.
3. ECE degrades badly; the three thresholds become meaningless as-is.
4. **The hypothesis actually worth testing:** the encoder beats the lexicon on
   **precision**, not recall — by correctly rejecting "St Mary's Hospital" and
   "Trinity Health", which a regex fires on and which D2 makes negatives. If transfer
   has value, it lives there. Recall is the regex's game.
5. Agreement with `f(mission)` on the paired overlap ≈ 40–60%, concentrated on
   tradition-token names.

**The bar is not zero, it is a regex.** `apply_rule_label` already fires on names.
Litofcenko et al. 2020 (*Voluntas*) is the closest prior: classifying nonprofits from
names alone, curated keyword rules reached ~85% while their ML attempt was
"unsatisfactory." Caveat: Austrian law requires names to relate to purpose, so 85% is
an optimistic ceiling for the US.

If (4) holds, productionize. If the encoder merely reproduces the lexicon's positives
with worse calibration, that is still a publishable measurement — the literature memo
finds **no study** that trains a BERT-family classifier on long text and quantifies
degradation on much shorter text. We would be producing that measurement, not
looking it up.

## 5. Validation ladder

**Rung 1 — paired overlap** (free; 560,354 orgs, minus 20,841 contaminated = 539,513).
`f(name)` vs `f(mission)` vs `lexicon(name)` vs existing gold. **Falsifies cheaply;
cannot validate** — gold was coded from mission text, so a name-model that is right
about a thin-mission church scores as wrong, and the population is 990-filers, which
excludes the churches that motivate the exercise.

*Contamination:* 20,841 EIN2s (3.7% of the overlap) appear in some manifest
(20,000 silver + 450 gold + 500 anchor + 175 test + 175 validation + 50 each
prompt_dev/monitor). Exclude them from all rung-1 metrics.

**Rung 2 — external IRS quasi-label** (free; available on all 3,436,970 BMF orgs).
Score against `NTEE-X ∪ church-code`. Different construct (auspice, not purpose) —
**but the bias is measurable**: on the overlap you hold the external flag *and* human
gold *and* mission labels simultaneously, so quantify the flag-to-construct offset
there, then carry that known offset onto the population where nothing else exists.
This is what makes the target population validatable at all.

**Rung 3 — behavioral probes** (free; ~200 synthetic strings). "First Baptist Church"
/ "First Community Church" / "First Baptist Soccer Club" / "St Mary's Hospital".
Diagnosis of *why* the numbers came out as they did, not an accuracy claim.

**Rung 4 — hand-coded sample** (~300–500 names, human, under D2). Drawn from
**BMF-only**, stratified on `NTEE-X` / church-code / neither, oversampling conflict
cases: saint-named seculars, faith-heritage naming, non-Christian traditions,
non-English. The only unbiased estimate on the target population, and the prerequisite
if prevalence is ever revisited. This is a **human gate** in the G1/G4 mould.

**Base-rate-shift falsification (free, and the strongest test in the design).**
Score both populations. Their external-flag base rates differ 3.5× (5.7% vs 19.8%).
If the model's positive rate tracks that shift, that is evidence of transportable
signal. **If it returns roughly the same positive rate in both, the model is
responding to input shape rather than measuring religion, and the arm is dead.**
Much harder to pass by accident than anything on rung 1.

**DBA — diagnostic only.** Verified: DBAs exist on 17.0% of orgs, but supply a
tradition token the name lacks in only **579 cases (0.61% of DBA-having, 0.10% of the
corpus)**, and cut the other way 6× more often (3,427). Too thin for a production arm.
Pull the 579 as a hand-inspected case study of name-level false negatives
(e.g. "Heritage Academy" → *Grace Bible Church*; "Bennetts Enrichment Learning Center"
→ *Bennetts Christian Day Care*).

## 6. Data plumbing

**Panel names.** `BEST_NAME_CASED` is already EIN2-invariant (0 of 1,537,078 EIN2s
have >1 distinct value) at 90.5% coverage. `BEST_NAME_BARE_CASED` **exists and is
populated** — 1,390,342 EIN2s (90.45%) — in
`panel/merged/panel_filled_gaps.parquet` (stage 09), and is dropped by stage 10's
deliberate 42-column researcher-facing contract (`src/config.R:277-330`). No upstream
re-run is needed; `panel_final` ⋈ `panel_filled_gaps` on `EIN2` recovers it. Note the
two files are **not** nested: `panel_final` adds `COMMON_LEVEL1`, `FORMATION_YEAR`,
`RULING_YEAR`, `COMMON_LEVEL1_MISMATCH`.

**BMF names.** `ORG_NAME_CURRENT` covers 3,436,969 of 3,436,970 orgs but is **raw
uppercase**, never through the R cleaning chain. Feeding it to DeBERTa would confound
field shift with **casing** shift (uppercase input is documented as harmful to these
models — Mayhew et al. AAAI 2020, cited in the NonProfitData plan). Cleaning is a
prerequisite, not a polish.

**Cleaner acceptance criteria (D7).** Because the guard-list approach forgoes the
free oracle a faithful R port would have had, the criteria must be explicit:

- The guard list is **derived from** `../NonProfitData/docs/content/5.reference/nonprofit-legal-suffixes-report.md`,
  not invented. That study is why the R chain deliberately keeps `foundation`,
  `trust`, `charitable`, `fund` — and protects `ministries` (93,846), `church`
  (22,688), `ministry` (19,863), `temple` (6,576), `synagogue`, `mosque` as
  "substantive religious identity."
- **Divergence audit, pass/fail:** run the cleaner over the 1,390,342 panel raw
  names, diff against the R-produced `BEST_NAME_BARE_CASED`. **Zero cases** where the
  cleaner strips a token the R chain kept *and* that token appears in
  `RELIGIOUS_LEXICON`. Other disagreements are reported, not blocking.
- The **same** cleaner is applied to both populations from raw fields
  (`F9_00_ORG_NAME_L1`/`NAME_CASED` panel-side, `ORG_NAME_CURRENT` BMF-side).
  Cleaning the two differently would confound the base-rate-shift test.

**Non-501(c)(3) missions — deliberately out of scope, closed.** 501CX (189,983) and
private foundations (8,000) do have mission text in `missions_panel.parquet`, dropped
by `src/corpus/06_collapse_missions.R:64`. **That filter is intentional**: the project
scope is 501(c)(3) public charities. These orgs are not a missed coverage
opportunity and must not be counted as part of the "name-only" population. Closed,
not deferred.

## 7. Layout

```
scripts/names/
  N1_build_name_frame.py    panel ⋈ BMF → name cross-sections
  N2_clean_names.py         cleaner + divergence audit (gates on §6)
  N3_score_transfer.py      checkpoint → scores, both populations
  N4_validate.py            rungs 1–3 + base-rate-shift test
  N5_draw_gold.py           rung-4 stratified sample → human gate

src/binary_classifier/names/      new subpackage
reuses: inference/  metrics/  evaluation/  viz/

data/interim/names/               frames, manifests, cleaned names
data/processed/names/
  predictions/                    transfer scores
  evaluation/                     rung 1–3 results
  gold/                           rung-4 coding template  ← human gate
```

`scripts/01–10` and `run_pipeline.py` are untouched.

**Checkpoint:** `data/models/checkpoints/microsoft__deberta-v3-base/default/s44/checkpoint-2690/`
(sha256 `8fd26faa…`, soft targets), per `data/processed/gold/selected_model.json`.
Present locally; the experiment is runnable today.

## 8. Verified vs inferred

**Verified** (direct query, reproducible):
560,354 mission rows / 560,354 unique EIN2 / 0 empty missions; 1,537,078 panel EIN2,
0 with >1 distinct `BEST_NAME_CASED`, 90.5% coverage; `BEST_NAME_BARE_CASED` present
and populated in `panel_filled_gaps` at 1,390,342; the in-scope 7.7% / 10.4% / 9.7%
flag rates and the 19.8% BMF-only rate;
3,436,970 BMF rows, 435,097 NTEE-X, 314,076 church-code, 478,014 either;
mission-havers 560,354 vs name-only 257,623 within 501C3 CHARITY; 93.2% of names LOW
vs 23.1% of missions; router-on-names 4.1% / 67.7% / 28.2%; 20,841 contaminated EIN2s
(3.7%); DBA present 17.0%, 579 adds / 3,427 subtracts; the deliberate
`06_collapse_missions.R:64` 501(c)(3) filter.

**Inferred** (stated as such): every prediction in §4; that the encoder's failure mode
will be silence rather than noise; that 579 DBA cases is too thin to productionize
(a judgement, not a measurement); that porting-vs-guard-list trades oracle for effort.

## 9. Open, deferred

- **Prevalence.** Requires a name-only anchor and stratum-combining; PPI++ assumes a
  shared feature distribution and its covariate-shift extension needs a known density
  ratio over a shared feature space, which a field change destroys. Rogan–Gladen fails
  identically (sensitivity/specificity are not transportable — spectrum bias,
  Ransohoff & Feinstein 1978 NEJM). Rung 4 is the prerequisite if this is revisited.
- **Upstreaming the cleaner** to NonProfitData once settled.

Closed during the session, recorded so they are not reopened: **subsector scope**
(501(c)(3) public charities only, by deliberate design — see §6) and the
`06_collapse_missions.R:81-86` **parse defect** (a bad-edit bug, since fixed upstream).
