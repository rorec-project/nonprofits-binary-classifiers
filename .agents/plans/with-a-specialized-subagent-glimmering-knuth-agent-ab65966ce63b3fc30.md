---
created: 2026-06-10
---

# Literature-Currency Audit — Sampling / Design-Weights / Prevalence-Estimation

**Type:** Read-only literature-currency audit (no code changes proposed; findings only).
**Date:** 2026-06-10. **Repo:** `nonprofits-binary-classifiers`, branch `refactor/harmonize-pipeline`.
**Scope:** Verify the SAMPLING / DESIGN-WEIGHT / PREVALENCE-ESTIMATION methods (used + planned) are current vs 2024–2026 literature.
**Verdict in one line:** The foundational survey-statistics machinery is **ALIGNED**; the two fast-moving families have drifted — **PPI → PPI++** (DRIFTED, easy fix, already in `ppi_py`) and **quantification: no estimator is selected and the modern KDEy / distribution-matching layer is absent** (GAP). The biggest open question is an *estimand/frame* problem the literature cannot fix, not a method swap.

---

## (a) Actual + planned design, with file:line

### Actual (built) design

**Stratified, positive-enriched silver pool** — `src/binary_classifier/data/sample.py`
- Frame restriction: silver drawn from HIGH+MEDIUM only, i.e. `Q ≥ 3.0` (`sample.py:95-96`, tiers via `quality.py:assign_tier` + `QThresholdsConfig`). LOW tier is excluded from the pool.
- Stratify across 26 NTEE major groups A–Z (`sample.py:99`, `sample.py:111`).
- Proportional allocation with floors/caps: `_allocate_stratum_targets` (`sample.py:486-540`); proportional base `pool_counts/total_pool*total_target` (`sample.py:501`); caps `{B,P}` at 2,500, floors `{V,Y,U}` at 200 (`sample.py:41-44`, `504-513`); residual redistributed proportionally over adjustable strata (`sample.py:515-535`).
- **Positive enrichment** to `_TARGET_POSITIVE_SHARE = 0.35` per stratum (`sample.py:45`, `132-145`); positives = religious-lexicon hit or rescue rule (`sample.py:102-108`).
- **`inclusion_prob` (current):** `n_target / len(stratum_df)` — a **stratum-level** rate only (`sample.py:159-161`, mirrored in gold at `sample.py:249-250`). It does **not** reflect the pos/neg enrichment, so as written it is *not* a usable Horvitz–Thompson weight for anything drawn under enrichment. Written to manifests at `_write_manifest` (`sample.py:455-470`); carried through `scripts/01_build_sample.py:100-101`.

**Q rubric driving enrichment** — `src/binary_classifier/data/quality.py`
- `compute_quality_score` (`quality.py:425-482`), tiers HIGH ≥5.0 / MEDIUM 3.0–4.5 / LOW <3.0 (`quality.py:430-431`, `assign_tier:580-599`).
- `has_religious_lexicon` (`quality.py:537-541`), `has_strong_tradition_lexicon` (`544-548`), positive-protective rescue `is_positive_rescue` for `4.5 ≤ Q < 5.0` (`quality.py:558-577`).
- `apply_rule_label` (`quality.py:602-633`) = the high-precision LOW/bare-label rule layer used at inference "to protect prevalence" (README.md:182).

**Gold set + human splits** — `build_gold_set` (`sample.py:169-255`, ~400, boundary-case retention), `split_human_sets` → prompt_dev/validation/test (`sample.py:258-306`). Sizes: silver 20,000 / gold 400 / prompt_dev 50 (`config.py:166-168`); SEED=42 (`config.py:260`).

### Planned design (per current plan + project tech-doc `docs/research/20260606-tech-calibration-quantification-prevalence.md`)
1. Fix `inclusion_prob` to the **per-cell (stratum × pos/neg) sampling rate** → a usable HT design weight.
2. Keep training **enriched**.
3. Add a **separate representative test/prevalence sample at the true prior**.
4. Add a **prevalence-CORRECTION (quantification) step** tied to inference (tech-doc default: average calibrated probabilities = PCC, with ACC/PACC/EM/SLD as sensitivity; PPI / gold-audit correction for CIs — see tech-doc lines 119–135).
5. **Document the `Q ≥ 3.0` frame exclusion.**

---

## (b) Per-method currency table (ALIGNED / DRIFTED / GAP) with 2024–2026 sources

| # | Method family (used/planned) | Verdict | Discriminating fact (dated) | Source URL |
|---|---|---|---|---|
| 1 | **Stratified sampling + floors/caps + proportional vs Neyman allocation** for the training pool | **ALIGNED** (minor refinement available) | Stratified designs for rare-event ML training pools remain standard in 2025; a 2025 *Canadian J. Statistics* paper refines stratified rare-event sampling to give **unbiased SEs** in multiple logistic regression — a refinement, not a replacement. Neyman (variance-optimal) allocation is still the textbook alternative to proportional; the repo's floors/caps are an ad-hoc coverage variant of the same idea. | https://onlinelibrary.wiley.com/doi/abs/10.1002/cjs.70008 |
| 2 | **Rare-class enrichment / case-control (King & Zeng 2001; Firth) + calibration-after-undersampling (Dal Pozzolo 2015)** | **ALIGNED** | King's rare-events logistic regression is still the reference; 2024–2025 work (HaiYing Wang, ICML; Haan-Ward 2025) confirms case-control/undersampling remains valid but **must be paired with weight/offset/prior correction** — exactly the "enriched-train + correct-later" stance the plan takes. Enriching to 35% then correcting the prior is the canonical move; nothing has superseded it. | https://gking.harvard.edu/files/0s.pdf · https://proceedings.mlr.press/v119/wang20a/wang20a.pdf |
| 3 | **Design weights / Horvitz–Thompson / post-stratification** to recover population quantities from a non-representative ML sample | **ALIGNED** (modern best practice = MRP, a strict superset) | HT + post-stratification remain valid. The 2024 best-practice framing is **MRP / model-assisted post-stratification** (Gelman et al.; embedded-MRP, *Statistics in Medicine* 2024), which **unifies** design weighting and small-area modeling and is the recommended tool when reporting many cells (state × NTEE × year). Plain HT weights are correct but lower-variance estimators exist. | https://onlinelibrary.wiley.com/doi/10.1002/sim.9956 · https://sites.stat.columbia.edu/gelman/research/unpublished/weight_regression.pdf |
| 4 | **QUANTIFICATION / prevalence estimation** — design currently names CC/ACC/PACC, SLD/EM (Saerens), ReadMe (King–Hopkins) | **DRIFTED + GAP** | (a) **DRIFT:** the 2024 SOTA layer the design omits is **KDEy** (kernel-density quantification, Moreo–González–del Coz, *Machine Learning* 2024) — beats distribution-matching and is competitive-to-superior vs EMQ — and the **distribution-matching family (DyS/HDy)**. (b) **Currency check:** EMQ/SLD remains a **top binary contender** — on standard LeQua **binary** benchmarks, per the KDEy 2024 study, **EMQ-BCTS attains best MAE and is statistically tied with KDEy-HD, DM-T, DM-HD** (this is reported on LeQua **2022 T1A** data inside the KDEy paper; I did **not** independently verify the official LeQua **2024** T1 ranking — see method note). Independently, Schumacher–Strohmaier–Lemmerich (*JMLR* 2025, 24-method benchmark) finds **no universal winner**, with the **best binary group = DyS/HDy, median-sweep/TSMax, Forman's mixture, Friedman's method** (their abstract does *not* name SLD/EMQ; EMQ's strength is evidenced by LeQua, not by this benchmark). (c) **GAP:** the *built* repo selects **no quantifier at all** — `grep` finds no quantification/ACC/SLD/PPI code in `src/`; it lives only in docs. ReadMe is a legacy baseline, not current. **2026 recommendation for binary text prevalence:** PCC on *target-calibrated* scores as the point estimate, **SLD/EMQ as the primary correction**, KDEy + DyS/HDy as robustness — benchmarked via **QuaPy**. | KDEy: https://link.springer.com/article/10.1007/s10994-024-06726-5 · LeQua 2024 proceedings (not yet fetched for official ranking): http://nmis.isti.cnr.it/sebastiani/Publications/LQ2024Proc.pdf · JMLR 2025: https://www.jmlr.org/papers/v26/21-0241.html · QuaPy: https://github.com/HLT-ISTI/QuaPy |
| 4b | **Quantification robustness to covariate/concept shift** (not just prior shift) | **GAP** | The project's own cited González–Moreo–Sebastiani (*DMKD* 2024) shows prior-shift quantifiers (ACC/PACC/SLD) **fail under covariate/concept shift**. A nonprofit population differs from the enriched train pool by *covariates* (sector mix, text length, era), not only the prior — so a pure prior-shift correction is insufficient. Mitigation in 2026: quantify/calibrate **by stratum** (per-NTEE) or use multicalibration (Linder et al. 2026, `facebookresearch/multicalibrated_llm_measurement`). | http://nmis.isti.cnr.it/sebastiani/Publications/DMKD2024a.pdf · https://github.com/facebookresearch/multicalibrated_llm_measurement |
| 5 | **Prediction-powered inference** — design names "PPI" (Angelopoulos 2023) | **DRIFTED** (refinement is now the default, and it is free) | Vanilla PPI (Science 2023) is **superseded for routine use by PPI++** (Angelopoulos–Duchi–Zrnic, arXiv **2311.01453**, v2 Mar-2024): adds a **power-tuning λ** that guarantees variance ≤ the labeled-only estimator (`λ=1` recovers PPI, `λ=0` recovers classical). It is **already the default in `ppi_py`**: `ppi_mean_ci(..., lam=None)` auto-estimates the optimal λ; the library also ships `ppi_distribution_label_shift_ci` (label-shift prevalence) and **cross-PPI**. 2025–2026 frontier = **stratified PPI** (Fisch et al., NeurIPS 2024 — pairs naturally with this repo's NTEE strata) and **active PPI / active statistical inference** (Zrnic–Candès 2024) for spending the gold-label budget where the model is uncertain. | PPI++: https://arxiv.org/abs/2311.01453 · ppi_py API: https://ppi-py.readthedocs.io/en/stable/ppi.html · Stratified PPI: https://arxiv.org/abs/2406.04291 · Active inference: https://arxiv.org/abs/2403.03208 |

**Verdict legend.** ALIGNED = current standard, no change needed. DRIFTED = method still valid but a newer refinement is now the default. GAP = the design lacks something the literature treats as required.

---

## Does the planned 4-part stack actually compose? (the key part-(c) question)

The plan stacks four things: **(1) HT design weights, (2) keep training enriched, (3) representative anchor at true prior, (4) quantification + PPI correction.** They are **mostly complementary, but (1) and (3) partially overlap** and the clean 2026 division of labor is:

- **Enriched silver pool → TRAINING ONLY.** Correct: imbalanced-learning best practice (rare-class enrichment + later prior correction). Do **not** compute population prevalence from this pool.
- **Representative anchor (at true prior) → the estimand backbone.** This is what makes prevalence claims valid. On the anchor you run either (i) **PPI++ / stratified-PPI** combining the small gold labels with model predictions on the population, **or** (ii) a **quantifier** (PCC + SLD/EMQ) on target-calibrated scores. **These two are alternative routes to the same target, not a pipeline to chain** — PPI++ gives CI-valid means using gold labels; quantification corrects label shift without gold labels. Recommended: run **PPI++ as the primary CI-valid estimator** (you have a gold audit set), and report the quantifier as a **methodological cross-check**.
- **HT design weights → only needed to recover population quantities *from the enriched pool itself*.** Once a representative anchor exists, that need **largely disappears** — the anchor *is* the representative sample. Weights are still worth fixing (cheap, and needed if anyone ever estimates anything on the silver pool directly, e.g. weighted training-loss or a weighted descriptive), but in the target design they are a **belt-and-suspenders** item, not the load-bearing one. Fixing `inclusion_prob` to the per-cell rate is correct and low-cost; just don't oversell it as "the" population-recovery mechanism.

**Bottom line:** the stack is the right *shape* for 2026, but it should be framed as **"enriched-train → representative anchor → PPI++ (primary) with quantification cross-check,"** with HT weights demoted to a correctness-hygiene fix rather than a co-equal pillar.

---

## (c) Ranked concrete updates

1. **Adopt PPI++ instead of vanilla PPI for the primary prevalence CI (DRIFTED, near-zero cost).** It is already the `ppi_py` default; just call `ppi_mean_ci` with `lam=None`. Prefer **stratified PPI** because the design is already NTEE-stratified. *Source: arXiv 2311.01453; ppi-py docs.*
2. **Select an explicit quantifier and wire it via QuaPy (closes the GAP).** Default = **PCC on target-calibrated scores + SLD/EMQ as the correction**; add **KDEy and DyS/HDy** as robustness. Don't ship "PCC only" — SLD/EMQ is among the empirically strongest binary methods on LeQua benchmarks, and DyS/HDy lead the JMLR 2025 benchmark. *Sources: QuaPy; KDEy 2024 (LeQua binary results); JMLR v26/21-0241.*
3. **Make quantification/calibration stratum-aware (covariate-shift GAP).** Estimate per-NTEE (or per key covariate cell) and aggregate, rather than one global prior correction — the repo's own DMKD-2024 cite shows global prior-shift methods break under covariate shift. *Source: DMKD 2024.*
4. **Fix `inclusion_prob` to the per-cell (stratum × pos/neg) rate (`sample.py:159-161`).** Correct and cheap, but frame it as design-weight *hygiene*, not the population-recovery mechanism (the anchor is that). *Source: HT/MRP best practice; Gelman MRPW.*
5. **Consider MRP / model-assisted post-stratification if reporting many cells** (state × NTEE × year). Strict superset of HT weighting, lower variance for thin cells. *Source: Statistics in Medicine 2024; Gelman MRPW.*
6. **(Optional, frontier) Active PPI / active statistical inference** to allocate the scarce human-label budget to high-uncertainty missions when expanding gold. *Source: Zrnic–Candès 2024.*

---

## UNRESOLVED — questions for the human

1. **★ Frame-vs-estimand mismatch (highest priority — no quantifier can fix this).** The silver pool excludes **LOW tier (`Q < 3.0`)** (`sample.py:95-96`) and LOW/bare-label missions are routed to the `apply_rule_label` rule layer at inference (`quality.py:602-633`, README.md:182). **Over what frame is the planned "representative test/prevalence sample at the true prior" drawn — ALL missions, or only `Q ≥ 3.0`?** If the anchor is `Q ≥ 3.0` but the prevalence *claim* is about all U.S. nonprofits, that is a **coverage gap** PPI/quantification cannot repair; the rule layer's accuracy on the excluded LOW stratum then directly biases the headline number. The anchor's frame must equal the claim's frame (or LOW must enter the anchor with its own rate). **This is the single most important thing to settle before building the prevalence step.**
2. **Is the headline estimand national prevalence, a time trend, or per-cell (state/NTEE/year) cells?** This decides PPI++ vs stratified-PPI vs MRP, and whether per-cell calibration is mandatory.
3. **Primary vs secondary estimator:** confirm the intended division of labor — **PPI++ on the gold anchor as the CI-valid primary**, quantifier (PCC+SLD/EMQ) as cross-check? Or quantifier-primary (relevant if the gold anchor is too small for tight PPI intervals — gold is only ~400, split three ways)?
4. **Is the gold/anchor sample large enough?** With gold ≈400 split into prompt_dev/validation/test (`config.py:166-168`, `sample.py:258-306`), the test slice (~175) may yield wide PPI/quantification CIs at the true (low) religious prior. Should a **dedicated representative prevalence sample** be sized separately from the boundary-enriched gold set (whose `inclusion_prob` is also a stratum-only rate, `sample.py:249-250`)?
5. **Covariate-shift severity:** is per-NTEE calibration sufficient, or is multicalibration over metadata (length/source/year, à la Linder 2026) warranted? Depends on how far the enriched train distribution sits from the target population on covariates other than the label.

---

### Method notes / confidence
- Foundational families (#1–3): light confirmation by design (low volatility — Cochran/Lohr/King lineage); single dated 2024–2025 source each; high confidence ALIGNED.
- Fast-movers (#4–5): deep search. PPI++-is-the-default and KDEy-vs-DM/EMQ were verified against **primary sources** (`ppi_py` API page; KDEy Springer/arXiv; JMLR 2025) rather than the repo's own docs, per the audit's independent-verification requirement. **Caveat (honest gap):** the official **LeQua 2024 T1** ranking was *not* independently verified — the `lequa2024.github.io` fetch returned no results table, and the "EMQ-BCTS best MAE, tied with KDEy-HD/DM" figure traces to the **KDEy 2024 paper's experiments on LeQua 2022 T1A data**, which predate the June-2024 competition. Treat "EMQ is a top binary contender" as supported by the KDEy study on LeQua benchmarks, not by a verified 2024 competition result. To close this, fetch `http://nmis.isti.cnr.it/sebastiani/Publications/LQ2024Proc.pdf`. This caveat does **not** change the DRIFTED+GAP verdict or any recommendation.
- The repo's own `docs/research/20260606-tech-calibration-quantification-prevalence.md` is current and high-quality and **already** anticipates most of this (per-stratum calibration, PPI, SLD sensitivity, DMKD-2024 covariate-shift warning). The net-new findings are: **PPI→PPI++ as the concrete default**, **KDEy/DyS as the missing modern quantifier layer**, the **built code selects no quantifier (docs-only)**, and the **frame/estimand mismatch (UNRESOLVED #1)**.
