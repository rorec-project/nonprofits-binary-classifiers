# Sampling design review — religious-vs-not nonprofit classifier

Read-only analysis. Deliverable is the agent's final message; this file is durable scratch.

## PART A — Actual sampling design (file:line)

### Stratification
- Stratify on `ntee_major_group`, the 26 NTEE letter categories A–Z. `sample.py:99` restricts to `^[A-Z]$`.
- Frame = HIGH+MEDIUM only (Q ≥ 3.0). `sample.py:96`. LOW is excluded from silver/gold and routed to the rule layer at inference (README:182 "protects prevalence"). So FRAME ≠ POPULATION: any prevalence estimate from this pool is conditional on Q ≥ 3.0 unless the exclusion is itself weighted.

### Per-stratum allocation — proportional + floors + caps (NOT Neyman)
- `_allocate_stratum_targets` (`sample.py:486-540`): base = proportional (`pool_counts/total * total_target`, line 501).
- Caps applied first: B, P clamped to `_CAP_SIZE=2500` (lines 504-506, const `sample.py:42,44`).
- Floors: V, Y, U raised to `_FLOOR_SIZE=200`, clipped to available pool (lines 509-513, const `sample.py:41,43`).
- Residual redistributed proportionally among the non-floor/non-cap strata (lines 516-535).
- Allocation uses only stratum SIZE, not within-stratum variance → proportional, not optimal/Neyman. Floors/caps are ad hoc (fixed 200/2500), not variance- or cost-derived.

### Positive enrichment to ~35%
- `_TARGET_POSITIVE_SHARE=0.35` (`sample.py:45`). Per stratum, `n_pos_target = floor(n_target*0.35)` capped by availability (`sample.py:132-135`), backfilled if negatives short (138-145).
- "Positive" = `is_positive_enriched` = `has_religious_lexicon OR is_positive_rescue` (`sample.py:102-108`). This is a NOISY PROXY (lexicon hit or rescue), NOT 35% truly-religious. The 35% is a surfacing target for the labeler, not a true class prior.

### Q rubric + rescue (quality.py)
- `compute_quality_score` (`quality.py:425-482`): weighted sum of length, purpose verb, beneficiary, activity, specificity, clause count; minus boilerplate (-1.5) / vague (-2.0) / brevity (-1.0) penalties. Tiers HIGH≥5.0, MEDIUM 3.0–<5.0, LOW<3.0 (`config.py:152-154`).
- `is_positive_rescue` (`quality.py:558-577`): rescues top-of-MEDIUM rows (4.5 ≤ Q < 5.0) that have a STRONG-tradition lexicon hit AND a concrete purpose verb. Protects denominational positives that would otherwise sit below HIGH.

### Gold set — boundary-case diversity
- `build_gold_set` (`sample.py:169-255`): ~15/stratum, remainder to largest strata. Within stratum diversity quota: ~35% boundary, 25% clear pos, 25% clear neg, remainder other (`sample.py:229-244`).
- `_classify_boundary` (`sample.py:543-585`): saint-named-secular, spiritual-not-religious, generic-ministry, faith-heritage, boilerplate-religious. Rescue rows tagged "rescue" (line 206).

### split_human_sets
- `sample.py:258-306`: stratified prompt-dev draw (~50) by NTEE, remainder split 50/50 into validation (~175) and test (~175). Sizes from `config.py:166-168`.

### inclusion_prob — THE KEY FINDING (retained but unusable as a design weight)
- Computed `sample.py:160` (silver), `sample.py:249` (gold): `n_target / len(stratum_df)` — a SINGLE value for every row in the stratum.
- BUT positives and negatives are drawn at DIFFERENT rates within a stratum (132-156). True unit inclusion prob: positive = `n_pos_target/len(pos_df)`, negative = `n_neg_target/len(neg_df)`. These diverge exactly when enrichment does anything.
- So stored `inclusion_prob` = marginal stratum sampling FRACTION, not the unit's inclusion probability under the enriched design. Even if a downstream HT/prior-correction step consumed it, it would UNDER-CORRECT for the enrichment it is meant to undo.
- Gold (`sample.py:249`) is worse: boundary-enriched, so its single per-stratum value is even less interpretable.
- Downstream: `inclusion_prob` is written to manifests (`sample.py:459`) and only checked for PRESENCE (`scripts/01_build_sample.py:101`). NO code reads it back for Horvitz–Thompson, prior correction, or calibration. grep confirms: zero consumers. QC (`qc/agreement.py`) computes agreement/PR-AUC/MCC/kappa but no prevalence estimate and no design weighting.

### Design-notes connection
- `.agents/docs/20260605-replication-calibration-prevalence.md`: project ALREADY plans prevalence/quantification (QuaPy CC/ACC/PACC, Saerens EM/SLD, King-Hopkins ReadMe, PPI/ppi_py, freq-e, Dal-Pozzolo-style undersampling correction is implied). So population-share estimation IS an intended purpose → retained, CORRECT design weights matter.
- `.agents/docs/20260605-replication-weak-supervision-noisy-labels.md`: Snorkel/WRENCH weak-supervision + validation-first (Pangakis). The 35% enriched pool is the weak-supervision training pool; prevalence must come from a separate representative estimate, not from pool composition.

## PART B — Authoritative best practice (sources)
1. Neyman vs proportional allocation — Wikipedia/Caltech lecture/ScienceDirect. Neyman puts effort where stratum SD is large; collapses to proportional when SDs equal; needs prior SD estimates.
2. King & Zeng 2001, Logistic Regression in Rare Events Data (Political Analysis 9:137-163) — case-control/choice-based sampling (all positives + fraction of negatives) is efficient; correct the resulting bias via PRIOR CORRECTION (adjust intercept by known population prior τ and sample fraction) or WEIGHTING. SSRN + gking.harvard.edu. Statistical Horizons (Allison): the real issue is ABSOLUTE event COUNT, not the percentage; Firth penalized likelihood as alternative.
3. Dal Pozzolo et al. 2015, Calibrating Probability with Undersampling — undersampling/enrichment warps the posterior toward the minority class; closed-form correction p = (p0·β)/((p0·β)+(1−p0)), β=sampling rate. AUC/ranking unaffected; calibration must be corrected. THE bridge between "enrich for training" and "biased prevalence."
4. Horvitz–Thompson — Wikipedia, StatCan, Salganik Bit by Bit. Design weight = 1/inclusion prob; unbiased for any probability design; post-stratification improves precision with auxiliary group sizes.
5. Rare Event Prediction survey (arXiv 2309.11356) + Data Representativity (arXiv 2203.04706): over-representing rare class in training biases the model to over-predict it; resampling must NEVER touch validation/test; separate representative sample needed for prevalence.

## PART C — Assessment + ranked options
Aligned: stratify on NTEE; floors for thin strata; enrich rare positives for training; retain a column intended as a design weight; LOW kept via rule layer; small gold + bootstrap CIs.
Diverges: (1) inclusion_prob wrong granularity → unusable for HT/prior correction; (2) no separate representative prevalence sample / no calibration-correction step; (3) frame=Q≥3.0 ≠ population; (4) proportional not Neyman; (5) ad hoc floors/caps; (6) enrichment defined on noisy lexicon proxy.

Ranked options: (1) fix inclusion_prob to per-cell pos/neg rate [cheap, high value]; (2) carve a separate representative prevalence/calibration sample at true prior [medium, high]; (3) add prior-correction / Dal-Pozzolo / quantification step consuming the weights [medium]; (4) Neyman/variance-aware allocation [low-medium value]; (5) document the frame + Q≥3.0 exclusion weight [cheap].
