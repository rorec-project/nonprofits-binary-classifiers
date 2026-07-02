# Independent Review — Harmonize Pipeline plan + current code

## Verdict
The plan is largely SOUND and its four load-bearing code claims are all TRUE against the source.
Methodology is current (PPI++ auto-lambda, Rogan-Gladen, design weights, EMQ cross-check). But it
carries three real methodological gaps that the audit also missed — the biggest is a **LOW-stratum
misspecification** that can move the headline 14% — and it over-scopes on figures/archiving. Two
must-verify validity assumptions (train/test disjointness; anchor sample_prob provenance) are not
addressed by plan or audit and should gate the paper.

Prioritized: (1) fix LOW stratum, (2) verify silver∩gold disjointness, (3) tie released-dataset
threshold to base-rate precision instead of enriched max-F1, (4) report per-org PPI both-ways
sensitivity, (5) trim figures/archive scope.

---

## CONFIRMED-SOUND
- PPI corpus-as-mean: `ppi_prevalence` weights only labeled anchor; `yhat_unlabeled` enters via its
  mean. Expanding the corpus by multiplicity (C-prev) correctly shifts the point estimate to per-org
  and needs no change to `ppi.py`. Verified.
- Guard-ordering bug (A1) real: `evaluate.py` writes anchor_oof/calibrator/rule_validation (141-161)
  before the G3 gate + `test_evaluation.exists()` refusal (166-171). Reorder is pure hardening.
- `_calibrator_payload` (evaluate.py 410-428) drops `pr_curve_points`/`achieved_*` though
  `pick_threshold` returns them (thresholds.py 98-106). B2a/B2b correct; B2b needed now.
- PPI++ auto-lambda, RG delta-variance, composite fixed-share variance: standard and correct.
- Plan's refusal to expand the labeled anchor residual (avoids independent-draws fallacy): correct.
- max-F1 threshold already persisted in calibrator payload; B1 requiring it is safe.

## MISSING-OR-RISKY (in priority order)
1. **LOW stratum mixes mechanisms but corrects with rule-only sens/spec (NOT flagged by audit/plan).**
   `_estimate_low` sets `p_obs = mean(pred_label)` over ALL tier==LOW (`estimate.py` 539-549) =
   50,470 rule rows + **59,704 `low_via_classifier`** rows. Router returns `low_via_classifier` with
   `rule_label=None` (router.py 53) so those are the classifier calibrated-threshold decision, not a
   rule. But Rogan-Gladen sens/spec come from `_rule_validation` (67 covered anchor LOW rows). So a
   ~54%-classifier `p_obs` is corrected by a pure-rule diagnostic. LOW is 20.7% of corpus and RG is
   18.36% vs observed 15.54% — this materially inflates the composite. Fix: split LOW into
   rule-labeled (RG) and classifier-labeled (route the 59.7k through PPI/calibrated like HM), or make
   rule_validation characterize the whole LOW mechanism. This is the single most important finding and
   the plan currently touches LOW only for tier-share counting and NTEE suppression.

2. **Train/test-anchor disjointness is not guaranteed by the code (paper-invalidating if it fails).**
   `sample.py` builds `build_silver_pool` (LLM-labeled training pool) and `build_gold_set`
   (anchor/frozen-test source) independently from the same HIGH+MEDIUM pool with the same seed and NO
   anti-join. Nothing here excludes gold/anchor EIN2 from silver. If they overlap, the classifier
   trained on silver has seen frozen-test/anchor texts, making PR-AUC 0.901 and PPI OOF residuals
   optimistic. MUST verify silver ∩ (gold ∪ anchor) = ∅ upstream; if not enforced, it is a leakage
   bug, not a documentation gap. This is the real substance behind the "circularity" question — note
   that eval/PPI are otherwise anchored to HUMAN labels, so LLM-silver-label bias in the classifier is
   first-order corrected by PPI; the residual risk is shared human/LLM construct bias (construct
   validity, not circularity).

3. **Released-dataset threshold (0.608) is enriched-max-F1, base-rate-optimistic — reopens Decision 3.**
   F1 is prevalence-dependent through precision, and Platt was fit on the ~50%-positive anchor, so
   `prob_calibrated` is calibrated to ~50% not ~14%. Both push 0.608 to look more precise than it will
   be on the population. The plan builds the exact machinery to fix this (B3 base-rate precision via
   prevalence-invariant TPR/FPR) but only reports it post-hoc; it does not USE it to SELECT the
   released threshold. More defensible: pick the released threshold = min t with base-rate
   precision(t, pi=0.14) >= target. This contradicts confirmed Decision 3 (max-F1) — surface
   explicitly so the user owns the call; do not footnote it.

4. **Per-org PPI estimand mismatch: labeled residual is per-unique-text, unlabeled is per-org.**
   After C-prev the unlabeled f-mean is per-org (correct) but the anchor residual E[Y-f] is estimated
   per-unique-text. For exact PPI unbiasedness under the per-org estimand the residual should be
   multiplicity-weighted. Likely negligible (residual is text-determined), but convert the worry into
   a measured number: report prevalence with anchor residual multiplicity-weighted vs not as a
   one-line sensitivity (each anchor row has a defined multiplicity — cheap).

5. **B3 base-rate precision needs a CI.** TPR/FPR from small high-threshold anchor cells; a headline
   "population-rate precision = X" without a bootstrap CI is under-powered. Plan emits point only.

6. **Duplicate-text CI non-independence (advisor Q a):** point estimate is fine; the CI barely moves
   because PPI's unlabeled-variance term is O(1/N), N~500k, dominated by the labeled O(1/n) term.
   Minor caveat: the plan's "LOW observed rate over raw EIN2 counts" inflates n_low and shrinks its
   binomial variance (a mild independent-draws fallacy) — immaterial because LOW variance is dominated
   by sens/spec CI, but keep unique-text n or note it.

7. **PR-curve figure is anchor-OOF, not the frozen-test 0.901 PR-AUC.** `pick_threshold` runs on
   anchor OOF probs/labels, so C2's curve and both operating points are anchor-derived. Label it as
   such; do not let the paper imply the figure's AUC equals the frozen-test 0.901.

8. **Reproducibility:** D1 manifest should pin the environment (uv.lock hash, torch/CUDA, the fp32
   override) — matters more than the `archive/<run_id>/` copy machinery.

9. **Per-NTEE:** suppression is anchor n>=10 only; some UNsuppressed cells still have huge CIs
   (Q 2.9-38%, S 0-27%). Consider flagging unstable unsuppressed cells by CI width too, not just
   suppressing fallbacks.

## SIMPLIFY-OR-DROP (over-engineering vs a CSS paper's actual claims)
- **8 new figures -> ~4 core + appendix.** KEEP C1 (real 103MB bug), C2 (PR frontier justifies low
  threshold), C3 (score dist / disagreement band = Decision-3 evidence), C5 (prevalence decomposition
  = ML-to-CSS bridge). DEMOTE to appendix/optional: C4 confusion, C6 rule forest, C7 quant
  sensitivity, C8 subgroup. C7 is valuable if referees probe estimator robustness; C4/C6/C8 are
  nice-to-have.
- **D1 archive/<run_id>/ copy machinery: SIMPLIFY** to one `run_manifest.json` beside outputs + git
  tag + env lock. Copying released artifacts into a run dir is optional.
- **D2 register stage 10 + relocate body into viz/render.py: SIMPLIFY** — adding "10" to
  `_STAGE_MODULES` is cheap and worth it; the full module relocation is optional polish.
- **scripts/regen_calibrator.py: KEEP** (genuinely needed now since test_evaluation.json exists and
  A1 blocks re-running stage 07) but treat as a one-off recovery script, not permanent surface.
- **evaluation/base_rate.py as its own module: KEEP** (small) — but it is the wrong altitude if it
  only reports and does not feed threshold selection (see MISSING #3).
- **B2a forward-fix, A1, C1: KEEP** — all cheap, load-bearing.
- **D3 opt-in canary: KEEP as documented/opt-in** (correct call).

## FACTUAL-CORRECTIONS to plan claims
- "dedup on LONGEST_MISSION" — load.py:117 dedups on `cfg.field` (configurable), not a hardcoded
  LONGEST_MISSION. A3's expand-back join key and Decision-1 must use `cfg.field`, else it breaks if
  field is CANONICAL_MISSION.
- A3 NaN/empty collision: real but conditional. Dedup collapses NaN to one group (pre-fillna);
  `_prepare_inference_frame` later `fillna("")`. Collision only bites if raw has BOTH genuine NaN AND
  genuine "" AND the sentinel == "". Asks: (a) sentinel != ""; (b) put `validate="many_to_one"` on the
  expand join (audit pseudocode has it; plan A3 omits it) so a collision hard-errors instead of
  silently double-counting into a released deliverable; (c) test the both-present case — the plan's
  "NaN rows get a non-null label" test does not catch a duplicated mapping key.
- Sequencing is otherwise sound (schema rev B1 before A3/C-prev/figures is correct; B2b before C2 is
  correct). No dependency-order errors found.
- `_existing_shard_matches` keys on `config_hash`+`threshold` via `_RESUME_METADATA_COLUMNS` (predict.py
  62-70, 420-436) — A2/B1's plan to add `threshold_precision` to that set is consistent with existing
  code.
