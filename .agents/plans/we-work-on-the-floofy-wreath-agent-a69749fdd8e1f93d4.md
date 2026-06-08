# Audit — Plan `we-work-on-the-floofy-wreath.md` (points 1–2)

Read-only review. Findings prioritized CRITICAL / IMPORTANT / MINOR. Line numbers refer to the plan unless noted.

## CRITICAL

1. **Human gold-coding process entirely unspecified (D).** The plan depends on human gold labels (≥85% gate line 99; "double-coded gold" line 99; `source_type='human'` slot line 93) but never says WHO codes the ~400 gold, with WHAT protocol/codebook, WHEN, or WHERE labels land. Verification step 6 ("eyeball a sample of labels + reasoning") is reviewing LLM output, not independent human coding. This blocks BOTH 2.2 (bake-off agreement needs human refs) and 2.4 (QC gate). It is the roadmap's only human-in-the-loop step. Fix: add a manual gold-coding sub-stage (coder identity, codebook, double-coding subset, where the human label column is written) before bake-off.

2. **586,718 vs 560,351 row-count not reconciled (F).** Upstream codebook `NonProfitData/docs/codebooks/corpus-missions.md:33` verifies `nrow = n_distinct(EIN2) = 586,718`. Plan (lines 7, 32) + annex (line 5) assert 560,351 "one per EIN2 (unique)" with no reconciliation. The 26,367-row gap is almost certainly the `COMMON_LEVEL1 == "501C3 CHARITY"` filter (audit lines 87–92), but the plan never states this. Fix: add one sentence — "560,351 = 501c3-charity subset of the 586,718-EIN2 missions corpus."

## IMPORTANT

3. **Single finalist vs majority-vote contradiction (B/D).** 2.2 (line 88) selects ONE prompt×model finalist; 2.4 (line 98) aggregates "prompt variants by majority vote"; crowd-kit/cleanlab/Dawid-Skene hooks presume multiple noisy labels per EIN2. With one finalist there is nothing to vote on and the WS-readiness rationale collapses. Fix: state whether 2.3 runs an ensemble (top-N prompts/models/seeds) or a single finalist, and make 2.4 consistent.

4. **4-way label → binary mapping undefined (D).** Schema (line 87) emits `binary_label ∈ {religious, nonreligious, ambiguous_review, insufficient_information}` AND a "derived binary rule from domains." Which governs? Line 93 says NaN=abstain but never maps `ambiguous_review`/`insufficient_information`. Fix: specify the source of truth (field vs domain-rule) and the abstain mapping for both non-binary verdicts.

5. **Join rate internally inconsistent (F/B).** Plan line 34 = 98.8%; annex line 5 headline = 99.4%. The annex contradicts itself: 99.4% (3,238 unmatched EINs) vs the parenthetical 6,805-row `?` bucket = 98.8%. Fix: reconcile to one number and state which denominator.

6. **Sprint ownership gaps (G).** `scripts/02_bakeoff_prompts.py` is only ever a "thin stub" (U1, line 150) — no agent builds the real bake-off harness (run prompt×model, agreement table, finalist), yet Verification step 4 runs it as a real script. Same void for the **canary-set definition** (line 94, unassigned) and the **LOW/bare-label inference rule layer** (lines 23/107, unassigned). Overlap: `04_quality_check.py` is both "thin stub" (U1) and "wired" (Q1); `run_pipeline.py` is both U1 and Orchestrator. Fix: assign bake-off harness, canary, and rule-layer to owners; de-duplicate 04/run_pipeline ownership.

7. **vLLM client vs serving (G).** S1 adds `vllm-client` (line 147) but line 167 runs `uv run vllm serve` — serving needs the full `vllm` package, not a client lib. Ad-hoc install on UCloud is outside `pyproject`/`uv.lock` → reproducibility hole for the production annotator. Fix: add full `vllm` (B200-pinned) to deps.

8. **Roadmap pt 3 F1 vs plan PR-AUC (B/E).** Roadmap line 61 wants "best validation F1" + per-epoch train/val loss + val F1 logged to a file; plan line 105 swaps to PR-AUC primary silently and the pt-3 stub doesn't commit to file-logging. Defensible for imbalance, but undeclared. Fix: call out the deviation, tie to R-06, and keep the file-logging requirement.

## MINOR

9. **Point-1 fine-tune model decision deferred (A).** Roadmap pt 1 asks for BOTH model decisions upfront; plan locks only the annotator, deferring the fine-tune grid to the pt-3 stub. Placement deviation, not omission. Fix: note it explicitly.

10. **R-issue coverage gaps (E).** Resolved: R-02 seeds, R-04 EIN2, R-05 class-weight, R-06 metric, R-07 deps, R-08 resume-by-content, R-09 paths, R-10 sample, R-12 .gitignore, R-13 dotenv, R-14 DATA_OF_CHOICE, R-15 README. **Not addressed:** R-11 (legacy moved "verbatim" → Windows/`/Users/caro` paths persist), R-16 (padding strategy never mentioned — a pt-3 detail), R-03 partial (per-epoch file logging not committed). Fix: state R-11/R-16 are acceptable-by-design if so.

11. **base-rate metrics mixing (B, minor).** Plan cites religious base rate ≈13% (line 34), lexicon hit rate 11.2% (annex 106), X group 7.27%. Different metrics; not contradictory but should be labelled to avoid confusion.

## Resolves cleanly (no action)
- `.agents/docs/` filenames, annex path, upstream parquet paths, `#references` anchor all resolve.
- No leftover "Appendix A" — fully migrated to "annex."
- Silver ~20k / gold ~400 / thin strata V257/Y200/U581 / bare-label ~6.8k≈6,827 consistent plan↔annex.
- Script names consistent across architecture / sprint / verification.

## Verdict
Internally mostly consistent and well-grounded, but NOT yet executable for points 1–2 without resolving the human gold-coding gap (#1) and the single-finalist-vs-vote contradiction (#3), which together break the QC gate and the WS store. The 586,718/560,351 and join-rate reconciliations (#2, #5) are data-integrity must-fixes. Sprint ownership voids (#6) and the vllm dep (#7) will surface at integration. Fixable with targeted edits, not a redesign.
