---
title: "How Many US Nonprofits Are Religious? Measuring a Population Share from Mission Text with LLM Weak Supervision and Prediction-Powered Inference"
author:
  - Jeanet Sinding Bentzen
  - Alessandro Pizzigolotto
date: "July 2026"
abstract: |
  We estimate that **14.4% of US nonprofit organizations have a religious mission (95% CI: 12.7–16.1%)**, measured over a cross-section of roughly 560,000 registered organizations that carry a mission-statement record. Recovering this share is not a classification problem but a _measurement_ problem: even a highly accurate document classifier is a biased estimator of a population proportion, because misclassification of a binary label is non-classical and does not average out. We therefore treat the classifier as an instrument to be de-biased rather than trusted. Our pipeline (i) labels a 20,000-record silver pool with an ensemble of large language models under human-audited quality gates, (ii) fine-tunes a DeBERTa-v3 encoder on the resulting soft labels, (iii) validates it against a frozen, human-coded test set that the model never sees during development, and (iv) corrects the full-corpus classifier counts into a population prevalence estimate using Prediction-Powered Inference (PPI++) anchored on a design-weighted, human-coded probability sample that includes the low-quality text the classifier cannot read. On the frozen test the classifier ranks well (ROC-AUC 0.949) and, at the population base rate, attains 91% precision at 77% recall — a usable but not spectacular instrument. We document every design decision, report a confidence interval on every quantity, and are explicit about where the evidence is thin: the frozen test is small (n=175) and enriched, the low-quality tier rests on a rule validated on 13 positive anchor rows, and one of our two quantification cross-checks (EMQ, 16.7%) sits above the primary interval. The result is a layered estimate rather than a single number: a firm 13.5% for the text-rich frame, a 14.4% full-frame point whose 12.7–16.1% interval is conditional on the low-tier rule, and a wider 11.5–18.4% envelope once that rule's uncertainty is propagated.
bibliography: references.bib
link-citations: true
---

# 1. Introduction

How many US nonprofits are religious? The question matters for the economics and sociology of the nonprofit sector — religious congregations and faith-based service providers occupy a distinctive regulatory, fiscal, and behavioral niche — yet it has no clean administrative answer. The IRS Business Master File records an NTEE code, but the "Religion-related" major group (X) captures only organizations that _self-classify_ as religious at registration; a faith-based hospital, a Catholic school, or a religiously motivated relief agency is typically coded under Health, Education, or International rather than under X. The organization's own description of what it does — its mission statement — carries the signal that the administrative code omits. Our object of study is therefore the *text*: we ask what share of organizations describe a religious mission, and we treat that share as a measured quantity with a stated frame and a confidence interval.

The natural temptation is to train a classifier on mission text, run it over the whole population, and count the positives. This is exactly the move that the text-as-data literature warns against. As @hopkinsking2010 put it, "even a method with a high percent of individual documents correctly classified can be hugely biased when estimating category proportions." The econometric statement of the same fact is sharper: misclassification of a _binary_ outcome is necessarily **non-classical** — the errors are mean-dependent because the variable is bounded in $\{0,1\}$ — so the bias does not wash out in aggregation and contaminates any downstream use of the counts [@meyermittag2017]. A classifier that is 90% accurate can still deliver a prevalence estimate that is wrong by several percentage points in a known direction. Classify-and-count is not a conservative shortcut; it is a biased estimator.

This paper is organized around that single methodological commitment: **the classifier is an instrument to be de-biased, not an oracle to be trusted.** We build the instrument as cheaply and as well as the current state of the art allows, we measure its error against human labels out of sample, and we then correct the population count using Prediction-Powered Inference (PPI++) [@angelopoulos2023ppi], a design-based estimator that uses a human-labeled probability subsample to remove the classifier's bias while retaining its statistical power. PPI++ has survey-sampling roots — it is a difference estimator — so it should feel familiar to an applied economist: the classifier's predictions on the unlabeled corpus buy precision, and the human labels on the anchor sample buy validity. Validity does _not_ require the classifier to be correct.

**Contribution.** We make three contributions, in decreasing order of generality. First, we give a fully documented, config-driven, human-gated recipe for turning cheap LLM labels into a *rigorous population prevalence estimate* — a template that transfers to any binary descriptive-share question over a text corpus. Second, we apply it to a substantive question and report the answer: religious mission prevalence among US nonprofits is 14.4% (95% CI 12.7–16.1% conditional on the low-tier rule; 11.5–18.4% once that rule's uncertainty is propagated, §7), concentrated overwhelmingly in NTEE group X (82%) but with a non-trivial tail across human-services, international-affairs, and community-improvement organizations. Third, and in the spirit of "validate, validate, validate" [@grimmerstewart2013], we conduct an evidence-first internal audit: we report the classifier's error at the true base rate rather than on the enriched evaluation set, we cross-check the prevalence estimate with a second quantifier, and we are explicit about the places where our evidence is thin. The paper is deliberately structured so that the substantive answer and its uncertainty live in the main text, while every implementation decision, threshold, and figure is documented in the appendices.

**Is the prediction "powerful"?** We answer this squarely in Section 6 and preview it here, because it is the question a skeptical reader asks first. On the frozen test the classifier *ranks* religious and non-religious missions almost perfectly (ROC-AUC 0.949). But ranking is not the deliverable. At the population base rate of roughly 11%, the honest characterization is: with a threshold tuned for 90% precision, the classifier recovers 77% of religious missions at 91% precision (false-positive rate 0.9%). That is a genuinely useful instrument — precise enough that PPI++'s correction is small — but it is not a solved problem, and its headline precision-recall AUC (0.901) clears our pre-registered acceptance bar (0.90) by only a hair. We neither oversell nor undersell it: the classifier is good enough to power a credible prevalence estimate on the high-quality frame, and the main residual uncertainty comes not from the classifier but from the low-quality text it was never designed to read.

**Framing and roadmap.** We are candid that this paper's centre of gravity is *methodological*: it is a measurement recipe and an unusually thorough internal validation of it, and the 14.4% is the recipe's first product rather than a fully developed substantive study of religious nonprofits. That said, the number is not merely illustrative. Because faith-based hospitals, schools, and relief agencies are routinely filed under secular NTEE codes, the administrative "Religion-related" category counts only a subset of organizations whose *stated mission* is religious; a mission-text measure recovers the difference, and quantifying that gap — how much larger religious presence is than the administrative category implies — is the substantive payoff the recipe unlocks. The rest of the paper proceeds as follows. Section 2 fixes the data, frame, and estimand. Section 3 lays out the measurement strategy. Sections 4 and 5 build the labeling ensemble and the classifier. Section 6 evaluates the instrument critically on the frozen test. Section 7 turns classifier counts into a population estimate with PPI++ and cross-checks it. Section 8 catalogs the limitations, and Section 9 concludes. The appendices carry the stage-by-stage technical reference, a description of every figure and table, the design-decision log, and a claim-to-artifact reproducibility map.

# 2. Data, Frame, and Estimand

**Source.** Mission text and organizational metadata come from a harmonized cross-section of the IRS/NCCS nonprofit universe (built by a sibling data project), keyed by a stable organization identifier, `EIN2`, that is carried through every downstream artifact so that the final estimate is auditable back to the source row. Each organization contributes its longest available mission field.

**Unit and frame.** The unit of analysis is the *organization*, not the document: several organizations share boilerplate mission text, and we count each organization once. The full prediction corpus after de-duplication and expansion is **560,354 organizations**. We partition this universe by a computable text-quality rubric $Q$ into HIGH, MEDIUM, and LOW tiers. HIGH+MEDIUM ($Q \geq 3.0$; 430,421 organizations, 76.8% of the frame) is text rich enough for an LLM or an encoder to read. LOW ($Q < 3.0$; 129,933 organizations, 23.2%) is bare labels and fragments — an organization name repeated, a truncated phrase — that no classifier can reliably score.

Two coverage boundaries define what our estimand does and does not speak to. First, the frame is organizations that carry a *mission-statement record*; organizations with no mission text at all are out of frame. These are plausibly non-random — very small, dormant, or recently registered organizations are the most likely to lack a mission field — so our $\theta$ is the religious share among mission-reporting nonprofits, not among the entire registered universe (§8). Second, within the frame the LOW tier is real population mass we must account for even though the classifier cannot read it; this is what the anchor sample and rule layer exist to handle (§7).

**Estimand.** Our target is the population share

$$\theta = \mathbb{E}\big[\mathbf{1}\{\text{mission is religious}\}\big],$$

where the expectation is taken over all `EIN2` organizations in the frame. We report $\theta$ for the HIGH+MEDIUM frame (where the measurement is strongest) and a composite $\theta$ over the *full* frame (HIGH+MEDIUM plus LOW), recombining the tiers by their organization-count shares. We are careful to keep these two estimands typographically and argumentatively distinct throughout, because the full-population number inherits the wide uncertainty of the LOW tier while the HIGH+MEDIUM number does not.

**A construct caveat, stated up front.** "Religious mission" is a property of *text*, a proxy for organizational identity, not a legal or doctrinal determination. A saint-named secular hospital, a "spiritual but not religious" wellness nonprofit, and a faith-heritage cultural society all sit near the decision boundary, and reasonable human coders disagree about them. Our labels speak to what the mission *says*, and every estimate below should be read as the prevalence of *religious mission language*, which we take to be the best available scalable measure of religious organizational identity but not identical to it.

# 3. Measurement Strategy

Our design is a weak-supervision pipeline [@ratner2017snorkel] wrapped in a de-biasing estimator. Figure 1 shows the ten-stage operational flow; the logic is a loop between cheap machine labels and a small, trusted human check, with four human gates (G1–G4) at which the pipeline halts until a person signs off.

**Figure 1. The ten-stage pipeline.** Arrows are data flow; bracketed `[HUMAN Gn]` marks a human gate (Appendix F).

```
  [01 Sample+Gold] --> [02 Bake-off+Slate] --> [03 Annotate full matrix]
        |  G1: code gold labels       |  G2: confirm slate
        v                             v
  [04 QC / Freeze] ------> [05 Anchor sample] ------> [06 Train sweep->refit]
   silver labels        |  G4: code anchor          |  G3: unlock frozen test
        v               v                           v
  [07 Evaluate] --> [08 Infer] --> [09 Prevalence] --> [10 Visualize]
   frozen test      full corpus     PPI++ + EMQ         figures
```

1. **Silver labels from an LLM ensemble.** We draw a stratified, positive-enriched pool of
20,000 HIGH+MEDIUM organizations and label each with an ensemble of large language models under a versioned codebook prompt. These are *silver* labels: abundant and cheap, but noisy. Aggregated by majority vote, they are the training fuel.

2. **Gold labels from a human.** A 450-record *gold* set is hand-coded by a human. The gold
is deliberately enriched with boundary cases and is used for three jobs and three jobs only: choosing which LLMs to trust (Section 4), gating label quality, and — for one frozen slice — final evaluation (Section 6). The gold is *never* in the training pool.

3. **A fine-tuned encoder.** We fine-tune a DeBERTa-v3 encoder [@he2021deberta] on the
silver labels (Section 5). We deliberately do *not* use the LLM ensemble itself as the production classifier: on taxonomic labeling tasks, fine-tuned encoders remain competitive with or better than zero-shot LLMs at a fraction of the inference cost [@ziems2024], and an encoder gives us calibrated probabilities we control.

4. **A de-biasing anchor.** A separate 500-record *anchor* sample is drawn from the *full*
frame — crucially including LOW-quality rows — with recorded inclusion probabilities. Hand-coded, it is the human-labeled probability subsample that PPI++ needs to correct the classifier's population count (Section 7).

Two disciplines hold the design together. **The model never sees the final exam:** the test split is frozen at sampling time and unlocked only by an explicit human sign-off that records the exact model checkpoint hash, so no leakage can occur during model iteration. **Cheap labels are always audited against expensive ones:** every LLM decision that affects the pipeline passes through a human-coded gate before it propagates.

# 4. Weak-Supervision Labeling

**Choosing the annotators (the bake-off).** We do not assume which LLM labels best; we measure it. Every candidate model x prompt combination is scored against the human-coded `prompt_dev` split on two metrics chosen for an imbalanced task: Cohen's $\kappa$, which is chance-corrected and so cannot be inflated by an always-say-majority baseline, and the bootstrap lower bound of minority-class F1, which guards against prompts that ace the negatives but miss the rare positives. A candidate is admitted only if it clears *both* $\kappa \geq 0.70$ and minority-F1 CI-lower $\geq 0.70$. Of 15 candidate arms, 12 clear the gate; the three failures — all `gpt-4o-mini` and `gemma` prompt variants — fail specifically on the minority-F1 floor (CI-lower 0.63–0.67) despite acceptable $\kappa$, exactly the failure mode the second gate is designed to catch. The strongest arms are `DeepSeek-V4-Flash` (v2: $\kappa = 1.0$, minority-F1 = 1.0) and `gpt-4o-mini` (v2: $\kappa = 1.0$). Full results are in Appendix C, Table C1, and Figure B14. A recurring pattern visible there is an *accuracy–abstention trade-off*: several top-$\kappa$ arms achieve their scores partly by abstaining on 26–36% of items, which the bake-off surfaces explicitly rather than hiding.

**The production slate.** A human confirms the slate (gate G2). For the production run we committed `DeepSeek-V4-Flash` under all three prompt templates (v1, v2, v3) and aggregated by majority vote across the three, trading a single point estimate for the noise-averaging of a three-vote ensemble. The choice of an open-weight model served locally via vLLM [@kwon2023vllm], rather than a closed API model, was deliberate: it makes the labeling step reproducible and cost-controlled, and the bake-off showed it to be the single best annotator.

**Freezing the silver labels.** The full 20,000-record pool is annotated and aggregated by majority vote, then QC-gated against the human-coded `validation` split before any label is written (gate G4 on labels). The resulting `silver_labels.csv` holds **19,891 organizations** (gold `EIN2`s are excluded by construction to prevent train/test contamination). Label consensus is high: mean pairwise inter-annotator agreement is 100%, the all-abstain rate is 4%, ties are 0%, and 17,540 of the items receive a unanimous three-vote consensus. Of the 19,122 rows that receive a label, **30.4% are religious** — well above the population base rate, by design, because the pool was positive-enriched to roughly 35% following the rare-events sampling logic of @kingzeng2001. This enrichment is a training convenience, not a population statement; it is precisely why we cannot classify-and-count and must de-bias.

# 5. The Classifier

**Architecture and arms.** We fine-tune DeBERTa-v3-base [@he2021deberta] as the primary encoder, chosen for the disentangled-attention design that gave it state-of-the-art classification performance at release, and we benchmark it against ModernBERT-base [@warner2024modernbert] and TF-IDF / MiniLM logistic-regression baselines [@reimers2019sbert]. Training minimizes a soft cross-entropy on the ensemble vote-shares rather than hard 0/1 labels. Soft targets are the default because they carry annotator-confidence information and are inherently more robust to label noise than hard thresholds, which makes techniques like label smoothing redundant and down-weights exactly the disagreement-band rows that a confident-learning prune [@northcutt2021confident] would drop.

**Selection.** We run a selection sweep over encoder x training-arm cells across three seeds (42–44), then refit the winner across five seeds (42–46) for honest variance reporting. Selection is on mean validation PR-AUC [@davisgoadrich2006] with a *parsimony tie-rule*: a challenger arm displaces the incumbent only if its seed-mean advantage exceeds the larger of the two arms' across-seed standard deviations; otherwise the simpler arm wins. The tie-rule is applied as a *sequential demotion*: starting from the top-ranked arm by mean PR-AUC, each simpler challenger replaces the incumbent when it falls within the larger of the two arms' across-seed standard deviations. On our sweep (Appendix C, Table C2) this chain runs `hard` (mean PR-AUC 0.958 ± 0.006) → `class_weighted` (0.953, gap 0.005 < SD) → soft `default` (0.941 ± 0.012, gap 0.012 <= SD), landing on the soft `default` DeBERTa arm (minority-F1 0.835, seed 44). **We flag this as the sharpest weak spot in our selection and state it plainly:** the transitivity of the chain matters, because `hard` beats the arm we ship *head-to-head* — 0.958 versus 0.941, a gap of 0.017 that exceeds the larger standard deviation. We therefore do not claim the shipped arm wins on validation PR-AUC; it does not. We ship it on principled grounds: soft vote-share targets are the more defensible objective under label noise (they down-weight the annotator-disagreement band rather than hard-thresholding it), and in an enriched-pool setting we weight calibration and noise-robustness above a 1.7-point validation edge that in any case does not survive to the frozen test. A reader who prioritizes the point estimate would select `hard`, and we would not contest that; a cleaner future version of the rule would compare every arm against the *simplest admissible* arm directly rather than transitively. A sharper version of the same objection targets the encoder itself: a linear TF-IDF logistic-regression baseline reaches validation PR-AUC ~0.938, statistically indistinguishable from DeBERTa's ~0.941 (Figure B12), so strict parsimony would ship the baseline. We retain the encoder on two grounds that validation PR-AUC — measured on in-distribution, silver-derived data — cannot see: a bag-of-words baseline keys on exact tokens and is brittle to paraphrase and to the vocabulary drift we expect between the enriched training pool and the full corpus, whereas the encoder generalizes semantically; and the pipeline is designed to be re-tasked to other sectors where lexical baselines transfer worst. We concede this is an argument from expected out-of-distribution robustness rather than a demonstrated frozen-test win (we evaluated only the selected encoder on the locked test), and that a TF-IDF baseline is a reasonable, cheaper alternative a replication could adopt.

**Calibration.** The classifier's raw scores are calibrated on the anchor sample by five-fold cross-fit Platt scaling [@platt1999] (fitted parameters $a=0.417$, $b=0.841$). Platt is retained over temperature scaling because its intercept absorbs the prior-shift offset introduced by training on the enriched pool; isotonic regression is excluded because it overfits at the anchor's size ($n \approx 500$). Calibration is excellent: expected calibration error (ECE) on held-out anchor scores is **0.007** [@guo2017calibration], with Brier 0.030 and log-loss 0.139. The reliability diagram (Figure B2) shows the scores are sharply bimodal — most mass sits at probability $\approx 0$ or $\approx 1$ — so the classifier is confident and, where confident, correct.

**Are the labels enough?** Two distinct sample sizes matter, and it is worth separating them because they bind in opposite ways. On the *training* side, the encoder is fine-tuned on the 19,122 labeled silver rows (5,807 religious, 13,315 non-religious; a further 769 of the 19,891-row pool abstain and carry no label, so they never enter training). For a base-size encoder on a binary text task this is comfortably in the sufficient regime, and three independent pieces of evidence say so: in-training evaluation PR-AUC on the silver holdout plateaus near 0.99 within two epochs (early stopping halts well before the 10-epoch budget); a linear TF-IDF logistic-regression baseline reaches validation PR-AUC ~0.94, essentially matching the transformers (Figure B12), which means the discriminative signal is strong, largely lexical, and already saturated at this data volume rather than starved for it; and the five-seed refit is stable (validation PR-AUC SD ~0.01). The gap that remains — silver-holdout PR-AUC ~0.99 versus human-labeled test PR-AUC ~0.90 — is therefore not a quantity problem but a *label-quality* problem: it reflects LLM-annotation noise and the enriched-pool distribution shift, neither of which more silver of the same kind would fix. The honest caveat is that we did not run a multi-fraction learning-curve ablation (only the full-data point; §B12), so silver sufficiency is argued from convergence evidence rather than demonstrated by an explicit plateau. On the *evaluation and de-biasing* side, by contrast, the binding constraint is the human budget: 450 gold and 500 anchor codings, of which only 36 religious organizations fall in the HIGH+MEDIUM anchor cells. This is what drives the width of the frozen-test CIs (§6) and of the prevalence interval — and it is precisely why PPI++ matters, since it extracts a threefold efficiency gain from that scarce anchor rather than demanding we code thousands more rows by hand (§7). If we could buy one more unit of human labeling, it should go to the anchor and the frozen test, not to more silver.

# 6. Evaluation on the Frozen Test

We now ask, critically, how good the instrument is. The frozen test is a human-coded, NTEE-stratified slice of the gold set, unlocked exactly once after model selection. It holds **175 organizations (77 religious, 98 non-religious)**. All three pre-registered acceptance criteria pass (Table 1).

**Table 1. Frozen-test acceptance (all pass).**

| Criterion | Threshold | Observed | Pass |
|---|---|---|---|
| PR-AUC | $\geq 0.90$ | **0.901** | **pass** (by 0.001) |
| Minority-F1 CI lower bound | $\geq 0.70$ | **0.840** | pass |
| ECE (anchor OOF) | $\leq 0.05$ | **0.007** | pass |

The headline classification metrics at the recall-first operating threshold (0.058) are: minority-class precision 0.817, recall 0.987, F1 0.894 (95% CI 0.840–0.938), MCC 0.809, Cohen's $\kappa$ 0.796, balanced accuracy 0.907, ROC-AUC 0.949. The confusion matrix is TN 81 / FP 17 / FN 1 / TP 76: at this threshold the model misses essentially no religious missions (one false negative of 77) at the cost of 17 false positives. Figure B3 shows the confusion matrices at all three operating points, and Figure B1 the PR and ROC curves.

**Why the headline number is not the population number.** Two facts must be held together. First, PR-AUC 0.901 is computed on a set that is **44% positive by construction** — the gold was enriched with hard, boundary, and positive-looking cases to stress-test coders and prompts. PR-AUC is prevalence-dependent, so 0.901 on a 44%-positive set is *not* the performance one would see at the 11% population base rate. Second, the acceptance margin is razor-thin (0.90139 vs. 0.90), so we treat the pass as real but not robust.

**The fair, population-level characterization.** The honest way to report classifier quality for a prevalence task is at the *population base rate*. We estimate that base rate at 10.9% (from the design-weighted anchor) and sweep the threshold to find the operating point that achieves 90% precision there. The result (Figure B5, and `base_rate_precision.json`): at threshold 0.094 the classifier reaches **precision 0.912 (95% CI 0.826–0.990) at recall 0.769**, with a false-positive rate of 0.9%. This is the number we stand behind as the description of the instrument: at realistic prevalence, the classifier finds roughly three-quarters of religious missions while keeping nine in ten of its positive calls correct. It costs about 23% of true positives to buy that precision — a real limitation, mitigated for the prevalence deliverable by the fact that PPI++ corrects for exactly this kind of systematic miss.

**Subgroups.** Performance is broadly stable across text length and sector (Figure B10, Appendix C Table C3). Minority-F1 by word-count bin is 1.00 (0–10 words, n=12), 0.84 (11–25, n=45), 0.93 (26–50, n=59), and 0.90 (51+, n=59); the shortest texts are easy because they are usually unambiguous, and the middle band is hardest. Across NTEE major groups most cells score $\geq 0.8$, with unstable exceptions in cells of three-to-five organizations (e.g., group G, n=3, F1 0.0) that we report as diagnostics, not claims. We do not over-read subgroup numbers on a 175-row test.

# 7. Population Prevalence via Prediction-Powered Inference

**Estimator.** For the HIGH+MEDIUM frame we use PPI++ [@angelopoulos2023ppi]. Intuitively, PPI++ starts from the classifier's mean prediction over the entire unlabeled corpus (a low-variance but biased quantity) and subtracts a bias-correction term equal to the classifier's mean error on the human-coded anchor sample (an unbiased but higher-variance quantity). The result is asymptotically unbiased *regardless of whether the classifier is correct*, with valid confidence intervals, and is never less efficient than using the anchor labels alone. Anchor rows carry Horvitz–Thompson design weights [@horvitzthompson1952] derived from their recorded inclusion probabilities, so the correction targets the population rather than the (stratified) anchor draw. The unlabeled corpus is expanded by organization multiplicity so the estimand is per-organization.

**Headline result.** For the HIGH+MEDIUM frame, the design-weighted PPI++ estimate of religious prevalence is **13.5% (95% CI: 11.7–15.4%)**; the unweighted variant is 13.0% (11.1–15.0%). Folding in the LOW tier (below) gives the full-frame composite of **14.4% (95% CI: 12.7–16.1%)**.

**What the classifier buys: PPI++ versus gold labels alone.** The value of the classifier is best seen by comparing PPI++ against the estimator that uses *only* the human labels — the classical, design-weighted survey estimate from the anchor sample, which needs no classifier and is what a purely manual audit would report. On the HIGH+MEDIUM frame the classical estimator gives **10.8% with a 95% CI of [7.8%, 14.4%]** — a half-width of ±3.3 percentage points, resting on just 36 religious organizations among the 351 anchor rows in that frame. PPI++ tightens this to ±1.9 percentage points, a **3.1× reduction in variance** and a 1.8× narrower interval; achieving the same precision with human labels alone would require roughly **1,100 anchor codings instead of 351**. This is the quantitative payoff of the design, and the reason the method exists: the classifier's predictions over the half-million-organization corpus contribute the statistical power, while the anchor labels contribute the validity. The two estimators do not perfectly agree, and we flag the difference rather than smooth it over. PPI++ places the HIGH+MEDIUM estimate about **2.7 percentage points above** the raw design-weighted anchor labels, and the label-only point (10.8%) sits just *below* the PPI++ lower bound of 11.7%. The same pattern holds at the headline: the design-weighted anchor across all 500 rows implies about **11.9%**, whereas our composite reports 14.4% — a ~2.5-point upward shift that the classifier's corpus signal and the LOW-tier Rogan–Gladen correction add on top of the hand labels. This shift is well within the sampling error of an anchor that contains only 36 religious organizations in the HIGH+MEDIUM cells (59 across the full frame), so it is not evidence of a broken correction; but it does mean the headline **leans on the classifier rather than resting on the human labels alone**, and the honest remedy is a larger anchor, not a different estimator. Readers who trust only the hand labels should therefore anchor on the wider label-only interval and read the PPI++ number as the model-assisted refinement of it. (Methodology: the classical CI is a design-weighted percentile bootstrap over the anchor HIGH+MEDIUM rows, 2,000 resamples, matching the resampling used for every other interval in this paper.)

**The LOW tier and its honest bounds.** The 130,000 LOW-quality organizations cannot be scored by the encoder, so the anchor's 149 LOW rows do double duty. Some LOW rows carry enough signal to route through the classifier (`low_via_classifier`, 47% of LOW mass, estimated by PPI at 14.8% [10.2–19.3%]); the rest are handled by a deterministic religious-lexicon rule and corrected for the rule's measured error via the Rogan–Gladen adjustment [@rogangladen1978] (53% of LOW mass, estimated at 19.4% [14.0–24.8%]). The rule is validated on the anchor's LOW cells (Figure B8): **specificity 1.00** (95% CI 0.93–1.00, n=54 — it produces no false positives) but **sensitivity 0.846** (95% CI 0.58–0.96) — and that recall is estimated on just **13 positive rows**. Because the Rogan–Gladen correction divides by (sensitivity + specificity - 1), a recall this uncertain propagates into a wide LOW-tier band: under systematic rule misclassification the LOW prevalence ranges from 11.0% to 28.4%. This changes how the headline should be read, and we are explicit about it. Our composite interval of 12.7–16.1% is a *sampling* interval: it propagates the estimators' sampling variance but conditions on the rule's point-estimated sensitivity. If we instead carry the full systematic LOW-tier band into the composite — combining the HIGH+MEDIUM sampling CI with the LOW systematic range at the estimated tier shares — the full-frame estimate widens to roughly **11.5–18.4%**. We therefore report a layered result rather than a single interval: the HIGH+MEDIUM number (13.5%, 95% CI 11.7–15.4%) is our firm result and does not depend on the rule at all; the composite point of 14.4% is our best full-frame estimate; its 12.7–16.1% interval is valid *conditional on* the rule's estimated sensitivity; and 11.5–18.4% is the honest outer envelope once the rule's own uncertainty — estimated, recall, on just 13 positive anchor rows — is allowed to propagate. A reader who wants a single defensible full-population statement should use the wider envelope; a reader focused on the text-rich frame should use the HIGH+MEDIUM number.

**Face validity by sector.** The per-NTEE prevalence forest (Figure B7, Appendix C Table C4) is a strong sanity check. Religious prevalence is overwhelmingly concentrated in NTEE group X, Religion-related, at **82.2% (95% CI 77.3–87.2%)**, with a secondary presence in Human Services (P, 19.4%), International/Foreign Affairs (Q, 17.4%, wide CI), Crime/Legal (I, 12.3%), and Community Improvement (S, 11.6%), and near-zero prevalence in Environment and Animal-related sectors (C, D $< 0.1\%$). Nine of 27 groups are suppressed for having fewer than ten anchor rows; their fallback point estimates are unstable and we label them not-estimated rather than report them. A battery of vocabulary diagnostics on the silver text (Figure B11 and Figures B15–B19) confirms the model separates on face-valid religious language rather than artifacts. The distinctively religious terms are theological and denominational (*christian, jesus christ, the gospel, faith based, word of god*, plus *jewish* and *catholic*), while the secular pole is civic, scientific, and service vocabulary (*public, research, county, low income, quality of life*). Tellingly, the two classes share a large generic-nonprofit vocabulary (*community, education, health, children*), so raw word frequencies barely separate them (Figure B19); it is the distinctiveness-weighted view — Fightin' Words log-odds z-scores in the sense of @monroe2008fightin (Figures B15–B18) — that exposes the clean religious/secular axis. That contrast is a useful caution for text-as-data measurement generally: the signal the classifier exploits is real but invisible to a naive word count.

**Cross-checks and one disagreement we do not hide.** We cross-check PPI++ with expectation- maximization quantification (EMQ/SLD) [@saerens2002]. On the HIGH+MEDIUM frame EMQ returns **16.7%**, which sits *above* the PPI++ upper confidence bound of 15.4% (Figure B9). This is a ~3-percentage-point disagreement between two reasonable quantifiers, and an evidence-first paper must name it rather than bury it. We read it as a caution, not a contradiction: EMQ assumes the classifier's class-conditional score distributions are stable between the labeled and unlabeled sets, an assumption strained by our enriched training pool, whereas PPI++'s validity does not depend on it. We therefore retain PPI++ as primary and report EMQ as a sensitivity bound indicating that the true share could plausibly sit at the upper end of, or slightly above, our stated interval.

# 8. Limitations

We organize the weak spots by where they bite, in rough order of how much they should move a reader's beliefs. This section is deliberately long; an evidence-first paper earns trust by cataloging its own soft spots.

**Sampling frame and representativeness.** The labeled frame is HIGH+MEDIUM text only; the full-population number rests on folding LOW back in through the anchor and rule layer, which is the single largest source of uncertainty (§7). The training and gold pools are positive-enriched to ~35% and ~40% respectively, so no raw count from them is a population statement — the entire de-biasing apparatus exists to undo this, and its correctness is load-bearing. Separately, the estimand is defined over the mission-reporting frame (§2): organizations with no mission record are excluded, and because they are likely smaller and less active than average, our share should be read as prevalence among mission-reporting nonprofits rather than among all registered organizations.

**Evaluation validity.** The frozen test is small (n=175, 77 positives) and enriched, so its minority-F1 CI is wide and its PR-AUC clears the acceptance bar by only 0.001. The operating threshold is very low (0.058), a recall-first point that accepts 17 false positives to avoid one false negative; we mitigate by shipping three labels (recall-first, max-F1, base-rate) so downstream users can choose their own trade-off. Calibration is excellent in aggregate but the mid-probability reliability bins are sparsely populated, so local miscalibration between 0.1 and 0.9 is weakly estimated.

**Label quality and weak supervision.** Production aggregation is majority vote; more elaborate label models (Dawid–Skene [@dawidskene1979], CROWDLAB) are run only as diagnostics and did not change the labels. LLM annotator reproducibility is best-effort: model identifiers are pinned where possible, but closed-model behavior can drift. The gold coding template captures only a binary label, not the richer multi-label codebook, so future sensitivity analyses on ambiguous "faith-inspired" cases would require re-coding.

**Prevalence estimation.** The EMQ cross-check (16.7%) exceeds the PPI++ interval (§7). The LOW-tier rule sensitivity is estimated on 13 positives, producing the wide composite band. Per-NTEE small-cell estimates are diagnostics, not publishable subgroup claims.

**Modeling choices.** The parsimony tie-rule selected the soft `default` DeBERTa arm over the higher-point-estimate `hard` arm (§5). The encoder grid was pruned to two transformers plus baselines; several standard robustness knobs (data augmentation, focal loss, decision-curve analysis) were intentionally skipped under a "one principled primary method plus minimal robustness" policy, each documented with a rationale in Appendix D.

**Construct validity.** The strongest still-open threat is **lexicon circularity**: the same religious lexicon that positively enriches the sampling pool also drives the LOW-tier rule and shapes which boundary cases reach the gold set. This risks inflating apparent accuracy on easy lexicon-hit cases and under-sampling hard lexicon-miss positives (saint-named secular organizations, faith-heritage framings). We recommend, and have not yet fully executed, reporting lexicon-hit versus lexicon-miss test slices separately. Relatedly, identical mission texts inherit a single classification rather than being scored independently, and the label measures text, not verified organizational identity (§2). Finally, the vocabulary diagnostics (Figure B19) surface two cosmetic preprocessing residues — an HTML non-breaking-space token (*nbsp*) and IRS/501(c) filing boilerplate (*internal revenue code*, *section 501*) appearing among high-frequency n-grams; these do not affect the classifier or any reported estimate, but should be stripped before the diagnostic figures are released.

**Reproducibility and infrastructure.** Heavy artifacts are versioned by cloud symlinks standing in for DVC; GPU stages are coupled to a specific HPC platform. The run manifest for this build was stamped mid-run (before the per-organization prediction artifact was written), so two of its provenance fields read "missing"; we verified against file timestamps that the 560,354-organization artifact the prevalence stage consumed is authoritative and the manifest fields are stale, not contradictory (Appendix E). A multi-GPU precision bug that had polluted the training log with degenerate runs was fixed and the "documentation curve" figure regenerated (Appendix B, Figure B12); the residual limitation there is that we ran only the full-data point, so we present a per-encoder snapshot rather than a true learning curve.

# 9. Conclusion

We set out to answer a descriptive question — what share of US nonprofits are religious — and to answer it *rigorously*, treating the classifier as an instrument whose error must be measured and removed rather than trusted. Our answer is layered by how much we ask of the low-quality text: a firm **13.5% over the high-quality frame (95% CI 11.7–15.4%)**, a **14.4% full-frame point** whose 12.7–16.1% interval is conditional on the low-tier rule, and a wider **11.5–18.4% envelope** once that rule's uncertainty propagates — concentrated in but not confined to the officially religious NTEE group. The classifier that powers this estimate is good — near-perfect ranking, 91% precision at 77% recall at the true base rate, and calibration error under 1% — but not miraculous, and we have been explicit that the remaining uncertainty lives mostly in the low-quality text tail and in a quantification cross-check that reads slightly high. The broader deliverable is the recipe: a documented, human-gated, evidence-first path from cheap LLM labels to a defensible population share with a confidence interval, portable to any binary prevalence question over text. The immediate next steps that would most strengthen the estimate are a larger frozen test, a lexicon-hit / lexicon-miss decomposition to close the circularity gap, and a larger LOW-tier anchor to tighten the full-population bound.

\newpage

# Appendix A. Reproducibility and Artifact Map

Every headline number in this paper is backed by a committed artifact; this appendix maps claims to files so the paper is auditable. Key artifacts, all under `data/processed/`:

- `evaluation/test_evaluation.json` — frozen-test metrics, confusion matrices at all three
thresholds, PR/ROC points, subgroups, calibration on anchor OOF.
- `evaluation/calibrator.json` — fitted Platt parameters, thresholds, achieved precision/recall.
- `evaluation/base_rate_precision.json` — the population-base-rate operating point (Table 1, §6).
- `evaluation/rule_validation.json` — LOW-tier rule sensitivity/specificity with Wilson CIs.
- `prevalence/prevalence_report.json` — PPI++ primary, LOW decomposition, EMQ cross-check,
composite, tier shares, and per-organization counts.
- `prevalence/prevalence_by_ntee.csv` — per-sector prevalence (Table C4).
- `predictions/predictions_full.parquet` — 560,354 per-organization predictions (three labels
  + calibrated probability).
- `../models/selection_report.json` — the encoder-selection sweep and tie-rule verdicts.
- `run_manifest.json` — git SHA, config hash, environment lock, input row counts.

Global determinism knobs: `SEED=42`, 2,000 bootstrap resamples for every CI. Runs are numerically stable given fixed inputs but not byte-identical across hardware. The build reported here corresponds to git SHA `3f19a0b` (config hash `bc8ff90…`), Python 3.13.12, run on an NVIDIA B200 (UCloud SDU/DeiC). See §Appendix E for a provenance reconciliation.

# Appendix B. Figure-by-Figure Reference

All figures render as PNG/SVG/PDF triplets in `data/processed/figures/`. Numbers below are read from the rendered figures and cross-checked against the source JSON.

![](../data/processed/figures/precision_recall_curve.svg)

**Figure B1 — `precision_recall_curve`.** PR curve (main) with inset ROC. The PR curve holds a 0.90–0.94 precision plateau across recall 0.2–0.9 before collapsing near recall 1.0; the three candidate thresholds (operating, base-rate, max-F1) cluster on the high-recall shoulder. Inset ROC hugs the top-left corner. *Reads:* PR-AUC 0.901, ROC-AUC 0.949. *Conclusion:* strong ranking with a stable high-precision operating region.

![](../data/processed/figures/reliability_diagram.svg)

**Figure B2 — `reliability_diagram`.** Binned calibration scatter (marker size proportional to bin count) against the 45° identity line; annotated ECE = 0.007. One dominant bubble at (0.03, 0.03) — most organizations are confidently, correctly negative — with the high-score bins landing on the diagonal. *Conclusion:* excellent calibration; scores are bimodal.

![](../data/processed/figures/frozen_test_confusion_matrix_operating.svg)

![](../data/processed/figures/frozen_test_confusion_matrix_max_f1.svg)

![](../data/processed/figures/frozen_test_confusion_matrix_base_rate.svg)

**Figure B3 — `frozen_test_confusion_matrix_{operating,max_f1,base_rate}`.** Three 2x2 heatmaps, now emitted one per threshold policy (the earlier combined three-panel `frozen_test_confusion_matrices` figure is superseded). Operating (thr 0.058): TN 81 / FP 17 / FN 1 / TP 76. Max-F1 (thr 0.608): TN 87 / FP 11 / FN 7 / TP 70. Base-rate (thr 0.094): TN 84 / FP 14 / FN 2 / TP 75. *Conclusion:* the three thresholds trace a clean recall–precision dial — the operating point sacrifices precision (17 false positives) to catch all but one of 77 religious missions, while max-F1 roughly halves the false positives at the cost of six more misses.

![](../data/processed/figures/score_distribution_by_tier_label.svg)

**Figure B4 — `score_distribution_by_tier_label`.** Stacked histograms of calibrated probability by tier (HIGH/MEDIUM/LOW), colored by predicted label, with the three thresholds and the inter-threshold band marked. Sharply bimodal in every tier (spikes near 0 and near 1). *Conclusion:* classification is insensitive to exact threshold placement within the wide gap; LOW shows marginally more mid-range mass.

![](../data/processed/figures/threshold_sweep.svg)

**Figure B5 — `threshold_sweep`.** Dual-axis sweep per tier: predicted-positive rate (left) and precision (right) vs. threshold. Predicted-positive rate collapses to a flat ~11–13% by threshold 0.05; precision climbs to a ~90% shoulder immediately. *Conclusion:* because scores are bimodal, both quantities are flat across a wide window, so the chosen operating points sit on a stable plateau.

![](../data/processed/figures/prevalence_decomposition.svg)

**Figure B6 — `prevalence_decomposition`.** Bar chart of each stratum's contribution to the composite with error bars and a "composite 14.4%" reference line. HM-PPI ~ 0.104 (widest CI), LOW-PPI ~ 0.120, LOW-RG ~ 0.144. *Conclusion:* method/stratum choice shifts the contribution by ~4 pp; the composite lands at 14.4%.

![](../data/processed/figures/prevalence_forest.svg)

**Figure B7 — `prevalence_forest`.** Per-NTEE forest of prevalence with CI whiskers. One dominant outlier: X ~ 0.82 (0.77–0.87). All others low: P ~ 0.19, Q ~ 0.17 (very wide), S ~ 0.12, I ~ 0.12, B ~ 0.09, most others ~ 0.005–0.01 with tight intervals. *Conclusion:* strong face validity — religion concentrates in group X with a modest service-sector tail.

![](../data/processed/figures/rule_validation_intervals.svg)

**Figure B8 — `rule_validation_intervals`.** Dot-and-whisker Wilson intervals for the LOW-tier rule. Specificity 1.00 (n=54, lower bound ~0.93); sensitivity 0.846 (n=13, CI ~0.58–0.97). *Conclusion:* the rule is highly specific but its recall is uncertain on a tiny positive sample — the driver of the wide LOW-tier band.

![](../data/processed/figures/quantification_sensitivity.svg)

**Figure B9 — `quantification_sensitivity`.** Interval/point plot across quantifiers. PPI weighted ~ 0.136 (0.117–0.155), PPI unweighted ~ 0.131 (0.112–0.150), anchor-multiplicity PPI ~ 0.131, and EMQ ~ 0.167 (point only, no interval). *Conclusion:* estimates are robust (~0.13–0.14) except EMQ, the visible high outlier.

![](../data/processed/figures/subgroup_performance.svg)

**Figure B10 — `subgroup_performance`.** Multi-metric strip plot (minority-F1, FPR, FNR) by word-count bin, NTEE group, and overall. The aggregate `upstream` row (n=175): FNR ~ 0.02, FPR ~ 0.18, minority-F1 ~ 1.0. Most subgroups score F1 >= 0.8 with mild FPR variation; several NTEE cells rest on n <= 10. *Conclusion:* limited subgroup bias, but small-n cells are low-information.

![](../data/processed/figures/ngram_log_odds.svg)

**Figure B11 — `ngram_log_odds`.** Top-30 *naive* signed n-gram log-odds (a simple aggregate precursor to the prior-shrunk weighted version in Figures B15–B17). Descending: *jesus* (~7.3), *jesus christ*, *of jesus*, *gospel*, *the gospel*, *christ*, *god*, *christ centered*, then *catholic*, *prayer*, *bible*, *discipleship*, *missionaries*. *Conclusion:* the classifier keys on intuitive, face-valid Christian/Catholic vocabulary; the weighted diagnostics that follow confirm the same ordering with a rigorous statistical prior.

![](../data/processed/figures/documentation_curve.svg)

**Figure B12 — `documentation_curve`.** Validation PR-AUC by encoder at full training data. After the run log was cleaned of degenerate rows from a since-fixed multi-GPU precision bug, all four learners now cluster tightly in the 0.90–0.95 band: DeBERTa-v3-base ~0.945 (highest), TF-IDF logistic regression ~0.938, ModernBERT-base ~0.935, and MiniLM logistic regression ~0.895. *Conclusion:* the transformers and even a linear TF-IDF baseline achieve near-identical validation PR-AUC, indicating the religious/non-religious signal in mission text is strong and largely lexical, and that no learner is data-starved at 20k silver rows. *Caveat:* only a single training fraction (1.0) was run, so this is a per-encoder snapshot, not a true learning curve; the x-axis is a degenerate auto-zoom around 1.0. A genuine multi-fraction learning curve — which would let us read off the point at which added silver stops helping — was not run and is future work (see §5).

![](../data/processed/figures/production_annotation_summary.svg)

**Figure B13 — `production_annotation_summary`.** Annotation diagnostics for the production slate. Mean pairwise agreement 100%; all-abstain 4%; tie 0%; per-prompt abstain rates DeepSeek v2 12% / v3 8% / v1 4%; vote-count distribution 0/1/2/3 votes = 784 / 604 / 1,413 / 17,540. *Conclusion:* near-perfect label consensus, most items unanimous.

![](../data/processed/figures/bakeoff_summary.svg)

**Figure B14 — `bakeoff_summary`.** Forest of minority-F1 (CI) and Cohen's $\kappa$ per modelxprompt arm, sorted, with a 0.70 decision floor and high-abstain (>=25%) rings. Top: DeepSeek v2 ($\kappa$ ~0.97). Several top scorers carry high-abstain rings (26–36%); `gpt-4o- mini v3` fails outright (F1 CI ~0.3–0.6). *Conclusion:* exposes the accuracy/abstention trade-off behind slate selection.

## Vocabulary diagnostics: what separates religious from non-religious missions

The figures above validate the classifier's *performance*; the figures below open its *content*, showing which words drive the religious/non-religious distinction in the silver-labeled text. They are diagnostics on the language itself, not on model outputs, and they serve two purposes: a face-validity check (does the model separate on religious vocabulary, or on artifacts?) and a substantive description of how religious and secular nonprofits talk about their missions. Because the two classes share a large generic-nonprofit vocabulary (*community, education, health, children*), raw word counts are uninformative about what is *distinctive*; the weighted diagnostics correct for this, and the contrast between the raw-frequency and distinctiveness-weighted views (Figures B18 vs. B19) is itself the argument for using weighting rather than counts. We report the diagnostics across three n-gram orders on purpose: unigrams isolate the single strongest markers, bigrams surface theological phrases, and trigrams resolve named entities and organizations (the YMCA, Habitat for Humanity), so the progression shows the signal sharpening from words to identities rather than repeating one result three times.

Figures B15–B17 report **weighted log-odds z-scores** in the sense of @monroe2008fightin ("Fightin' Words"), which place an informative Dirichlet prior over the vocabulary so that rare words are shrunk toward zero and only terms with reliably class-skewed usage receive large scores. The x-axis is the z-score (positive = distinctively religious, negative = distinctively non-religious); each panel shows the 30 highest-magnitude n-grams, which for this corpus are overwhelmingly on the religious side. They are the statistically rigorous companion to the naive aggregate log-odds of Figure B11.

![](../data/processed/figures/ngram_weighted_log_odds_unigram.svg)

**Figure B15 — `ngram_weighted_log_odds_unigram`.** Weighted log-odds z-scores for single words. The religious pole is led by *christian* (z~36, by far the strongest single signal), then *faith* (~27), *church* (~25), *spiritual* (~25), *jewish* (~23), *religious* (~23), *love* (~21), *christ* (~20), *ministry* (~20), *god* (~16), *catholic* (~15), *jesus* (~14). The much smaller non-religious pole is defined by secular civic and scientific words: *public* (z~-18, largest negative), *research* (~-14), *county* (~-13), *promote* and *improve* (~-11). *Conclusion:* the separating vocabulary is unambiguously and intuitively religious, with denominational identity (Christian, Jewish, Catholic) as the strongest axis.

![](../data/processed/figures/ngram_weighted_log_odds_bigram.svg)

**Figure B16 — `ngram_weighted_log_odds_bigram`.** Weighted log-odds for word pairs. Religious: *jesus christ* (z~18), *of jesus* (~17), *and spiritual* (~16.5), *faith based* (~15.5), *the gospel* (~15), *of god* (~12.5), *of christ* (~11.5), *young mens christian* / *mens christian* (~11), *christ centered* (~10.5), *jewish community* (~10.5). Non-religious: *to promote* (z~-11, largest negative), *the public* (~-10.5), *low income* (~-9), *to improve* (~-9). *Conclusion:* religious bigrams are explicitly theological or denominational, while the secular side is service-delivery and administrative boilerplate.

![](../data/processed/figures/ngram_weighted_log_odds_trigram.svg)

**Figure B17 — `ngram_weighted_log_odds_trigram`.** Weighted log-odds for word triples. Religious: *of jesus christ* (z~15.5), *the gospel of* (~11.5), *gospel of jesus* (~11), *mens christian association* (~10.5), *young mens christian* (~10.5), *the love of* (~9.5), *of the jewish* (~8.5), *habitat for humanity* (~7.5), *word of god* (~6.5), *the healing ministry* (~6). Non-religious: *quality of life* (z~-7.5, largest negative), *to improve the* (~-7). *Conclusion:* trigrams sharpen the signal into named phrases and organizations (e.g., YMCA — *young mens christian association* — and *habitat for humanity* surface as faith-associated), while only generic mission clichés mark the secular side.

Figures B18–B19 present the same information as class-conditional **wordclouds**, where word size encodes weight and color encodes class (blue = religious, orange = non-religious). Two complementary views are shown. The **distinctive** clouds (B18) size each word by its log-odds distinctiveness, suppressing shared vocabulary; the **frequency** clouds (B19) size each word by its raw within-class frequency. The pair is deliberately juxtaposed to make the weighting argument visible.

![](../data/processed/figures/wordcloud_distinctive_unigram_religious.svg)
![](../data/processed/figures/wordcloud_distinctive_unigram_nonreligious.svg)
![](../data/processed/figures/wordcloud_distinctive_bigram_religious.svg)
![](../data/processed/figures/wordcloud_distinctive_bigram_nonreligious.svg)
![](../data/processed/figures/wordcloud_distinctive_trigram_religious.svg)
![](../data/processed/figures/wordcloud_distinctive_trigram_nonreligious.svg)

**Figure B18 — `wordcloud_distinctive_{unigram,bigram,trigram}_{religious,nonreligious}`.** Distinctiveness-weighted (log-odds) wordclouds, six panels. The religious panels (blue) are dominated, across all three n-gram orders, by *christian, faith, christ, church, jewish, spiritual* (unigrams); *jesus christ, the gospel, of jesus, faith based, christ centered* (bigrams); and *of jesus christ, gospel of jesus, mens christian association, word of god, habitat for humanity* (trigrams). The non-religious panels (orange) are dominated by *public, research, county, cancer, animals* (unigrams); *to promote, the public, low income, high school, affordable housing* (bigrams); and *to improve the, quality of life, the city of, the sport of, the game of* (trigrams). *Conclusion:* once shared vocabulary is stripped away, the two classes occupy cleanly separated lexical spaces — religion/theology versus civic, scientific, health, animal-welfare, sport and municipal service.

![](../data/processed/figures/wordcloud_frequency_unigram_religious.svg)
![](../data/processed/figures/wordcloud_frequency_unigram_nonreligious.svg)
![](../data/processed/figures/wordcloud_frequency_bigram_religious.svg)
![](../data/processed/figures/wordcloud_frequency_bigram_nonreligious.svg)
![](../data/processed/figures/wordcloud_frequency_trigram_religious.svg)
![](../data/processed/figures/wordcloud_frequency_trigram_nonreligious.svg)

**Figure B19 — `wordcloud_frequency_{unigram,bigram,trigram}_{religious,nonreligious}`.** Raw-frequency wordclouds, six panels. Here the two classes look *superficially similar*: the most frequent unigrams in both are generic nonprofit words — *community, education, health, children, people, families, youth, care* — with the religious panel additionally surfacing *christian, christ, church, faith, god* and the secular panel *school, students, housing, research, county*. The same overlap appears in bigrams (both share *non profit, health care*) and trigrams (both share IRS filing language such as *internal revenue code, section 501, primary exempt purpose*). *Conclusion:* raw frequency is a poor discriminator because the classes share a large mission-boilerplate vocabulary — precisely why the distinctiveness-weighted views (B15–B18) are needed to read the signal. (Two innocuous preprocessing residues are visible in these raw-frequency panels — an HTML non-breaking space and IRS filing boilerplate such as *internal revenue code*; they affect neither the classifier nor any estimate, and are discussed in §8.)

# Appendix C. Supplementary Tables

**Table C1 — Bake-off (model x prompt), sorted by rank.** Dual gate: $\kappa \geq 0.70$ AND minority-F1 CI-lower $\geq 0.70$. 12 of 15 arms clear.

| Rank | Model | Prompt | $\kappa$ | Min-F1 | F1 CI-low | Abstain | n | Clears |
|---|---|---|---|---|---|---|---|---|
| 1 | DeepSeek-V4-Flash | v2 | 1.000 | 1.000 | 1.000 | 0.22 | 39 | pass |
| 2 | gpt-4o-mini | v2 | 1.000 | 1.000 | 1.000 | 0.26 | 37 | pass |
| 3 | gpt-5-mini | v3 | 0.947 | 0.966 | 0.870 | 0.18 | 41 | pass |
| 4 | DeepSeek-V4-Flash | v3 | 0.945 | 0.966 | 0.875 | 0.22 | 39 | pass |
| 5 | gpt-5-mini | v2 | 0.941 | 0.966 | 0.878 | 0.30 | 35 | pass |
| 6 | gpt-5-nano | v2 | 0.932 | 0.957 | 0.833 | 0.36 | 32 | pass |
| 7 | DeepSeek-V4-Flash | v1 | 0.906 | 0.941 | 0.833 | 0.10 | 45 | pass |
| 8 | gpt-5-nano | v3 | 0.900 | 0.933 | 0.815 | 0.10 | 45 | pass |
| 9 | gpt-5-mini | v1 | 0.898 | 0.933 | 0.815 | 0.14 | 43 | pass |
| 10 | gpt-5-nano | v1 | 0.886 | 0.923 | 0.786 | 0.20 | 40 | pass |
| 11 | gemma-3-27b-it | v1 | 0.848 | 0.897 | 0.741 | 0.10 | 45 | pass |
| 12 | gemma-3-27b-it | v2 | 0.795 | 0.867 | 0.706 | 0.14 | 43 | pass |
| 13 | gpt-4o-mini | v1 | 0.786 | 0.846 | 0.667 | 0.08 | 46 | fail |
| 14 | gpt-4o-mini | v3 | 0.769 | 0.833 | 0.625 | 0.14 | 43 | fail |
| 15 | gemma-3-27b-it | v3 | 0.757 | 0.839 | 0.667 | 0.08 | 46 | fail |

**Table C2 — Encoder-selection sweep (3 seeds each).** Selection on mean validation PR-AUC with the parsimony tie-rule. Winner: DeBERTa `default` soft (bold).

| Arm | Encoder | Targets | Val PR-AUC (mean ± sd) | Val min-F1 (mean) |
|---|---|---|---|---|
| hard | DeBERTa-v3-base | hard | 0.958 ± 0.006 | 0.859 |
| class_weighted | DeBERTa-v3-base | soft | 0.953 ± 0.001 | 0.849 |
| default | ModernBERT-base | soft | 0.940 ± 0.007 | 0.841 |
| **default** | **DeBERTa-v3-base** | **soft** | **0.941 ± 0.012** | **0.835** |

Tie-rule chain: `hard` (0.958) → `class_weighted` (0.953, within sd → tie to simpler soft) → `default` DeBERTa (0.941, within sd → tie to simpler default) beats `default` ModernBERT (0.940, within sd → tie to DeBERTa). Final refit uses 5 seeds (42–46); representative seed 44, checkpoint `checkpoint-2690`.

**Table C3 — Frozen-test subgroups by word count.**

| Word-count bin | n | Minority-F1 | FNR | FPR |
|---|---|---|---|---|
| 0–10 | 12 | 1.000 | 0.000 | 0.000 |
| 11–25 | 45 | 0.837 | 0.053 | 0.231 |
| 26–50 | 59 | 0.926 | 0.000 | 0.118 |
| 51+ | 59 | 0.896 | 0.000 | 0.241 |

**Table C4 — Per-NTEE prevalence (estimated groups, $n_{\text{anchor}} \geq 10$).** Estimator: PPI/Rogan–Gladen composite. Nine groups suppressed for $n < 10$.

| NTEE | Sector (abbrev.) | n | Prevalence | 95% CI |
|---|---|---|---|---|
| X | Religion-related | 34 | 0.822 | 0.773–0.872 |
| P | Human Services | 58 | 0.194 | 0.133–0.256 |
| Q | Int'l/Foreign Affairs | 14 | 0.174 | 0.003–0.346 |
| I | Crime/Legal | 10 | 0.123 | 0.119–0.128 |
| S | Community Improvement | 19 | 0.116 | 0.000–0.259 |
| B | Education | 73 | 0.085 | 0.058–0.112 |
| F | Mental Health | 13 | 0.075 | 0.064–0.085 |
| T | Philanthropy | 23 | 0.068 | 0.000–0.162 |
| E | Health | 26 | 0.046 | 0.000–0.124 |
| O | Youth Development | 16 | 0.010 | 0.006–0.015 |
| A | Arts/Culture | 47 | 0.006 | 0.001–0.011 |
| N | Recreation/Sports | 36 | 0.005 | 0.000–0.011 |
| W | Public/Societal Benefit | 10 | 0.005 | 0.001–0.009 |
| L | Housing/Shelter | 18 | 0.004 | 0.002–0.007 |
| G | Disease/Medical | 10 | 0.003 | 0.000–0.007 |
| M | Public Safety | 11 | 0.001 | 0.000–0.009 |
| C | Environment | 13 | 0.001 | 0.000–0.005 |
| D | Animal-related | 15 | 0.0003 | 0.000–0.005 |

(NTEE abbreviations follow the NCCS major-group scheme; where the anchor sector label differs from the classic letter mapping we report the code as recorded.)

**Table C5 — Prevalence summary (all estimands).**

| Estimand | Estimator | Estimate | 95% CI |
|---|---|---|---|
| HIGH+MEDIUM | Classical (anchor labels only, design-weighted) | 0.108 | 0.078–0.144 |
| HIGH+MEDIUM | PPI++ weighted (primary) | 0.135 | 0.117–0.154 |
| HIGH+MEDIUM | PPI++ unweighted | 0.130 | 0.111–0.150 |
| HIGH+MEDIUM | EMQ (cross-check) | 0.167 | — |
| LOW · classifier-routed | PPI | 0.148 | 0.102–0.193 |
| LOW · rule-only | Rogan–Gladen | 0.194 | 0.140–0.248 |
| LOW · composite | PPI+RG | 0.172 | 0.136–0.208 |
| **Full frame · composite** | **share-weighted** | **0.144** | **0.127–0.161** |

*Efficiency note:* on the HIGH+MEDIUM frame the classical (anchor-only) interval has half-width ±3.3 pp; PPI++ narrows it to ±1.9 pp — a 3.1× variance reduction, equivalent to coding ~1,100 anchor rows by hand instead of 351 (§7). The classifier both tightens *and* shifts the estimate: the wide classical interval contains the PPI++ point (0.135), but the label-only point (0.108) falls just below the PPI++ interval — an upward move of ~2.7 pp that §7 discusses as leaning on the classifier signal.

*Systematic-band note:* the composite 12.7–16.1% interval is a sampling interval conditional on the LOW-tier rule's point-estimated sensitivity. Propagating the full LOW systematic band (11.0–28.4%) through the tier shares widens the full-frame estimate to a **11.5–18.4%** outer envelope (§7). The HIGH+MEDIUM row does not depend on the rule and is unaffected.

# Appendix D. Design Decisions and Intentional Simplifications

We follow a "one principled primary method per concern, plus minimal robustness" policy; tertiary machinery is pushed to optional diagnostics or omitted with a stated rationale.

| Decision | Choice | Rationale |
|---|---|---|
| Label aggregation | Majority vote (production) | Simplest defensible aggregation; Dawid–Skene/CROWDLAB run only as diagnostics and did not change labels. |
| Primary quantifier | PPI++ | Design-based, valid under classifier misspecification; survey-sampling roots. |
| Quantification cross-check | EMQ (SLD) | Lightweight, interpretable; KDEy optional. |
| Training targets | Soft vote-shares | Preserve annotator confidence; robust to label noise; make label smoothing redundant. |
| Training arms | `default`, `class_weighted` (hard/pruned optional) | Soft targets already down-weight the disagreement band a prune targets. |
| Calibration | Platt (vs. temperature); isotonic excluded | Platt intercept absorbs enrichment prior-shift; isotonic overfits at n~500. |
| Threshold policy | Precision-floor (0.80), plus max-F1 and base-rate labels shipped | Lets downstream users pick their own precision/recall trade-off. |
| Final seeds | 5 (42–46) | Convention for variance reporting; captures optimization stochasticity. |
| Acceptance (calibration) | ECE-only | Standard, reviewer-expected; Brier/log-loss gating is future work. |
| Data augmentation | None | Ensemble soft labels already add noise robustness for short texts. |
| Decision-curve analysis | Dropped | Requires a treat/abstain cost ratio undefined for a prevalence estimand. |

# Appendix E. Provenance Reconciliation

`run_manifest.json` (stamped 2026-07-02 14:52 UTC) reports `wave2_completeness.status = "missing"` and `input_row_counts.predictions_full_parquet = null`, which appears to contradict `prevalence_report.json` consuming `predictions_full.parquet` and reporting 560,354 organizations. The reconciliation: the manifest was generated ~11 hours *before* `predictions_full.parquet` was written (file mtime 2026-07-03 01:40) and before the prevalence stage ran (2026-07-03 01:42). At stamp time the per-organization artifact did not yet exist, so those two fields were correctly `missing`/`null` *as of that timestamp*; the manifest was simply not re-stamped after the later stages completed. The 560,354-organization count is authoritative (corroborated by the file itself and the prevalence report). This is a housekeeping gap in when the manifest is written, not a data inconsistency; a released build should re-stamp the manifest as the terminal step.

# Appendix F. Human Checkpoints (Gates)

The pipeline refuses to advance past four human gates, so no GPU or API spend precedes a human sign-off:

- **G1 (Labels):** the gold coding template is fully coded 0/1 before bake-off, QC, training,
and evaluation.
- **G2 (Slate):** a human confirms the production annotator slate before full annotation.
- **G4 (Anchor):** the anchor sample is fully coded before evaluation and prevalence.
- **G3 (Test unlock):** the frozen test is unlocked only by a signed record naming the exact
checkpoint hash and matching the acceptance config — the discipline that keeps the model from ever seeing the final exam during development.

# References
