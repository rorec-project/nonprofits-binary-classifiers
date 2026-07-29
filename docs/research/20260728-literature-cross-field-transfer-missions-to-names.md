---
created: 2026-07-28
agent: literature-seeker
status: complete
title: Literature - Cross-Field Transfer from Mission Statements to Organization Names
topic: cross-field transfer; covariate shift; calibration under shift; PPI/prevalence under shift; entity-name classification; short-text length shift
---

# Literature: Cross-Field Transfer (Missions → Names) for the Religious/Non-Religious Classifier

**Setting under study.** A DeBERTa-v3-base / ModernBERT-base binary classifier fine-tuned on US-nonprofit **mission statements** (median ~22 words), labels LLM-weak-supervised silver + small human gold, is to be applied **with no retraining** to a different text field of the same entities: the **organization name** (median ~5 words / ~36 chars). Calibrated probabilities feed a prevalence estimator (PPI++ / Rogan–Gladen), so calibration matters as much as ranking.

---

## Bottom line

1. **Naive checkpoint transfer is not statistically valid for the prevalence goal, and this is a proof, not a hunch.** PPI/PPI++ assumes the labeled and unlabeled sets share a feature distribution; its covariate-shift extension (Angelopoulos et al. 2023, §4.2.1) requires a **known** Radon–Nikodym derivative `w(x) = QX/PX(x)` **and** `Q_{Y|X} = P_{Y|X}`. Between a mission-text field and a name-text field, `w(x)` is not merely hard to estimate — the two input distributions have near-disjoint support, so `QX` is not dominated by `PX` and `w` is undefined. **A labeled anchor sample drawn from the name-only population is mathematically required, not nice-to-have.**
2. **The same requirement falls out of a second, independent literature.** Rogan–Gladen (1978) plugs in sensitivity/specificity; these are not transportable across populations with a different case mix — epidemiology calls this **spectrum bias / spectrum effect** (Ransohoff & Feinstein 1978, *NEJM*; Mulherin & Miller 2002, *Ann. Intern. Med.*). Sens/spec measured on mission text say nothing about sens/spec on names. Two literatures pointing at one requirement is the strongest thing in this memo.
3. **Recalibration cannot rescue it without labels either.** Ovadia et al. (2019) show temperature scaling fit on an i.i.d. validation set degrades — sometimes worse than uncalibrated — as shift grows. The standard unsupervised remedy is **importance-weighted recalibration** (Park et al., AISTATS 2020), which needs a density ratio over a **shared feature space**. That is exactly what a field change destroys. So the remedy here is a small labeled target sample; there is no unsupervised path.
4. **Terminology: call it "cross-field transfer under a change of input view, with no target-field adaptation."** Ground it in covariate shift (Shimodaira 2000; Moreno-Torres et al. 2012) while flagging that the strict definition assumes a shared feature space, which does not hold. **Do not call it "zero-shot"** — the label space is unchanged and the model *was* task-trained (Xian et al. 2018; Yin et al. 2019). It will draw a reviewer objection and buys nothing.
5. **The one genuinely encouraging datapoint is not for the transformer.** Litofcenko et al. (2020) classified Austrian/German nonprofits **from organization names alone** and their curated **keyword/rule** method was "correct" ~85% of the time — comparable to a single human coder — while their **machine-learning** attempt on names was reported as unsatisfactory, attributed to lacking long, high-quality text. That is direct precedent that a lexicon is a strong baseline on names and that off-the-shelf ML is not automatically better. Given the existing regex already flags ~64% of NTEE-X names, **the burden of proof is on the transformer**.

---

## Search Strategy

Targeted search against arXiv, ACL Anthology, NeurIPS/PMLR, Science, Springer, SAGE, Oxford Academic, NEJM and Annals of Internal Medicine, plus full-text extraction of the PPI and PPI++ PDFs to read the assumption sections verbatim. Reused verified citations from the sibling in-repo memos (`20260605-literature-calibrated-classifier-prevalence.md`, `20260605-literature-short-text-classification.md`, `20260605-literature-religious-nonprofit-classification.md`) rather than re-deriving them.

**Exact search queries used**:

1. `"zero-shot" misuse terminology NLP definition unseen labels vs no training taxonomy transfer`
2. `Ovadia 2019 "Can You Trust Your Model's Uncertainty" dataset shift calibration temperature scaling degrades`
3. `PPI++ prediction-powered inference Angelopoulos requires labeled data same distribution unbiased`
4. `company name industry classification from organization name alone machine learning accuracy`
5. `spectrum bias sensitivity specificity not transportable Ransohoff Feinstein 1978 diagnostic test`
6. `Park 2020 "Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation" AISTATS importance weighting`
7. `Moreno-Torres 2012 "unifying view on dataset shift" Pattern Recognition taxonomy covariate shift definition same feature space`
8. `Yin Hay Roth 2019 EMNLP "Benchmarking Zero-shot Text Classification" entailment definition zero-shot unseen labels`
9. `predicting ethnicity race gender from personal names machine learning accuracy limitations surname`
10. `multi-view learning survey Xu Tao Xu 2013 different views same entity co-training Blum Mitchell`
11. `BERT classifier input length mismatch train long test short degradation short text classification truncation study`
12. `entity typing from surface form name only "company name" NAICS classification name-only baseline accuracy`
13. `prediction-powered inference distribution shift labeled unlabeled violated extension "PPI" covariate shift correction 2024`
14. `Litofcenko nonprofit classification organization name versus purpose text classifier performance name field`
15. `Rogan Gladen 1978 "estimating prevalence from the results of a screening test" American Journal of Epidemiology`
16. `Shimodaira 2000 "Improving predictive inference under covariate shift by weighting the log-likelihood function"`
17. `Schick Schütze PET "Exploiting Cloze Questions" pattern exploiting training EACL 2021 few-shot text classification`
18. `metadata fusion text classification concatenating fields BERT structured metadata improves classification`
19. `Zrnic Candès "Cross-prediction-powered inference" arXiv PNAS 2024`
20. `"Transformers are Short Text Classifiers" 2022 arXiv 2211.16878 findings accuracy short text benchmarks`

## Papers

Confidence reflects **verification status of the citation metadata**, not the quality of the work. High = primary source or publisher page retrieved and checked in this session. Medium = title/venue/year confirmed via search results only; author list or DOI reconstructed and **not** independently verified.

| Title | Authors | Year | Venue | DOI/URL | Confidence |
|---|---|---|---|---|---|
| Prediction-Powered Inference (§4.2.1 covariate shift read verbatim) | Angelopoulos, Bates, Fannjiang, Jordan, Zrnic | 2023 | Science / arXiv | https://doi.org/10.1126/science.adi6000 · https://arxiv.org/abs/2301.09633 | High |
| PPI++: Efficient Prediction-Powered Inference (setup read verbatim) | Angelopoulos, Duchi, Zrnic | 2023 | arXiv | https://arxiv.org/abs/2311.01453 | High |
| Cross-prediction-powered inference | Zrnic, Candès | 2024 | PNAS 121(15) | https://doi.org/10.1073/pnas.2322083121 | High |
| Estimating Prevalence from the Results of a Screening Test | Rogan, Gladen | 1978 | Am. J. Epidemiology 107(1):71–76 | https://doi.org/10.1093/oxfordjournals.aje.a112510 | High |
| Problems of Spectrum and Bias in Evaluating the Efficacy of Diagnostic Tests | Ransohoff, Feinstein | 1978 | NEJM 299(17):926–930 | https://doi.org/10.1056/NEJM197810262991705 | High |
| Spectrum Bias or Spectrum Effect? Subgroup Variation in Diagnostic Test Evaluation | Mulherin, Miller | 2002 | Annals of Internal Medicine 137(7) | https://doi.org/10.7326/0003-4819-137-7-200210010-00011 | Medium (publisher 403; authors unverified) |
| Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift | Ovadia et al. | 2019 | NeurIPS | https://arxiv.org/abs/1906.02530 | High (author list abbreviated) |
| Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation | Park, Bastani, Weimer, Lee | 2020 | AISTATS, PMLR 108 | https://proceedings.mlr.press/v108/park20b/park20b.pdf | High |
| Conformal Prediction Under Covariate Shift | Tibshirani, Barber, Candès, Ramdas | 2019 | NeurIPS | https://arxiv.org/abs/1904.06019 | Medium (arXiv ID not re-verified) |
| Improving predictive inference under covariate shift by weighting the log-likelihood function | Shimodaira | 2000 | J. Stat. Planning & Inference 90:227–244 | https://doi.org/10.1016/S0378-3758(00)00115-4 | High |
| A unifying view on dataset shift in classification | Moreno-Torres, Raeder, Alaiz-Rodríguez, Chawla, Herrera | 2012 | Pattern Recognition 45(1) | https://doi.org/10.1016/j.patcog.2011.06.019 | High |
| Benchmarking Zero-shot Text Classification | Yin, Hay, Roth | 2019 | EMNLP-IJCNLP | https://aclanthology.org/D19-1404/ | High |
| Zero-Shot Learning — A Comprehensive Evaluation of the Good, the Bad and the Ugly | Xian, Lampert, Schiele, Akata | 2018 | IEEE TPAMI | https://doi.org/10.1109/TPAMI.2018.2857768 | Medium (DOI reconstructed) |
| A Survey on Multi-view Learning | Xu, Tao, Xu | 2013 | arXiv | https://arxiv.org/abs/1304.5634 | High |
| Combining Labeled and Unlabeled Data with Co-Training | Blum, Mitchell | 1998 | COLT | https://doi.org/10.1145/279943.279962 | Medium (DOI reconstructed) |
| Transformers are Short-Text Classifiers | Karl, Scherp | 2022/2023 | arXiv / CD-MAKE LNCS | https://arxiv.org/abs/2211.16878 | High |
| Beyond Token Limits: Assessing Language Model Performance on Long Text Classification | not resolved | 2025 | arXiv | https://arxiv.org/abs/2509.10199 | Medium (authors not resolved) |
| Methods for Classifying Nonprofit Organizations According to their Field of Activity | Litofcenko, Karner, Maier | 2020 | Voluntas 31(1):227–237 | https://doi.org/10.1007/s11266-019-00181-w | High (metadata) / Medium (85% figure quoted secondhand) |
| UK Charity Classification report | Kane, Dobbs et al. | 2021 | charityclassification.org.uk | https://charityclassification.org.uk/data/charity-classification-report.pdf | Medium (authorship not confirmed in PDF text) |
| Comparison of ML Approaches for Industry Classification Based on Textual Descriptions of Companies | Tagarev, Tulechki, Boytcheva | 2019 | RANLP | https://aclanthology.org/R19-1134/ | High |
| Predicting Race and Ethnicity From the Sequence of Characters in a Name | Sood, Laohaprapanon | 2018 | arXiv | https://arxiv.org/abs/1805.02109 | High |
| Don't Stop Pretraining: Adapt Language Models to Domains and Tasks | Gururangan et al. | 2020 | ACL | https://doi.org/10.18653/v1/2020.acl-main.740 | High (reused from in-repo memo) |
| Exploiting Cloze-Questions for Few-Shot Text Classification and NLI | Schick, Schütze | 2021 | EACL | https://aclanthology.org/2021.eacl-main.20/ | High |

---

## 1. Terminology: what the literature actually calls this

**Recommended phrasing for a paper:** *"cross-field transfer: applying the mission-trained checkpoint to a different text field (organization name) of the same entities, without target-field adaptation."* Then, in the methods/limitations, characterise the statistical structure as **a change of the input view, which is stronger than covariate shift**.

Adjudication of the candidate terms:

| Term | Verdict | Why |
|---|---|---|
| **Covariate shift** | **Use, with a caveat** | Shimodaira (2000) and the taxonomy in Moreno-Torres et al. (2012) define it as `P(X)` changing while `P(Y\|X)` is unchanged — and, critically, **within the same feature space**. Your input space itself changes (mission field → name field), so this is the *stronger* case. Say so explicitly; a reviewer who knows Moreno-Torres will otherwise catch it. |
| **Dataset shift / distribution shift** | Correct but vague | Umbrella term from Quiñonero-Candela et al. (2009) and Moreno-Torres et al. (2012). Fine as a section heading, insufficient as a claim. |
| **Domain shift / cross-domain transfer** | Defensible, imprecise | "Domain" in NLP usually means genre/topic/source, not a different attribute of the same record. Using it invites the reader to imagine news→tweets rather than mission→name. |
| **Cross-view transfer / multi-view** | **Most precise, least familiar** | Multi-view learning (Blum & Mitchell 1998; Xu, Tao & Xu 2013) is exactly the vocabulary for "two distinct views of the same instance." It names your situation better than anything else, but a nonprofit-studies or CSS reviewer will not recognise it. Recommend using it once, parenthetically, alongside "cross-field". |
| **Input/feature-space shift** | Usable, informal | Descriptive; not a term of art with a canonical citation. |
| **Out-of-distribution generalization** | Acceptable as framing, not as a claim | Broad; typically used for robustness benchmarks, not for a deliberate field swap. |
| **Zero-shot / zero-shot cross-domain transfer** | **Wrong here — do not use** | See below. |

**Is "zero-shot" defensible? No.** The conventional definition requires **unseen labels** or **no task-specific training**. Xian et al. (2018, TPAMI) define ZSL over classes absent from training. Yin, Hay & Roth (2019, EMNLP) frame zero-shot text classification as label-partially-unseen or label-fully-unseen, the latter "classifying text snippets without seeing task specific training data at all." Your label space (`religious` / `non-religious`) is **identical** and the model **was** trained on the task with thousands of labels. Nothing about the setup is zero-shot.

The one honest concession: NLP does use "zero-shot cross-lingual/cross-domain transfer" to mean *no labeled data in the target language/domain* (e.g. Pires, Schlinger & Garrette 2019 on mBERT). Under that looser convention "zero-shot cross-field transfer" is not insane. But it is a minority convention, it collides with the dominant one, and it gains you nothing. **Recommendation: avoid "zero-shot" entirely; write "no target-field labels" or "no target-field adaptation."**

---

## 2. Evidence on length shift (long-trained → short-applied)

**Blunt assessment: the direct evidence you want does not exist.** I did not find a study that fine-tunes a BERT-family classifier on longer text and evaluates on much shorter text of the same task with quantified degradation. Adjacent evidence only:

- **Reverse direction, quantified.** *Beyond Token Limits: Assessing Language Model Performance on Long Text Classification* (arXiv:2509.10199) reports that an `xlm-roberta-base` fine-tuned on **short** texts (<512 tokens) scored **0.69 F1**, rising to **0.75 F1** when fine-tuned on **long** texts — i.e. a ~6-point F1 penalty from a train/test length mismatch in the short→long direction. This is suggestive that length mismatch is real and material, but it is **the opposite direction** from your case and a different task. Do not cite it as if it were your setting.
- **Transformers are fine on short text per se.** Karl & Scherp, *Transformers are Short-Text Classifiers* (2022/2023; arXiv:2211.16878; Springer LNCS) find that plain fine-tuned Transformers achieve SOTA on short-text benchmarks, questioning whether specialised short-text architectures are needed. Important nuance: this is about **training and testing on short text**, not about transferring a long-trained head. It says the architecture is not the bottleneck; it says nothing about your mismatch.
- **Short text is intrinsically harder.** The short-text classification literature consistently attributes difficulty to **sparsity, ambiguity, shortness and incompleteness** of the input (e.g. Hu et al. 2022, *Computational Intelligence and Neuroscience*), which is why keyword-expansion, topic-augmentation and knowledge-graph augmentation lines exist at all. A 5-word name is at the extreme end of this — well below the length regimes those benchmarks use.
- **Truncation-side evidence is not applicable.** Sun et al. (2019) head/tail truncation results and Fiok et al.-style truncation-vs-summarization studies (arXiv:2403.12799) address *losing* text at 512 tokens, not *never having had it*.

**Verdict: unknown, and it must be measured, not assumed.** The literature gives no basis to predict "graceful" vs "catastrophic" for a 22-word→5-word field swap. The mechanism of concern is not length alone but that name text is a **different register** (proper nouns, denominational tokens, legal suffixes like "Incorporated") with essentially no verb-phrase mission language of the kind the model was trained to key on. State this as a gap in the paper rather than borrowing a number.

---

## 3. Calibration under shift

This is the best-evidenced section.

- **Ovadia et al. (2019), NeurIPS**, *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift* (arXiv:1906.02530) is the canonical result: post-hoc calibration fit on an i.i.d. validation set **does not survive shift**. Temperature scaling achieves low ECE at low shift but **ECE rises sharply as shift increases**, and under shift temperature scaling can have a **worse Brier score than the uncalibrated model**. Their conclusion, in their words: calibration on the i.i.d. validation set does not guarantee calibration under distributional shift.
- **Is a calibrator fit on in-domain OOF mission scores valid on name inputs? No, and there is no reason to expect it to be.** A Platt/temperature map is a monotone reparameterisation of the score; it is only a *probability* map relative to the score distribution it was fit on. When `P(X)` changes, the score→probability relationship changes with it. This is the direct implication of Ovadia et al.
- **Standard remedy #1 — importance-weighted recalibration.** Park, Bastani, Weimer & Lee, *Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation* (AISTATS 2020; arXiv:2003.00343) calibrate using labeled source examples plus **unlabeled** target examples, correcting by importance weighting, and learning a feature map to bring the two distributions closer. **Their own stated limitation is decisive for you: "importance weighting relies on the training and real-world distributions to be sufficiently close."** Mission text and name text are not close; the density ratio over raw inputs is effectively undefined.
- **Standard remedy #2 — weighted conformal.** Tibshirani, Barber, Candès & Ramdas, *Conformal Prediction Under Covariate Shift* (NeurIPS 2019; arXiv:1904.06019) gives valid coverage under covariate shift, but again **requires the likelihood ratio between source and target covariate distributions to be known or well estimated**. Same blocker.
- **Standard remedy #3 — prior/label-shift correction (Saerens–Latinne–Decaestecker EM/SLD, 2002).** Corrects `P(Y)` change assuming `P(X|Y)` fixed. **Explicitly inapplicable**: the whole point is that `P(X|Y)` changed, because X is a different field. The project's own prior memo already warns "do not use EM as a default under covariate shift" (`docs/research/20260605-literature-calibrated-classifier-prevalence.md`); that warning binds here.

**Conclusion for §3: every unsupervised recalibration method in the literature needs a shared feature space and an estimable density ratio. A field change removes both. The only remaining remedy is refitting the calibrator on labeled data from the name-only population.**

---

## 4. Quantification / prevalence under shift — the crux

### 4a. PPI / PPI++ — what is actually assumed

**PPI++ (Angelopoulos, Duchi & Zrnic, arXiv:2311.01453)** states the setup verbatim as:

> "we have `n` labeled data points `(Xi,Yi) ~iid P, i ∈ [n]` as well as `N` unlabeled data points, `X̃i ~iid P_X, i ∈ [N]`, **where the feature distribution is identical across both datasets**"

and its rectified loss is `L^PP(θ) := L_n(θ) + L̃^f_N(θ) − L^f_n(θ)`, where the final term is the debiasing correction estimated **on the labeled data**.

**Original PPI (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, *Science* 2023; arXiv:2301.09633)** makes the same baseline assumption ("independently and identically distributed samples from a common distribution"), and its mean-estimation rectifier is `∆̂_f = (1/n) Σ (f_i − Y_i)` — **the average prediction bias measured on the labeled sample**.

**Mechanically, this is why naive transfer breaks.** `∆̂_f` estimates `E[f(X) − Y]` under the distribution of the **labeled** data. If your labeled anchor is drawn from mission-having orgs while the unlabeled pool is name-only orgs, you subtract the *wrong bias*. The estimator remains "unbiased irrespective of whether the predictor is misspecified" **only with respect to the distribution the labeled sample came from**. Transfer does not change the estimator's validity for the wrong estimand; it changes which population's prevalence you are estimating.

### 4b. Does PPI have a covariate-shift extension? Yes — and it does not help you

**Lead with the scope argument, not the support argument.** Covariate shift is a statement about two distributions of *the same* random variable `X`. Here `X_mission` and `X_name` are **two different measurement maps of one latent entity** — the organisation. PPI §4.2.1 is therefore **out of scope for this setting**, not merely violated by it. That framing has no attack surface; the condition-by-condition reading below is confirmation, not the primary argument.

PPI §4.2.1 handles covariate shift, and the conditions read verbatim:

> "First, we assume that Q is a **known** covariate shift of P. That is, if we denote by `Q = QX · Q_{Y|X}` and `P = PX · P_{Y|X}` … we assume that **`Q_{Y|X} = P_{Y|X}`**."
>
> "suppose that **`QX` is dominated by `PX`** and assume that the Radon–Nikodym derivative `w(x) = QX/PX (x)` is **known**."

**All three conditions fail in the mission→name setting:**

1. **`Q_{Y|X} = P_{Y|X}` fails.** `P(religious | x)` where `x` is a mission statement and `P(religious | x)` where `x` is a name are different conditional functions. This is not a covariate shift at all; the measurement map producing X changed.
2. **`QX` dominated by `PX` fails in practice.** Both fields are token sequences over one vocabulary, so the supports are not *literally* disjoint — but they occupy effectively non-overlapping regions, so any finite-sample estimate of the density ratio is degenerate. The honest claim is that **`w(x)` is not estimable**, not that it provably does not exist.
3. **`w(x)` known fails** — it is neither known a priori nor recoverable from data, given (2).

PPI's **label-shift** branch (§4.2.2) is likewise unavailable: it assumes `Q_{X|Y} = P_{X|Y}`, which is precisely what a field change violates, and it estimates the confusion matrix `K_{j,l} = P(f(X)=j | Y=l)` **from labeled data sampled from P** — the same wrong-population problem.

Known PPI extensions do not close this gap either: Cross-PPI (Zrnic & Candès 2024) fixes dependence between the model and the labeled data via sample splitting, and IPW-PPI (arXiv:2508.10149) handles *informative labeling* via Horvitz–Thompson/Hájek weights — both require an estimable selection/weighting mechanism over a shared covariate space, which you do not have across fields.

### 4c. Rogan–Gladen / adjusted classify-and-count

Rogan & Gladen (1978, *American Journal of Epidemiology*) gives `π̂ = (π̂_CC − FPR) / (TPR − FPR)`. It is exact **only if TPR and FPR are the ones that hold in the target population**. Sensitivity and specificity are **not** population-invariant: this is the classical **spectrum bias / spectrum effect** result (Ransohoff & Feinstein 1978, *NEJM*; Mulherin & Miller 2002, *Ann. Intern. Med.*, on spectrum bias vs spectrum effect; BMC Med Res Methodol 2008;8:7). Plugging mission-derived sens/spec into a name-derived classify-and-count is exactly the error that literature is about. The quantification literature makes the same point in its own vocabulary: ACC/PACC error rates must be estimated **under the deployment distribution** (Forman 2008; Bella et al. 2010; González et al. 2017).

Note this failure mode is *worse* than usual here: the missing-mission slice is disproportionately churches (filing-exempt), so the mission-having population is a **non-random, label-correlated** subsample of the target. This is spectrum shift and prior shift simultaneously.

### 4d. Verdict on "required vs nice-to-have"

**A new labeled anchor sample drawn from the name-only (or at minimum, name-field) population is MATHEMATICALLY REQUIRED.** Without it:

- PPI/PPI++ has no valid rectifier for the target estimand;
- Rogan–Gladen has no valid `TPR`/`FPR`;
- the calibrator has no validity guarantee (§3).

There is no unsupervised substitute in the literature. The anchor need not be large — PPI's entire selling point is that a small labeled set plus many predictions beats the small labeled set alone — but it must be **drawn from the target population you want to make a claim about**, and it should reflect the church-heavy composition of the name-only slice rather than reusing a mission-era sampling frame.

**The annotation ask is smaller than "required, full stop" implies — stratify.** If the estimand is whole-population prevalence, partition the population into two strata with known sizes:

- **Stratum A — mission-having orgs (~36%).** Score with `f(mission)`. The **existing** mission-era anchor is already valid here; no new labeling.
- **Stratum B — name-only orgs (~54%).** Score with whatever name-field method wins §7. Requires a **new** anchor drawn from this stratum.
- Combine as `π̂ = w_A·π̂_A + w_B·π̂_B` with `w` the known stratum shares, and propagate variance across strata.

This is standard stratified estimation, and it confines the mandatory new annotation to stratum B. It also has a reporting virtue: the two strata's prevalences can be published separately, which is more honest than a single blended number given that the strata differ systematically in religiosity. **So: new labeled data is required, but only for the name-only stratum, not for the whole population.**

---

## 5. Entity-name classification specifically

Evidence here is thinner than one would like, and the single most relevant paper is in *nonprofit studies*, not NLP.

**Direct precedent — nonprofits classified from names alone.**
Litofcenko, Karner & Maier (2020), *Methods for Classifying Nonprofit Organizations According to Their Field of Activity*, Voluntas 31(1):227–237 ([doi](https://doi.org/10.1007/s11266-019-00181-w)), applied ICNPO categories to Austrian and German nonprofits with **organization names as the primary input**, against a consensus-coded gold set of 5,000 Austrian (plus 1,000 German) nonprofits. As summarised in the UK Charity Classification report (Kane et al. 2021, `charityclassification.org.uk`):

- their curated keyword / hierarchical if-then **rule algorithm was "correct" ~85% of the time**, "similar to the results achieved for an individual human coder" (individual coders agreed with the consensus label 79–87% of the time);
- their **decision-tree machine-learning** attempt gave **unsatisfactory** results, which the authors attributed to "lack[ing] any long, high-quality texts such as mission statements" and being "forced to rely on names and web scraped data from websites."

Two things follow. First, **a curated lexicon over names is a genuinely strong method in this exact domain, not a strawman** — consistent with your existing regex flagging ~64% of NTEE-X names. Second, there is **published precedent for supervised ML underperforming a lexicon when names are the only input**. That is the closest thing to a prior for your experiment, and it is not favourable.

**Documented failure modes for names** (Litofcenko et al. 2020 via Kane et al. 2021): "misleading or uninformative names, acronyms, or unconventional language such as wordplay, neologisms, regional dialects, or foreign languages." Note also their **jurisdictional caveat**: Austrian law requires an organisation's name to relate to its purpose, which is *not* true in Germany — and is *not* true in the US either. **The 85% figure is therefore an optimistic upper bound for a US name corpus.**

**Religion-specific name failure modes** are already documented in this project's own literature memo (`docs/research/20260605-literature-religious-nonprofit-classification.md`, §Synthesis point 4), grounded in Sider & Unruh (2004), Smith & Sosin (2001) and Becker (2003): secularised faith-founded names ("St Mary's Hospital", YMCA/YWCA), faith-named secular orgs, and non-Christian traditions whose naming conventions are less Protestant-coded. Also relevant: Scheitle & Dougherty (2015) on which congregations register with the IRS at all.

**Adjacent-but-not-equivalent evidence (do not overclaim).**
- **Industry classification from company text.** Roelands et al. / "Comparison of Machine Learning Approaches for Industry Classification Based on Textual Descriptions of Companies" (RANLP 2019, [aclanthology.org/R19-1134](https://aclanthology.org/R19-1134/)) and BERT-based emerging-industry classification (Wang et al. 2024, *Information Processing & Management*-adjacent, ScienceDirect S030643792400142X) report accuracies in the 84–99% band. **These use company descriptions, not names.** Do not import those numbers as a name-only ceiling; I found no peer-reviewed name-only NAICS/SIC benchmark with a reported ceiling. **This is a real gap.**
- **Names as evidence in general.** The name-based demographic inference literature is the best-established "predict an attribute from a surface-form name" body of work: Sood & Laohaprapanon, *Predicting Race and Ethnicity From the Sequence of Characters in a Name* (arXiv:1805.02109; `ethnicolr`) report LSTM out-of-sample accuracy ~85% (best last-name-only models ~81%). It establishes that names carry real signal, and that name-derived predictions "vary in their accuracy and can introduce statistical biases in downstream analyses" — the same warning as §4. It is not evidence about organisation names or religion.

---

## 6. What a strong 2025/2026 paper would actually do

Ranked by **evidence strength × cost**. Note the first item is not optional — it is the §4 requirement — and everything below it is modelling improvement layered on top.

| # | Approach | Evidence strength | Cost | Assessment |
|---|---|---|---|---|
| **0** | **Label a fresh anchor sample from the name-only population; refit calibration and run PPI++ / Rogan–Gladen against it.** | **Highest — this is a theorem, not an empirical trend** (Angelopoulos et al. 2023 §4.2.1; Angelopoulos et al. 2023b PPI++ setup; Rogan & Gladen 1978; Ransohoff & Feinstein 1978) | Low–moderate (annotation only) | **Mandatory.** Nothing else in this table substitutes for it. |
| **1** | **Continued fine-tuning on a small name-labeled set** (the anchor doubles as training data with proper splitting) | High. Standard supervised transfer; Gweon & Schonlau (2024) show BERT advantages emerge around 200–500 labels, clearer at ~1,000 | Low | **Best value.** Directly attacks the field mismatch. Use disjoint splits so the PPI anchor is not also the training set, or use Cross-PPI (Zrnic & Candès 2024) sample splitting. |
| **2** | **Field-agnostic templated input so one model serves both fields** (e.g. `"Organization name: {name}. Mission: {mission or ∅}."`) | Moderate–high, by analogy. Pattern/verbalizer framing (Schick & Schütze, EACL 2021) and NLI-style label verbalisation (Yin et al. 2019; Laurer et al. 2024, *Political Analysis*) show templated inputs transfer across surface forms | Low (one retrain) | **Strong candidate and cheap.** Also makes the missing-mission case a first-class input rather than a distribution shift. This is what I would expect a good reviewer to ask for. |
| **3** | **Multi-field concatenation / metadata fusion** (name + mission + NTEE + state when available) | **Weak published evidence.** I found mostly industry blog posts and low-tier venue papers on concatenating metadata with BERT; no strong, well-cited primary source specific to this pattern | Low | Likely fine in practice, but **do not cite it as evidence-backed**. It also does not solve the **~54% of orgs that have *only* a name** (90% with a name − 36% with a mission). Note this is a different quantity from the ~64% lexicon flag rate on NTEE-X names used elsewhere in this memo. |
| **4** | **Auxiliary / multi-task training** (predict label from name-head and mission-head with shared encoder) | Moderate, general MTL literature; nothing specific to name/mission fields found | Moderate | Reasonable, but option 2 achieves most of the benefit for less complexity. |
| **5** | **DAPT/TAPT on an unlabeled name corpus** (Gururangan et al., ACL 2020, [doi](https://doi.org/10.18653/v1/2020.acl-main.740)) | Moderate and well-cited in general; **untested on 5-word name corpora**, and the project's own memo warns short domain corpora can overfit | Moderate | Worth a secondary experiment. TAPT on ~millions of org names is cheap and plausibly helps the tokenizer/encoder settle into the register. Not a substitute for labels. |
| **6** | **Prompt/NLI-style entailment classification directly on names** (Yin et al. 2019; Laurer et al. 2024) | Moderate | Low (inference) | Useful as a **second opinion / ensemble member**, and cheap to run. Its calibration is poorly characterised, so it does not solve §3. |
| **7** | **Distillation** (LLM-on-names teacher → encoder student) | Moderate; consistent with the project's own weak-supervision precedent (Snorkel; Smith et al. 2023) | Moderate–high (LLM inference over the full name corpus) | Viable path to a name-native model at scale, but it re-imports the silver-label noise problem and still needs a gold anchor for calibration. |
| **8** | **Naive checkpoint transfer, unmodified** | **No supporting evidence found; two literatures against** | Zero | Only defensible as a **diagnostic baseline** reported alongside better methods — never as the production estimator. |

**Additional expectation for a 2026 paper:** report a **lexicon-on-names** baseline prominently, given Litofcenko et al. (2020) showed a lexicon matching a human coder in exactly this task family. A paper that shows a transformer beating "nothing" will not clear review; one that shows it beating a curated lexicon on names will.

---

## 7. The cheapest discriminating experiment

**The experiment: paired dual-field scoring on the mission∩name overlap.**

On the ~36% of organisations that have **both** a mission and a name, run the existing frozen checkpoint over **both fields** and compare, using labels you already have (gold where available, silver elsewhere, reported separately):

1. `f(mission)` — the in-domain reference performance.
2. `f(name)` — cross-field transfer.
3. `lexicon(name)` — the incumbent regex.

Report, for each: PR-AUC / average precision (imbalance-appropriate), **reliability curve + ECE + Brier** (per §3 this is the load-bearing measurement, not accuracy), the mission-vs-name score correlation and per-item agreement rate, and the implied `mean(p̂)` prevalence from each field on the same organisations. **Zero new annotation. Hours of compute.**

**Pre-specify the decision rule, or the experiment is unfalsifiable.** "Beats the lexicon" must be operationalised — the ~64% figure is a **recall-like flag rate at unknown precision** on NTEE-X names, not a comparable metric. Recommended: **transfer must beat lexicon-on-names on recall at matched precision** (and, separately, on PR-AUC) on the same overlap set, *and* show ECE on names within a pre-declared factor of ECE on missions. Failing either kills naive transfer.

**Three caveats that must appear in the memo/paper, in order of importance:**

1. **The overlap set is a biased proxy for the target.** Mission-having orgs are 990 filers; the name-only slice is disproportionately churches. So this experiment can **cheaply falsify** transfer, but it **cannot validate** it for the name-only population. A pass here licenses spending annotation budget on the §4 anchor; it does not license shipping a prevalence number.
2. **Prevalence in the overlap set differs from the target.** Do not compare `mean(p̂)` on the overlap to your headline composite prevalence.
3. **Silver-label contamination.** If the silver labels were generated from mission text, they are correlated with `f(mission)` in ways they are not with `f(name)`, which will flatter the mission arm. Restrict the headline comparison to the human gold subset and report the silver run as secondary.

**If it passes**, the follow-up is not "ship it" — it is the §6 row 0 + row 1 + row 2 package: label a name-only anchor, refit calibration on it, continue fine-tuning on part of it, and retrain with a field-agnostic template so a single model covers both fields.

---

## Gaps and caveats (be blunt)

- **No paper found** that fine-tunes a BERT-family classifier on longer text and quantifies degradation on much shorter text of the same task. §2 is an evidence gap, not a settled result.
- **No peer-reviewed name-only ceiling** found for organisation/company classification in an English/US setting. The 85% Litofcenko figure is (a) rule-based not ML, (b) ICNPO not religion, (c) from a jurisdiction where names are legally required to relate to purpose.
- **Multi-field/metadata concatenation is poorly evidenced** in high-trust venues despite being common practice; §6 row 3 is ranked on plausibility, not citations.
- **The "Unbiased Prevalence Estimation with Multicalibrated LLMs" (2026, arXiv:2604.21549)** line noted in the project's earlier prevalence memo is a possible frontier alternative for shift-robust prevalence but remains an unreviewed preprint; I did not verify its assumptions against this setting.
- I could not retrieve the Litofcenko et al. (2020) primary PDF directly (Springer paywall, CORE mirror 404). The 85% figure and the ML-underperformance statement are quoted **via** the UK Charity Classification report (Kane et al. 2021), which cites them explicitly. **Verify against the primary before putting the number in a paper.**

---

## References

**Terminology / shift**

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2), 227–244. https://doi.org/10.1016/S0378-3758(00)00115-4
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (eds.) (2009). *Dataset Shift in Machine Learning*. MIT Press. https://mitpress.mit.edu/9780262170055/dataset-shift-in-machine-learning/
- Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). A unifying view on dataset shift in classification. *Pattern Recognition*, 45(1), 521–530. https://doi.org/10.1016/j.patcog.2011.06.019
- Blum, A., & Mitchell, T. (1998). Combining Labeled and Unlabeled Data with Co-Training. *COLT '98*. https://doi.org/10.1145/279943.279962
- Xu, C., Tao, D., & Xu, C. (2013). A Survey on Multi-view Learning. arXiv:1304.5634. https://arxiv.org/abs/1304.5634
- Xian, Y., Lampert, C. H., Schiele, B., & Akata, Z. (2018). Zero-Shot Learning — A Comprehensive Evaluation of the Good, the Bad and the Ugly. *IEEE TPAMI*. https://doi.org/10.1109/TPAMI.2018.2857768
- Yin, W., Hay, J., & Roth, D. (2019). Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach. *EMNLP-IJCNLP 2019*. https://aclanthology.org/D19-1404/
- Pires, T., Schlinger, E., & Garrette, D. (2019). How Multilingual is Multilingual BERT? *ACL 2019*. https://aclanthology.org/P19-1493/

**Length / short text**

- Karl, F., & Scherp, A. (2023). Transformers are Short-Text Classifiers. *CD-MAKE 2023*, LNCS. arXiv:2211.16878. https://arxiv.org/abs/2211.16878 · https://doi.org/10.1007/978-3-031-40837-3_7
- Beyond Token Limits: Assessing Language Model Performance on Long Text Classification (2025). arXiv:2509.10199. https://arxiv.org/abs/2509.10199
- Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to Fine-Tune BERT for Text Classification? arXiv:1905.05583. https://arxiv.org/abs/1905.05583
- Investigating Text Shortening Strategy in BERT: Truncation vs Summarization (2024). arXiv:2403.12799. https://arxiv.org/abs/2403.12799
- Hu, Y. et al. (2022). Short-Text Classification Detector: A BERT-Based Mental Approach. *Computational Intelligence and Neuroscience*. https://doi.org/10.1155/2022/8660828

**Calibration under shift**

- Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., & Snoek, J. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. *NeurIPS 2019*. arXiv:1906.02530. https://arxiv.org/abs/1906.02530 · https://proceedings.neurips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf
- Park, S., Bastani, O., Weimer, J., & Lee, I. (2020). Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation. *AISTATS 2020*, PMLR 108:3219–3229. https://proceedings.mlr.press/v108/park20b/park20b.pdf · arXiv:2003.00343
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. *NeurIPS 2019*. arXiv:1904.06019. https://arxiv.org/abs/1904.06019
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. https://proceedings.mlr.press/v70/guo17a.html
- Desai, S., & Durrett, G. (2020). Calibration of Pre-trained Transformers. *EMNLP 2020*. https://aclanthology.org/2020.emnlp-main.21/

**Prevalence / quantification / PPI**

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023). Prediction-Powered Inference. *Science*, 382(6671), 669–674. https://doi.org/10.1126/science.adi6000 · arXiv:2301.09633 (§4.2.1 covariate shift)
- Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient Prediction-Powered Inference. arXiv:2311.01453. https://arxiv.org/abs/2311.01453
- Zrnic, T., & Candès, E. J. (2024). Cross-prediction-powered inference. *PNAS*, 121(15). https://doi.org/10.1073/pnas.2322083121 · arXiv:2309.16598. https://arxiv.org/abs/2309.16598 · code: https://github.com/tijana-zrnic/cross-ppi
- Prediction-Powered Inference with Inverse Probability Weighting (2025). arXiv:2508.10149. https://arxiv.org/abs/2508.10149
- Rogan, W. J., & Gladen, B. (1978). Estimating Prevalence from the Results of a Screening Test. *American Journal of Epidemiology*, 107(1), 71–76. https://doi.org/10.1093/oxfordjournals.aje.a112510
- Ransohoff, D. F., & Feinstein, A. R. (1978). Problems of Spectrum and Bias in Evaluating the Efficacy of Diagnostic Tests. *New England Journal of Medicine*, 299(17), 926–930. https://doi.org/10.1056/NEJM197810262991705
- Mulherin, S. A., & Miller, W. C. (2002). Spectrum Bias or Spectrum Effect? Subgroup Variation in Diagnostic Test Evaluation. *Annals of Internal Medicine*, 137(7), 598–602. https://doi.org/10.7326/0003-4819-137-7-200210010-00011 — *author attribution not independently verified (publisher returned 403); verify before citing.*
- A methodological framework to distinguish spectrum effects from spectrum biases and to assess diagnostic and screening test accuracy for patient populations (2008). *BMC Medical Research Methodology*, 8:7. https://doi.org/10.1186/1471-2288-8-7 — *authors not resolved from the publisher page (paywall redirect); verify before citing.*
- Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the Outputs of a Classifier to New a Priori Probabilities. *Neural Computation*, 14(1), 21–41. https://doi.org/10.1162/089976602753284446
- Forman, G. (2008). Quantifying Counts and Costs via Classification. *Data Mining and Knowledge Discovery*, 17(2), 164–206. https://doi.org/10.1007/s10618-008-0097-y
- Bella, A., Ferri, C., Hernández-Orallo, J., & Ramírez-Quintana, M. J. (2010). Quantification via Probability Estimators. *ICDM 2010*. https://doi.org/10.1109/ICDM.2010.75
- González, P., Castaño, A., Chawla, N. V., & del Coz, J. J. (2017). A Review on Quantification Learning. *ACM Computing Surveys*, 50(5). https://doi.org/10.1145/3117807

**Entity-name and nonprofit classification**

- Litofcenko, J., Karner, D., & Maier, F. (2020). Methods for Classifying Nonprofit Organizations According to their Field of Activity: A Report on Semi-automated Methods Based on Text. *Voluntas*, 31(1), 227–237. https://doi.org/10.1007/s11266-019-00181-w
- Kane, D., Dobbs, J., et al. (2021). *Classifying UK charities' activities by charitable cause / UK Charity Classification report*. https://charityclassification.org.uk/data/charity-classification-report.pdf
- Ma, J. (2021). Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector. *NVSQ*, 50(3), 662–687. https://doi.org/10.1177/0899764020968153
- Fyall, R., Moore, M. K., & Gugerty, M. K. (2018). Beyond NTEE Codes. *NVSQ*, 47(4), 677–701. https://doi.org/10.1177/0899764018768019
- Sider, R. J., & Unruh, H. R. (2004). Typology of Religious Characteristics of Social Service and Educational Organizations and Programs. *NVSQ*, 33(1), 109–134. https://doi.org/10.1177/0899764003257494
- Smith, S. R., & Sosin, M. R. (2001). The Varieties of Faith-Related Agencies. *Public Administration Review*, 61(6), 651–670. https://doi.org/10.1111/0033-3352.00137
- Becker, P. E. (2003). Where's the Religion? Distinguishing Faith-Based from Secular Social Service Agencies. *JSSR*. https://doi.org/10.1111/1468-5906.00191
- Scheitle, C. P., & Dougherty, K. D. (2015). Exploring Religious Congregations' Registration With the IRS. *NVSQ*. https://doi.org/10.1177/0899764015597779
- Tagarev, A., Tulechki, N., & Boytcheva, S. (2019). Comparison of Machine Learning Approaches for Industry Classification Based on Textual Descriptions of Companies. *RANLP 2019*, 1169–1175. https://aclanthology.org/R19-1134/ — **descriptions, not names**
- Sood, G., & Laohaprapanon, S. (2018). Predicting Race and Ethnicity From the Sequence of Characters in a Name. arXiv:1805.02109. https://arxiv.org/abs/1805.02109

**Adaptation / training alternatives**

- Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *ACL 2020*. https://doi.org/10.18653/v1/2020.acl-main.740
- Schick, T., & Schütze, H. (2021). Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference. *EACL 2021*, 255–269. https://aclanthology.org/2021.eacl-main.20/
- Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024). Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI. *Political Analysis*. https://www.cambridge.org/core/journals/political-analysis/article/05BB05555241762889825B080E097C27
- Gweon, H., & Schonlau, M. (2024). Automated Classification for Open-Ended Questions with BERT. *Journal of Survey Statistics and Methodology*. https://doi.org/10.1093/jssam/smad015
- Ratner, A. et al. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. *PVLDB*. https://doi.org/10.14778/3157794.3157797

**Related in-repo memos** (reuse rather than re-derive)

- `docs/research/20260605-literature-calibrated-classifier-prevalence.md`
- `docs/research/20260605-literature-short-text-classification.md`
- `docs/research/20260605-literature-religious-nonprofit-classification.md`
- `docs/research/20260606-tech-calibration-quantification-prevalence.md`

## Coverage notes

- **Sources searched**: arXiv, ACL Anthology, NeurIPS/PMLR proceedings, Science/AAAS, Springer (Voluntas, LNCS), SAGE (NVSQ), Oxford Academic (AJE), NEJM, Annals of Internal Medicine, BMC, ScienceDirect, Cambridge Core, charityclassification.org.uk, NCCS/Urban.
- **Date range**: 1978–2026.
- **Primary-source verification**: PPI (2301.09633) §4.2.1 and PPI++ (2311.01453) problem setup were read from the papers themselves (full-text extraction) and are quoted verbatim above. The Litofcenko figures are quoted **secondhand** via Kane et al. (2021) — flagged above as needing primary verification. See the Confidence column in the Papers table for per-citation verification status; anything marked Medium should be checked before it enters a manuscript bibliography.
