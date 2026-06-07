---
created: 20260606
agent: tech-seeker
scratchpad: .agents/notebooks/20260606-tech-calibration-quantification-prevalence-scratchpad.md
status: complete
title: Tech - Calibration Quantification Prevalence
topic: calibration; quantification; prevalence; transformers; misclassification
---

# Tech: Calibration, Quantification, and Population Prevalence from Transformer Classifiers

## From Replication Packages

Prior matching replication output: `.agents/docs/20260605-replication-calibration-prevalence.md`.

| Paper | Language | Key Files | Method | Source URL |
|-------|----------|-----------|--------|-----------|
| Guo et al. (2017) | Python/PyTorch | `temperature_scaling.py` | Temperature scaling / ECE | https://github.com/gpleiss/temperature_scaling |
| Mukhoti et al. (2020) | Python/PyTorch | `Losses/`, `Metrics/`, `temperature_scaling.py` | Focal loss calibration | https://github.com/torrvision/focal_calibration |
| QuaPy / Moreo et al. (2021) | Python | `quapy/`, examples | CC, ACC, PCC, PACC, EMQ/SLD, HDy | https://github.com/HLT-ISTI/QuaPy |
| QuantificationLib | Python | `quantificationlib/`, examples | Aggregative quantification | https://github.com/AICGijon/quantificationlib |
| Keith & O'Connor (2018) | Python/Jupyter | `code/`, notebooks | Prevalence intervals | https://github.com/slanglab/doc_prevalence |
| Hopkins & King ReadMe | R/Python | `R/`, demos | Direct nonparametric prevalence estimation | https://github.com/iqss-research/ReadMeV1 |
| Angelopoulos et al. (2023) | Python | `ppi_py/` | Prediction-powered inference | https://github.com/aangelopoulos/ppi_py |
| Linder et al. (2026) | Python | simulation and application scripts | Multicalibrated LLM prevalence | https://github.com/facebookresearch/multicalibrated_llm_measurement |

## Package Recommendations

| Language | Package | Version | Key Function | Source |
|----------|---------|---------|-------------|--------|
| Python | scikit-learn | current stable | `CalibratedClassifierCV`, `IsotonicRegression`, logistic calibration patterns | https://scikit-learn.org/stable/modules/calibration.html |
| Python | netcal | current stable | Temperature/logistic/beta calibration; ECE/reliability diagrams | https://github.com/EFS-OpenSource/calibration-framework |
| Python | QuaPy | current docs report 0.2.0 | Quantification workflow: CC/ACC/PCC/PACC/EMQ/HDy/evaluation protocols | https://hlt-isti.github.io/QuaPy/ |
| Python | ppi_py | current GitHub | Prediction-powered inference point estimates/CIs using gold audit labels | https://github.com/aangelopoulos/ppi_py |
| R/Python | ReadMe | legacy but still canonical | Aggregate text class proportion estimation | https://github.com/iqss-research/ReadMeV1 |

## Implementation Examples

No full code requested. Practical formula examples:

### Example 1: PCC prevalence from calibrated transformer scores

**Source**: Bella et al. (2010); Guo et al. (2017); Silva Filho et al. (2023).  
**Package version**: method-level; use current `scikit-learn` / `netcal` / `QuaPy`.

```text
Fit transformer on train split.
Fit calibrator on held-out calibration labels: p_cal = g(logit_or_score).
Estimate cell/population prevalence: pi_hat = mean_i p_cal_i.
Bootstrap documents + calibration sample, or use PPI/gold-audit correction for CI.
```

**Notes**: This is the default when the target estimand is prevalence, not a case list. It avoids threshold-induced bias but depends on target-population calibration.

### Example 2: ACC / Rogan-Gladen correction for threshold counts

**Source**: Forman (2008); epidemiological prevalence correction lineage.  
**Package version**: method-level; implemented in QuaPy / QuantificationLib.

```text
pi_CC = share(predicted_label == 1)
pi_ACC = (pi_CC - FPR_hat) / (TPR_hat - FPR_hat)
clip pi_ACC to [0, 1], and report uncertainty in TPR_hat/FPR_hat.
```

**Notes**: Useful when only hard labels are retained, but unstable when TPR≈FPR or validation positives are few.

## Version and Compatibility Notes

- For binary BERT/RoBERTa classifiers, start with held-out Platt/logistic or temperature scaling; isotonic is attractive but needs more calibration data and can overfit.
- Vector/matrix/Dirichlet scaling mainly matter for multiclass outputs; matrix/Dirichlet can overfit with small calibration sets.
- Focal loss may improve ECE in some imbalanced neural settings, but it is not a substitute for post-hoc calibration and should be checked with Brier/log-loss/reliability plots.
- Quantification methods often assume prior/label shift; recent work shows they can fail under covariate/concept shift. Use subgroup calibration or multicalibration when target populations differ by sector, geography, year, text source, or language.
- Weak/LLM labels should not be treated as gold labels for calibration or inference; reserve human/expert audit labels for final calibration and measurement-error correction.

## Search Strategy and Exact Queries Used

Sources searched/used: local prior literature and replication artifacts, web-indexed publisher pages, arXiv, PMLR, ACL Anthology, JMLR, Springer, ACM, Wiley/SAGE/Science, RePEc/IDEAS, and GitHub replication/package pages.

Exact queries run in this workstream:

1. `2026 "Unbiased Prevalence Estimation" "Multicalibrated LLMs" authors arxiv`
2. `2024 2025 quantification learning survey classify and count adjusted count probabilistic classify count EM prior shift prevalence estimation machine learning`
3. `2023 2024 2025 classifier calibration survey distribution shift neural networks transformer calibration temperature scaling isotonic Dirichlet calibration ECE Brier`
4. `applied economics computational social science text classifiers estimate prevalence trends population measurement machine learning labels survey text as data prevalence classifier misclassification`
5. `econometrics misclassification binary dependent variable machine learning generated variables regression measurement error classifier labels prediction powered inference data mined variables DOI`
6. `"A Comparative Evaluation of Quantification Methods" 2025 JMLR authors`
7. `Esuli Moreo Sebastiani 2023 quantification learning survey overview QuaPy quantification methods`

Existing local search-query set reused from `.agents/docs/20260605-literature-calibrated-classifier-prevalence.md` covered Platt scaling, isotonic regression, temperature scaling, Dirichlet calibration, quantification, Hopkins-King, PPI, focal loss, BERT calibration, and weak/LLM labels.

## Key Papers

| # | Paper | Authors | Year | Venue/source | DOI/link | Why it matters |
|---|---|---|---|---|---|
| 1 | Probabilistic Outputs for SVMs | Platt | 1999 | Advances in Large Margin Classifiers | https://www.researchgate.net/publication/2437816 | Platt/logistic scaling foundation. |
| 2 | Transforming Classifier Scores into Accurate Multiclass Probability Estimates | Zadrozny & Elkan | 2002 | KDD | https://doi.org/10.1145/775047.775151 | Isotonic calibration foundation. |
| 3 | On Calibration of Modern Neural Networks | Guo, Pleiss, Sun & Weinberger | 2017 | ICML | https://proceedings.mlr.press/v70/guo17a.html | Temperature scaling; modern neural nets are often miscalibrated. |
| 4 | Beyond Temperature Scaling | Kull et al. | 2019 | NeurIPS | https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html | Dirichlet calibration; multiclass scaling. |
| 5 | Classifier Calibration: A Survey | Silva Filho et al. | 2023 | Machine Learning | https://doi.org/10.1007/s10994-023-06336-7 | Best calibration overview; metrics, proper scoring rules, post-hoc methods. |
| 6 | Calibration of Pre-trained Transformers | Desai & Durrett | 2020 | EMNLP | https://aclanthology.org/2020.emnlp-main.21/ | Directly relevant to BERT/transformer text classifiers. |
| 7 | Calibrating Deep Neural Networks using Focal Loss | Mukhoti et al. | 2020 | NeurIPS | https://papers.neurips.cc/paper/2020/hash/aeb7b30ef1d024a76f21a1d40e30c302-Abstract.html | Focal loss can affect calibration under imbalance. |
| 8 | Quantifying Counts and Costs via Classification | Forman | 2008 | Data Mining and Knowledge Discovery | https://doi.org/10.1007/s10618-008-0097-y | CC/ACC and quantification framing. |
| 9 | Adjusting Outputs to New A Priori Probabilities | Saerens, Latinne & Decaestecker | 2002 | Neural Computation | https://doi.org/10.1162/089976602753284446 | EM / SLD prior-shift adjustment. |
| 10 | Quantification via Probability Estimators | Bella et al. | 2010 | ICDM | https://doi.org/10.1109/ICDM.2010.75 | PCC/PACC: probability-based prevalence estimation. |
| 11 | A Review on Quantification Learning | González et al. | 2017 | ACM Computing Surveys | https://doi.org/10.1145/3117807 | Core quantification survey. |
| 12 | Learning to Quantify | Esuli, Fabris, Moreo & Sebastiani | 2023 | Springer open-access book | https://link.springer.com/book/10.1007/978-3-031-20467-8 | Recent comprehensive quantification methods/evaluation guide. |
| 13 | Binary Quantification and Dataset Shift | González, Moreo & Sebastiani | 2024 | DMKD | http://nmis.isti.cnr.it/sebastiani/Publications/DMKD2024a.pdf | Shows prior-shift methods can fail under covariate/concept shift. |
| 14 | A Comparative Evaluation of Quantification Methods | Schumacher, Strohmaier & Lemmerich | 2025 | JMLR | https://www.jmlr.org/papers/v26/21-0241.html | Large benchmark of 24 quantification methods. |
| 15 | A Method of Automated Nonparametric Content Analysis for Social Science | Hopkins & King | 2010 | AJPS | https://doi.org/10.1111/j.1540-5907.2009.00428.x | Classic aggregate text proportion estimation; warns against document-level focus. |
| 16 | Uncertainty-Aware Generative Models for Inferring Document Class Prevalence | Keith & O'Connor | 2018 | EMNLP | https://aclanthology.org/D18-1487/ | Bayesian/group-level prevalence intervals for text. |
| 17 | Text as Data | Gentzkow, Kelly & Taddy | 2019 | JEL | https://doi.org/10.1257/jel.20181020 | Applied-economics text-as-data gateway; generated variables in downstream analysis. |
| 18 | Misclassification in Binary Choice Models | Meyer & Mittag | 2017 | Journal of Econometrics | https://doi.org/10.1016/j.jeconom.2017.06.012 | Econometric correction for misclassified binary variables. |
| 19 | Prediction-Powered Inference | Angelopoulos et al. | 2023 | Science | https://doi.org/10.1126/science.adi6001 | Valid inference using predictions plus gold audit labels. |
| 20 | Unbiased Prevalence Estimation with Multicalibrated LLMs | Linder, Leeper, Haimovich, Tax, Perini & Vojnovic | 2026 | arXiv | https://arxiv.org/abs/2604.21549 | Frontier result: multicalibration for unbiased prevalence under covariate shift. |

## Synthesis: Estimating Population Prevalence from Calibrated Scores

Let `Y_i` be the latent mission label and `p_i` a calibrated estimate of `P(Y_i=1|text_i, metadata_i)`. For a population or subgroup cell, estimate prevalence as `mean(p_i)`. This probabilistic classify-and-count estimator is preferable to threshold counting for measurement because it preserves uncertainty and avoids arbitrary threshold-induced prevalence changes.

Use ACC/PACC/EM/SLD as robustness checks when class-prior shift is plausible. Do not assume these solve covariate or concept shift; under real population differences, calibrate by meaningful strata or use multicalibration over metadata features.

Uncertainty should combine: finite document/sample variation, calibration-label uncertainty, and label-audit uncertainty. For publishable social-science inference, use a probability audit sample of human labels and either bootstrap the full pipeline, use Bayesian/generative prevalence intervals, or use prediction-powered inference / validation correction.

## Practical Recommendations

1. Define the estimand first: national prevalence, yearly trend, state/sector cell, or regression variable.
2. Keep separate train, calibration, and audit-test labels; the audit sample should be probability-based and include key strata.
3. Default prevalence estimator: average calibrated probabilities; report threshold counts only as sensitivity/case-list outputs.
4. Compare calibration methods: uncalibrated, Platt/logistic, temperature scaling, isotonic; choose by Brier/log-loss and reliability curves, not ECE alone.
5. For imbalanced data, report PR-AUC, precision/recall, Brier/log-loss, subgroup calibration, and prevalence error. Do not optimize only F1.
6. Report ACC/PACC/EM/SLD sensitivity if deployment prevalence likely differs from training prevalence.
7. Under shift, audit and calibrate by organization type, source, geography, year, text length, and other metadata; consider multicalibration if many cells are reported.
8. Treat LLM labels as weak/noisy labels; do not use them as the only calibration gold standard.
9. For downstream regressions, avoid naïve hard-label plug-ins; use validation-based measurement-error correction, PPI, sensitivity analysis, or joint models.
10. Report CIs and sensitivity to calibration method, threshold, audit sampling design, and assumed FPR/FNR.

## Gaps, Caveats, and Papers to Read First

- Organization mission text has little direct literature; borrow from political text, survey/open-response coding, sentiment quantification, and economics text-as-data.
- Calibration under covariate/concept shift is the main risk. Marginal reliability plots can look good while subgroup prevalence is biased.
- Recent LLM prevalence/multicalibration work is promising but mostly 2026 preprint-stage.
- Small positive-class audit samples make FPR/FNR, isotonic calibration, ACC, and subgroup estimates unstable.

Read first: Hopkins & King (2010); Forman (2008); Bella et al. (2010); Guo et al. (2017); Silva Filho et al. (2023); Keith & O'Connor (2018); González, Moreo & Sebastiani (2024); Schumacher et al. (2025); Angelopoulos et al. (2023); Linder et al. (2026).
