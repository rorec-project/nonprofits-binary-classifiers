---
created: 20260605
status: complete
title: Literature Synthesis Map - Mission Text Binary Classifier
reports:
  - .agents/docs/20260605-literature-llm-weak-supervision-noisy-labels.md
  - .agents/docs/20260605-literature-imbalanced-text-validation.md
  - .agents/docs/20260605-literature-calibrated-classifier-prevalence.md
  - .agents/docs/20260605-literature-short-text-classification.md
  - .agents/docs/20260605-literature-religious-nonprofit-classification.md
---

# Literature Synthesis Map: Religious Mission Text Classification

## Workstreams

| Workstream | User topics covered | Report |
|---|---|---|
| Annotation, weak supervision, and noisy labels | LLM-as-annotator, weak supervision, fine-tuning on noisy/programmatic labels, LLM-generated label recipes | `.agents/docs/20260605-literature-llm-weak-supervision-noisy-labels.md` |
| Imbalanced validation and metrics | Evaluation metrics for imbalanced text classification in social science | `.agents/docs/20260605-literature-imbalanced-text-validation.md` |
| Calibration and prevalence estimation | Population-scale prevalence, calibrated transformer classifiers, misclassification correction | `.agents/docs/20260605-literature-calibrated-classifier-prevalence.md` |
| Model alternatives and fine-tuning | BERT, RoBERTa, DeBERTa, DistilBERT, ELECTRA, SBERT/MiniLM, small open-weight LMs, fine-tuning recipes | `.agents/docs/20260605-literature-short-text-classification.md` |
| Religious/nonprofit domain measurement | Religious organization or nonprofit mission classification, NTEE/Form 990, faith-based organization constructs | `.agents/docs/20260605-literature-religious-nonprofit-classification.md` |

## Cross-Literature Bottom Line

The strongest design is a calibrated, audited, weak-supervision pipeline rather than an LLM-only classifier or a plain thresholded BERT model.

For religious mission classification, the literature points to five requirements:

1. Define the construct narrowly: classify observable religious mission or expression in text, not latent religiosity.
2. Use multiple noisy label sources: rules, NTEE/administrative labels, LLM prompts, and human-coded audit labels.
3. Fine-tune encoder classifiers on aggregated or confidence-weighted labels, with clean held-out labels reserved for calibration and evaluation.
4. Evaluate as social measurement: report precision/recall, PR-AUC, MCC, calibration, prevalence error, and subgroup errors, not accuracy alone.
5. Estimate population prevalence from calibrated probabilities and validation-label corrections, not only from thresholded predicted labels.

## Read-First Queue

| Priority | Papers/sources | Reason |
|---|---|---|
| 1 | Grimmer & Stewart (2013); Gentzkow, Kelly & Taddy (2019) | Core text-as-data validity standards for social science and economics. |
| 2 | Hopkins & King (2010); Keith & O'Connor (2018) | Aggregate prevalence estimation from text classifiers. |
| 3 | Gilardi, Alizadeh & Kubli (2023); Pangakis, Wolken & Fasching (2023); Ziems et al. (2024) | LLM annotation promise and validation risks. |
| 4 | Ratner et al. (2016, 2017); Smith et al. (2024) | Weak supervision architecture for rules, prompts, and abstentions. |
| 5 | Zhu et al. (2022); Qi et al. (2023); Wang et al. (2023); Lu & Smith (2025) | Transformer fine-tuning under realistic noisy labels. |
| 6 | Davis & Goadrich (2006); Saito & Rehmsmeier (2015); Hernandez-Orallo et al. (2012) | Metrics, PR curves, and threshold/cost evaluation under imbalance. |
| 7 | Guo et al. (2017); Desai & Durrett (2020); Silva Filho et al. (2023) | Probability calibration for neural and transformer classifiers. |
| 8 | Forman (2008); Bella et al. (2010); Gonzalez et al. (2017); Angelopoulos et al. (2023) | Quantification, adjusted prevalence, and prediction-powered inference. |
| 9 | Ma (2021); Fyall, Moore & Gugerty (2018); Santamarina, Lecy & van Holm (2023) | Closest nonprofit mission/activity classification precedent. |
| 10 | Sider & Unruh (2004); Bielefeld & Cleveland (2013); Smith & Sosin (2001/2002) | Religious and faith-based organization construct definitions. |

## Recommended Empirical Design

### Label Design

Use a multi-label coding layer and derive the binary outcome later.

| Label | Meaning |
|---|---|
| `religious_purpose_explicit` | Mission or purpose text explicitly states worship, ministry, evangelism, religious education, faith formation, scripture, prayer, or similar content. |
| `religious_identity_or_affiliation` | Text/name identifies denomination, congregation, church, mosque, synagogue, temple, saint naming, or faith affiliation. |
| `religious_service_content` | Program/service text includes prayer, chaplaincy, evangelism, religious counseling, missionary activity, or faith-based instruction. |
| `religion_related_NTEE_X` | Administrative label indicates religion-related nonprofit classification. |
| `faith_inspired_ambiguous` | Religious origin or branding appears, but current mission text is not explicitly religious. |

Recommended binary positive rule for the first project iteration: positive if `religious_purpose_explicit`, `religious_service_content`, or high-confidence `religious_identity_or_affiliation`; keep `faith_inspired_ambiguous` separate for audit and sensitivity analysis.

### Annotation Pipeline

1. Build a codebook before large-scale labeling.
2. Hand-label 300-800 examples using stratified sampling across NTEE group, predicted probability, text length, organization type, and religious-name tokens.
3. Double-code a subset and adjudicate disagreements.
4. Create labeling functions with abstentions: keyword patterns, NTEE X, organization-name heuristics, denomination lists, and negative rules.
5. Run 3-5 LLM prompts with different operationalizations of religious mission.
6. Aggregate labels using majority vote and a probabilistic weak-supervision label model; compare both.
7. Prioritize human review for high prompt disagreement, high LF conflict, near-threshold classifier scores, and high-loss training examples.

### Model Grid

| Family | Models | Role |
|---|---|---|
| Classical | TF-IDF word+char n-grams with logistic regression or SVM | Cheap baseline and interpretability check. |
| Frozen embeddings | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` plus logistic regression | Strong low-cost baseline and active-learning support. |
| Encoder fine-tuning | DistilBERT, RoBERTa-base, DeBERTa-v3-base, ELECTRA-base | Main supervised benchmark set. |
| Domain adaptation | TAPT/DAPT on unlabeled mission text | Secondary experiment if unlabeled text is large. |
| Small open-weight LMs | Gemma/Phi/Qwen/Llama-style 1-8B prompting or LoRA | Comparison arm, not default production path. |

Initial fine-tuning grid: learning rate `{1e-5, 2e-5, 3e-5, 5e-5}`, epochs `{3, 5, 8}` with early stopping, batch `{16, 32}`, max length `{128, 256}`, at least 5 seeds, weighted and unweighted cross-entropy, focal loss only if imbalance requires it.

### Evaluation Bundle

Report the following for each model and threshold:

| Category | Required outputs |
|---|---|
| Base rates | Train, validation, test, and deployment prevalence. |
| Threshold metrics | Confusion matrix, precision, recall, F1/F-beta, balanced accuracy, MCC. |
| Ranking metrics | PR-AUC/average precision as primary, ROC-AUC as secondary. |
| Calibration | Brier score, log-loss, reliability curves, ECE, subgroup calibration. |
| Threshold analysis | Predicted-positive rate, FP/1,000, FN/1,000, expected cost. |
| Social measurement | Prevalence estimate, adjusted prevalence, confidence intervals, validation-label sampling design. |
| External validity | Errors by organization type, NTEE group, tradition, geography, year, source, text length. |

### Calibration and Prevalence

Use held-out expert labels for post-hoc calibration. Compare no calibration, Platt/logistic scaling, temperature scaling, and isotonic regression.

For population estimates, prefer averaging calibrated probabilities over threshold-counting. Report thresholded classify-and-count only as a sensitivity check. If hard-label prevalence is needed, include adjusted classify-and-count using validation-estimated TPR and FPR, with uncertainty intervals.

For downstream regressions, avoid treating predicted hard labels as observed truth. Use validation-based correction, prediction-powered inference, or explicit misclassification sensitivity analysis.

## Main Gaps and Risks

| Gap/risk | Consequence | Mitigation |
|---|---|---|
| Few papers directly classify religious mission from short nonprofit text | Methods must be adapted from adjacent nonprofit, political-text, and survey-text literatures | Build a project-specific gold audit set and document construct decisions. |
| NTEE X is incomplete and single-purpose | Administrative labels can miss multi-mission or non-filing religious organizations | Treat NTEE as one noisy label source, not ground truth. |
| Churches often do not file Form 990 | Form 990 based data undercovers congregations | Use BMF, National Congregations Study, ARDA/ICPSR, denominational directories, and external benchmarks where feasible. |
| Religious language is tradition- and culture-specific | Keyword/rule systems can encode Christian or English-language bias | Audit by religious tradition and expand examples beyond Christian terminology. |
| LLM labels can shift with prompt and model version | Apparent agreement may hide construct drift | Use multiple prompts, log versions, preserve raw responses, and validate against human labels. |
| Imbalance can hide poor positive-class performance | Accuracy and ROC-AUC may look strong despite weak precision or recall | Make PR-AUC, precision/recall, MCC, and threshold curves primary. |
| Calibration may fail under subgroup/domain shift | Prevalence estimates become biased | Use subgroup calibration, stratified audits, and sensitivity checks. |

## Immediate Next Research Steps

1. Convert the domain recommendations into a coding codebook for religious mission labels.
2. Assemble candidate validation sources: IRS Form 990 text, BMF/NCCS NTEE, 1023-EZ purpose codes, National Congregations Study, ARDA/ICPSR, Candid/GuideStar if available.
3. Build the first audit sample with random, positive-enriched, near-threshold, and religious-token strata.
4. Implement weak labeling functions and LLM prompt labels with abstention/disagreement tracking.
5. Run the model grid and choose finalists using PR-AUC, positive-class precision/recall, calibration, and prevalence error.
6. Estimate final population prevalence using calibrated probabilities and report uncertainty from manual-label validation.
