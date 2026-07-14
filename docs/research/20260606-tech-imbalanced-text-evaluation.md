---
created: 20260606
agent: tech-seeker
scratchpad: docs/research/notebooks/20260606-tech-imbalanced-text-evaluation-scratchpad.md
status: complete
title: Tech - Imbalanced Text Evaluation
topic: imbalanced text classification; evaluation metrics; validation design; social science measurement; noisy labels
---

# Tech: Imbalanced Text Classification Evaluation for Social Science

## From Replication Packages

Upstream: `docs/research/20260605-replication-imbalanced-validation.md`.

| Paper | Language | Key Files | Method | Source URL |
|-------|----------|-----------|--------|-----------|
| Hopkins & King (2010) / ReadMe | R | `R/prototype.R`, `demo/clinton.R` | Aggregate prevalence / nonparametric content analysis | https://github.com/iqss-research/ReadMeV1 |
| ValiText | R/Shiny | `R/app_ui.R`, `R/app_server.R` | Validation checklist app for text-based measures | https://github.com/lukasbirki/ValiTex |
| WRENCH | Python | `wrench/evaluation.py` | Weak-supervision benchmark metrics | https://github.com/JieyuZ2/wrench |
| BOXWRENCH | Python/Jupyter | `end_model_training/pipelines.py`, `val_size_experiment.py` | Validation-size experiments for weak supervision | https://github.com/jeffreywpli/stronger-than-you-think |
| hmeasure | R | `R/library_metrics.R` | H-measure and cost-sensitive ROC alternatives | https://github.com/canagnos/hmeasure |
| precrec | R/C++ | `R/main_evalmod.R` | PR/ROC curves and AUC metrics | https://github.com/evalclass/precrec |
| Brodersen/micp | R | `R/micp.stats.R` | Bayesian inference for balanced classification accuracy | https://github.com/kaybrodersen/micp |

## Package Recommendations

| Language | Package | Version | Key Function | Source |
|----------|---------|---------|-------------|--------|
| Python | scikit-learn | current stable | `precision_recall_fscore_support`, `average_precision_score`, `roc_auc_score`, `balanced_accuracy_score`, `matthews_corrcoef`, `cohen_kappa_score`, `confusion_matrix` | https://scikit-learn.org/stable/modules/model_evaluation.html |
| Python | QuaPy | current stable / active OSS | Quantification / supervised prevalence estimation protocols | https://github.com/HLT-ISTI/QuaPy |
| R | precrec | CRAN/current | PR and ROC curves, AUC, multi-run evaluation | https://evalclass.github.io/precrec/ |
| R | hmeasure | CRAN/GitHub | H-measure for cost-sensitive evaluation | https://github.com/canagnos/hmeasure |
| R | irr / krippendorff | CRAN/current | Cohen's kappa, Krippendorff's alpha for hand-coded validation sets | https://cran.r-project.org/ |

## Implementation Examples

### Example 1: Multi-metric classifier report

**Source**: Davis & Goadrich (2006), Saito & Rehmsmeier (2015), Chicco & Jurman (2020), Park & Montgomery (2025).  
**Package version**: scikit-learn current stable.

```python
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, average_precision_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
)

y_score = model.predict_proba(X_test)[:, 1]
y_pred = (y_score >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=[0, 1], zero_division=0
)
report = {
    "confusion_matrix": cm,
    "minority_precision": precision[1],
    "minority_recall": recall[1],
    "minority_f1": f1[1],
    "macro_f1": precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)[2],
    "pr_auc_average_precision": average_precision_score(y_test, y_score),
    "roc_auc": roc_auc_score(y_test, y_score),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "mcc": matthews_corrcoef(y_test, y_pred),
    "cohen_kappa": cohen_kappa_score(y_test, y_pred),
}
```

**Notes**: Do not report accuracy alone. Treat ROC-AUC as secondary under rare positives; pair PR-AUC with thresholded precision/recall/Fβ and explicit threshold selection.

### Example 2: Population prevalence estimate rather than classify-and-count

**Source**: Hopkins & King (2010); Forman (2005, 2008); Bella et al. (2010); Keith & O'Connor (2018).  
**Package version**: QuaPy current stable or custom validation-sample correction.

```python
# Conceptual adjusted-count correction for binary prevalence.
# Estimate TPR/FPR on a gold validation set representative of the target domain.
raw_prev = y_pred_target.mean()
adjusted_prev = (raw_prev - fpr_validation) / (tpr_validation - fpr_validation)
adjusted_prev = min(1.0, max(0.0, adjusted_prev))
```

**Notes**: Use quantification methods or validation-sample misclassification correction when the downstream estimand is the share of organizations in the positive class. Report uncertainty intervals; check assumptions under domain/covariate shift.

### Example 3: Validation design for noisy/weak labels

**Source**: Song et al. (2020); Birkenmaier et al. (2023/2024); Park & Montgomery (2025); BOXWRENCH (2025); AlleNoise (2025).  
**Package version**: N/A.

```text
1. Separate weak/LLM labels used for training from expert-coded validation labels.
2. Stratify validation by model score bands and predicted class; oversample likely positives.
3. Double-code a subset; report coder count, codebook, adjudication, Krippendorff alpha/kappa.
4. Evaluate final model on untouched gold set; report confusion matrix and minority metrics.
5. Audit false positives/false negatives qualitatively for construct-validity failures.
6. Revalidate on new text sources/time periods before population inference.
```

**Notes**: Human labels are not automatically gold; low intercoder reliability propagates into model evaluation and downstream estimates.

## Version and Compatibility Notes

- Metric definitions vary: PR-AUC may mean average precision or trapezoidal PR-AUC. State the implementation.
- PR-AUC is prevalence-sensitive; compare against a baseline equal to positive prevalence.
- ROC-AUC can look strong under extreme imbalance while precision is poor; never use it as the sole headline metric for rare-class detection.
- MCC and balanced accuracy summarize all confusion-matrix cells but can still shift with prevalence/reference-standard error; report the confusion matrix.
- Cohen's kappa is useful for coder agreement, but for classifier performance it can be prevalence- and marginal-distribution-sensitive; use cautiously.
- Cross-validation is not a substitute for an externally hand-coded audit set when labels are weak/LLM-generated or when deployment domain differs.
- For population measurement, classifier accuracy metrics do not guarantee unbiased prevalence estimates; use quantification/misclassification correction and uncertainty intervals.
