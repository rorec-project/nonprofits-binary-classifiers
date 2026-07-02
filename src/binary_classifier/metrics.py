"""Shared metric helpers for binary classifier validation.

This module computes the full binary-classification metric bundle used in
the quality-check gate (stage 04) and final evaluation (stage 07). It
includes confusion-matrix counts, minority-class precision/recall/F1, MCC,
balanced accuracy, Cohen's kappa, Krippendorff's alpha, PR-AUC, ROC-AUC, and
bootstrap confidence intervals.

Metrics are computed from human-coded validation or test labels and the
corresponding silver-label predictions or model probabilities. The MCC
(Chicco & Jurman, 2020) is reported as the primary agreement metric because
it is informative even under strong class imbalance. PR-AUC and ROC-AUC are
computed when confidence scores are available; PR-AUC is preferred for
imbalanced settings (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015).
Bootstrap CIs for accuracy and minority-F1 support the variance-aware freeze
gate. Threshold choice and expected-loss metrics follow Hernandez-Orallo et
al. (2012).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metric_bundle(
    y_true,
    y_pred,
    *,
    y_score=None,
    minority_class: int = 1,
    seed: int = 42,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> dict:
    """Compute the full binary-classification metric bundle.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels aligned to ``y_true``.
        y_score: Optional score for the positive class. When provided, PR-AUC
            and ROC-AUC are computed on non-missing scores when possible.
            PR-AUC is preferred for imbalanced settings (Davis & Goadrich,
            2006; Saito & Rehmsmeier, 2015).
        minority_class: Class whose precision, recall, F1, and bootstrap F1 CI
            should be reported.
        seed: Seed for bootstrap resampling.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g. 0.95).

    Returns:
        Dict with confusion-matrix counts, minority-class precision/recall/F1,
        MCC, balanced accuracy, Cohen's κ, Krippendorff's α, PR-AUC,
        bootstrap CIs, and ``roc_auc`` when ``y_score`` is provided.

    Raises:
        ValueError: If label vectors or score vectors have inconsistent lengths.

    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length.")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    # Precision, recall, F1 for both classes; report minority explicitly
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        zero_division=0,
    )
    precision = float(precisions[minority_class])
    recall = float(recalls[minority_class])
    f1 = float(f1s[minority_class])

    mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    kappa = float(cohen_kappa_score(y_true_arr, y_pred_arr))
    alpha = _krippendorff_alpha_nominal_two_raters(y_true_arr, y_pred_arr)

    # PR-AUC and ROC-AUC when confidence scores exist.
    pr_auc = None
    roc_auc = None
    if y_score is not None:
        scores = np.asarray(y_score, dtype=float)
        if len(scores) != len(y_true_arr):
            raise ValueError("y_score must have the same length as y_true.")
        mask = ~np.isnan(scores)
        if mask.any():
            masked_true = y_true_arr[mask]
            masked_scores = scores[mask]
            pr_auc = float(average_precision_score(masked_true, masked_scores))
            if len(np.unique(masked_true)) == 2:
                roc_auc = float(roc_auc_score(masked_true, masked_scores))

    # Bootstrap CI
    ci = bootstrap_ci(
        y_true_arr,
        y_pred_arr,
        minority_class,
        seed=seed,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )

    bundle = {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "minority_class": minority_class,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": balanced_acc,
        "cohens_kappa": kappa,
        "krippendorff_alpha": alpha,
        "pr_auc": pr_auc,
        "bootstrap_ci": ci,
    }
    if y_score is not None:
        bundle["roc_auc"] = roc_auc
    return bundle


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
# We resample with replacement to produce percentile CIs for accuracy and
# minority-class F1.  The minority-F1 lower bound is the variance-aware half
# of the freeze gate, preventing a high point estimate from freezing labels
# when validation uncertainty remains too wide.
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true,
    y_pred,
    minority_class: int,
    seed: int,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for accuracy and minority-class F1.

    The minority-F1 lower bound is the variance-aware half of the freeze gate,
    preventing a high point estimate from freezing labels when validation
    uncertainty remains too wide.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels aligned to ``y_true``.
        minority_class: The class to report F1 for.
        seed: Seed for bootstrap resampling.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g. 0.95).

    Returns:
        Dict with ``accuracy`` and ``minority_f1`` each containing ``lower`` and
        ``upper``.

    Raises:
        ValueError: If label vectors have inconsistent lengths.

    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length.")

    rng = np.random.default_rng(seed=seed)
    n = len(y_true_arr)
    accs = np.empty(n_resamples, dtype=float)
    f1s = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        accs[i] = float(np.mean(y_true_arr[idx] == y_pred_arr[idx]))
        f1s[i] = f1_score(
            y_true_arr[idx],
            y_pred_arr[idx],
            pos_label=minority_class,
            zero_division=0,
        )

    alpha = 1 - confidence_level
    lower_p = alpha / 2 * 100
    upper_p = (1 - alpha / 2) * 100

    return {
        "accuracy": {
            "lower": float(np.percentile(accs, lower_p)),
            "upper": float(np.percentile(accs, upper_p)),
        },
        "minority_f1": {
            "lower": float(np.percentile(f1s, lower_p)),
            "upper": float(np.percentile(f1s, upper_p)),
        },
    }


# ---------------------------------------------------------------------------
# Inter-rater agreement
# ---------------------------------------------------------------------------
# Cohen's κ is the standard pairwise agreement metric for two complete raters.
# Krippendorff's α is reported alongside κ because it extends naturally to
# missing/abstaining annotators if the gate later admits those cells.
# ---------------------------------------------------------------------------


def _krippendorff_alpha_nominal_two_raters(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute nominal Krippendorff's alpha for two validation label vectors.

    The validation gate currently compares pairwise-complete human and silver
    labels, but alpha is reported alongside κ because it extends naturally to
    missing/abstaining annotators if the gate later admits those cells
    (Krippendorff's alpha).

    Args:
        y_true: Human validation labels.
        y_pred: Silver labels aligned to ``y_true``.

    Returns:
        Krippendorff's alpha for nominal labels, or ``nan`` when no labels are
        available.

    """
    if len(y_true) == 0:
        return float("nan")

    observed_disagreement = float(np.mean(y_true != y_pred))
    pooled = np.concatenate([y_true, y_pred])
    labels = np.unique(pooled)

    # For two complete raters, nominal alpha is 1 minus observed disagreement over
    # chance disagreement computed from the pooled label distribution.
    expected_agreement = sum(
        float((np.sum(pooled == label) / len(pooled)) ** 2) for label in labels
    )
    expected_disagreement = 1 - expected_agreement
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else 0.0
    return float(1 - observed_disagreement / expected_disagreement)
