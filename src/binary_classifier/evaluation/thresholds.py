"""Threshold selection for binary classification decisions.

This module selects operating thresholds from validation probabilities,
supporting precision-floor and max-F1 policies.  Threshold selection is
performed after calibration and before final evaluation, ensuring the
deployed threshold satisfies the project's precision requirements.

Data provenance
    Thresholds are derived from validation-set probabilities (calibrated) and
    human labels, then applied to the held-out test set.

Methodology
    Candidate thresholds are the unique observed probabilities.  Under the
    precision-floor policy, the selector takes the highest-recall threshold
    whose precision meets the floor; if no candidate meets the floor, it falls
    back to the maximum-precision threshold.  This framing follows Forman
    (2008), who treats threshold choice as an explicit cost/trade-off decision.

Key citations
    * Forman (2008) — Quantifying counts and costs via classification.

DOI
    * Forman (2008): https://doi.org/10.1007/s10618-008-0097-y
"""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

ThresholdPolicy = Literal["precision_floor", "max_f1"]


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
# Candidate thresholds are the unique observed probabilities.  The
# precision-floor policy selects the highest-recall threshold whose precision
# meets the floor, falling back to the maximum-precision threshold if the
# floor is unattainable.  The max-F1 policy simply selects the threshold that
# maximizes F1.  This follows Forman (2008), who treats threshold choice as
# an explicit cost/trade-off decision rather than a fixed 0.5 cutoff.
# ---------------------------------------------------------------------------


def pick_threshold(
    probs: Sequence[float],
    labels: Sequence[int],
    policy: ThresholdPolicy,
    precision_floor: float,
) -> dict[str, Any]:
    """Pick a decision threshold from validation probabilities.

    The positive/minority class is label ``1``. Candidate thresholds are the
    unique observed probabilities, using ``probability >= threshold`` as the
    positive decision rule. Under the precision-floor policy, the selector takes
    the highest-recall threshold whose precision meets the floor. If no
    candidate meets the floor, it falls back to the threshold with the maximum
    achieved precision and sets ``floor_unattainable`` to ``True``.

    This framing follows Forman (2008), who treats threshold choice as an
    explicit cost/trade-off decision rather than a fixed ``0.5`` cutoff.

    Args:
        probs: Positive-class probabilities aligned to ``labels``.
        labels: Binary ground-truth labels where ``1`` is the positive class.
        policy: Threshold policy, either ``"precision_floor"`` or ``"max_f1"``.
        precision_floor: Required precision for the precision-floor policy.

    Returns:
        Serializable threshold-selection report with the selected threshold,
        achieved precision/recall, max-F1 threshold, and PR curve points.

    Raises:
        ValueError: If inputs are empty, misaligned, non-binary, non-finite, or
            if ``policy``/``precision_floor`` are invalid.
    """
    probs_arr, labels_arr = _validate_inputs(probs, labels)
    if policy not in {"precision_floor", "max_f1"}:
        raise ValueError("policy must be 'precision_floor' or 'max_f1'.")
    if not 0.0 <= precision_floor <= 1.0:
        raise ValueError("precision_floor must be between 0 and 1.")

    points = _pr_curve_points(probs_arr, labels_arr)
    max_f1_point = _best_point(points, key="f1")

    floor_unattainable = False
    if policy == "max_f1":
        selected = max_f1_point
    else:
        eligible = [p for p in points if p["precision"] >= precision_floor]
        if eligible:
            selected = _best_point(eligible, key="recall")
        else:
            selected = _best_point(points, key="precision")
            floor_unattainable = True

    return {
        "threshold": selected["threshold"],
        "policy": policy,
        "achieved_precision": selected["precision"],
        "achieved_recall": selected["recall"],
        "max_f1_threshold": max_f1_point["threshold"],
        "pr_curve_points": points,
        "floor_unattainable": floor_unattainable,
    }


def _validate_inputs(
    probs: Sequence[float],
    labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    if probs_arr.ndim != 1 or labels_arr.ndim != 1:
        raise ValueError("probs and labels must be one-dimensional.")
    if len(probs_arr) == 0:
        raise ValueError("probs and labels must not be empty.")
    if len(probs_arr) != len(labels_arr):
        raise ValueError("probs and labels must have the same length.")
    if not np.isfinite(probs_arr).all():
        raise ValueError("probs must be finite.")
    if ((probs_arr < 0.0) | (probs_arr > 1.0)).any():
        raise ValueError("probs must be between 0 and 1.")
    if not np.isin(labels_arr, [0, 1]).all():
        raise ValueError("labels must be binary 0/1 values.")
    return probs_arr, labels_arr


def _pr_curve_points(
    probs: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for threshold in sorted(np.unique(probs)):
        y_pred = probs >= threshold
        tp = int(np.sum((labels == 1) & y_pred))
        fp = int(np.sum((labels == 0) & y_pred))
        fn = int(np.sum((labels == 1) & ~y_pred))

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        points.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return points


def _best_point(
    points: Sequence[dict[str, float]],
    *,
    key: Literal["precision", "recall", "f1"],
) -> dict[str, float]:
    """Return deterministic best point, breaking ties toward safer thresholds."""
    return max(
        points,
        key=lambda p: (
            p[key],
            p["recall"],
            p["precision"],
            p["threshold"],
        ),
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)
