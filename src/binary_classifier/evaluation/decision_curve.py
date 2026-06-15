"""Decision-curve analysis helpers for report-only evaluation."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def net_benefit(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Compute Vickers-Elkin net-benefit points for binary predictions.

    Net benefit at threshold ``t`` is ``TP / n - FP / n * t / (1 - t)``. Each
    point also includes treat-all and treat-none reference lines.

    Args:
        y_true: Ground-truth binary labels where ``1`` is the positive class.
        y_prob: Positive-class probabilities aligned to ``y_true``.
        thresholds: Decision thresholds in the open interval ``(0, 1)``.

    Returns:
        Serializable decision-curve points.

    Raises:
        ValueError: If inputs are empty, misaligned, non-binary, non-finite, or
            if thresholds are outside ``(0, 1)``.
    """
    y_true_arr, y_prob_arr, threshold_arr = _validate_inputs(y_true, y_prob, thresholds)
    n = len(y_true_arr)
    prevalence = float(np.mean(y_true_arr == 1))

    points: list[dict[str, Any]] = []
    for threshold in threshold_arr:
        y_pred = y_prob_arr >= threshold
        tp = int(np.sum((y_true_arr == 1) & y_pred))
        fp = int(np.sum((y_true_arr == 0) & y_pred))
        odds = float(threshold / (1.0 - threshold))
        points.append(
            {
                "threshold": float(threshold),
                "net_benefit": float(tp / n - (fp / n) * odds),
                "treat_all_net_benefit": prevalence - (1.0 - prevalence) * odds,
                "treat_none_net_benefit": 0.0,
                "tp": tp,
                "fp": fp,
                "n": n,
            }
        )
    return points


def _validate_inputs(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    threshold_arr = np.asarray(thresholds, dtype=float)
    if y_true_arr.ndim != 1 or y_prob_arr.ndim != 1 or threshold_arr.ndim != 1:
        raise ValueError("y_true, y_prob, and thresholds must be one-dimensional.")
    if len(y_true_arr) == 0:
        raise ValueError("y_true and y_prob must not be empty.")
    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError("y_true and y_prob must have the same length.")
    if not np.isin(y_true_arr, [0, 1]).all():
        raise ValueError("y_true must contain binary 0/1 labels.")
    if not np.isfinite(y_prob_arr).all():
        raise ValueError("y_prob must be finite.")
    if ((y_prob_arr < 0.0) | (y_prob_arr > 1.0)).any():
        raise ValueError("y_prob must be between 0 and 1.")
    if len(threshold_arr) == 0:
        raise ValueError("thresholds must not be empty.")
    if not np.isfinite(threshold_arr).all():
        raise ValueError("thresholds must be finite.")
    if ((threshold_arr <= 0.0) | (threshold_arr >= 1.0)).any():
        raise ValueError("thresholds must be in the open interval (0, 1).")
    return y_true_arr, y_prob_arr, threshold_arr
