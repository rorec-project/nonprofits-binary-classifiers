"""Subgroup diagnostics for binary evaluation reports.

This module computes per-subgroup error rates and small-cell suppression
flags for evaluation reports.  It supports grouping by categorical columns
(e.g. NTEE major group) and by word-count bins.

Data provenance
    Subgroup reports are generated from the evaluation DataFrame (which
carries ``EIN2``, ``text``, and any grouping columns) together with the
    aligned ground-truth labels, predicted labels, and probabilities.

Methodology
    For each subgroup, we compute minority-class F1, false-positive rate
    (FPR), and false-negative rate (FNR).  Subgroups with fewer than
    ``min_n`` rows are suppressed: the subgroup identifier and ``n`` are
    retained, but metric values are set to ``None``.  This prevents
    unreliable point estimates from driving conclusions.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Subgroup diagnostics
# ---------------------------------------------------------------------------
# We compute per-subgroup minority-F1, FPR, and FNR for categorical
# groupings (e.g. NTEE major group) and word-count bins.  Subgroups with
# fewer than ``min_n`` rows are suppressed: the identifier and ``n`` are
# retained, but metric values are set to ``None``.  This prevents unreliable
# point estimates from driving conclusions.
# ---------------------------------------------------------------------------


def subgroup_report(
    df: pd.DataFrame,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
    *,
    by: str | Sequence[str],
    length_bins: Sequence[int],
    min_n: int,
) -> list[dict[str, Any]]:
    """Compute report-only subgroup metrics with small-cell suppression.

    Rows are produced for each requested grouping column and for word-count bins.
    The positive/minority class is label ``1``. Suppressed rows keep the subgroup
    identifier and ``n`` but set metric values to ``None``.

    Args:
        df: Evaluation rows aligned to the label/probability arrays. ``EIN2`` is
            preserved only as row-level input; report rows retain subgroup
            identifiers rather than individual records.
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Positive-class probabilities, used for alignment validation.
        by: Column name or column names to summarize, e.g. an NTEE major group.
        length_bins: Upper bounds for word-count bins, e.g. ``[10, 25, 50]``.
        min_n: Minimum subgroup size required before reporting metrics.

    Returns:
        List of serializable subgroup rows with ``n``, minority-F1, FPR, FNR, and
        a ``suppressed`` flag.

    Raises:
        ValueError: If inputs are misaligned, invalid, or length bins are not
            strictly increasing.
        KeyError: If a requested grouping column is missing.

    """
    y_true_arr, y_pred_arr = _validate_aligned(df, y_true, y_pred, y_prob)
    if min_n < 1:
        raise ValueError("min_n must be at least 1.")
    bins = _validate_length_bins(length_bins)

    rows: list[dict[str, Any]] = []
    for column in _as_columns(by):
        if column not in df.columns:
            raise KeyError(f"Grouping column not found: {column}")
        values = df[column].astype("string").fillna("<missing>")
        rows.extend(_rows_for_groups(column, values, y_true_arr, y_pred_arr, min_n))

    if bins:
        word_counts = _word_counts(df)
        labels = [_length_bin_label(count, bins) for count in word_counts]
        rows.extend(
            _rows_for_groups(
                "word_count_bin",
                pd.Series(labels, index=df.index, dtype="string"),
                y_true_arr,
                y_pred_arr,
                min_n,
            ),
        )

    return rows


def _validate_aligned(
    df: pd.DataFrame,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    lengths = {len(df), len(y_true_arr), len(y_pred_arr), len(y_prob_arr)}
    if len(lengths) != 1:
        raise ValueError("df, y_true, y_pred, and y_prob must have equal lengths.")
    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1 or y_prob_arr.ndim != 1:
        raise ValueError("y_true, y_pred, and y_prob must be one-dimensional.")
    if not np.isin(y_true_arr, [0, 1]).all():
        raise ValueError("y_true must contain binary 0/1 labels.")
    if not np.isin(y_pred_arr, [0, 1]).all():
        raise ValueError("y_pred must contain binary 0/1 labels.")
    if not np.isfinite(y_prob_arr).all():
        raise ValueError("y_prob must be finite.")
    return y_true_arr, y_pred_arr


def _validate_length_bins(length_bins: Sequence[int]) -> list[int]:
    bins = [int(edge) for edge in length_bins]
    if any(edge < 0 for edge in bins):
        raise ValueError("length_bins must be non-negative.")
    if any(left >= right for left, right in zip(bins, bins[1:], strict=False)):
        raise ValueError("length_bins must be strictly increasing.")
    return bins


def _as_columns(by: str | Sequence[str]) -> list[str]:
    if isinstance(by, str):
        return [by]
    return list(by)


def _rows_for_groups(
    grouping: str,
    values: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(values.unique()):
        mask = values.to_numpy() == value
        n = int(np.sum(mask))
        row: dict[str, Any] = {
            "grouping": grouping,
            "value": str(value),
            "n": n,
            "suppressed": n < min_n,
        }
        if n < min_n:
            row.update({"minority_f1": None, "fpr": None, "fnr": None})
        else:
            row.update(_rates(y_true[mask], y_pred[mask]))
        rows.append(row)
    return rows


def _word_counts(df: pd.DataFrame) -> list[int]:
    if "word_count" in df.columns:
        return [int(count) for count in df["word_count"]]
    if "text" not in df.columns:
        raise KeyError("df must contain 'word_count' or 'text' for length bins.")
    return [len(str(text).split()) for text in df["text"]]


def _length_bin_label(count: int, bins: Sequence[int]) -> str:
    lower = 0
    for upper in bins:
        if count <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f"{bins[-1] + 1}+"


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Compute minority-class F1, FPR, and FNR from the confusion matrix.
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "minority_f1": _safe_divide(2.0 * precision * recall, precision + recall),
        "fpr": _safe_divide(fp, fp + tn),
        "fnr": _safe_divide(fn, fn + tp),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)
