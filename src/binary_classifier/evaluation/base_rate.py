"""Base-rate-adjusted precision diagnostics for stage 07."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from binary_classifier.evaluation.thresholds import pick_threshold
from binary_classifier.prevalence.weights import design_weights


def precision_at_base_rate(tpr: float, fpr: float, pi: float) -> float:
    """Adjust precision to a target population base rate."""
    denominator = tpr * pi + fpr * (1.0 - pi)
    if denominator == 0.0:
        return 0.0
    return float((tpr * pi) / denominator)


def base_rate_report(
    anchor_oof: pd.DataFrame,
    *,
    operating_threshold: float,
    max_f1_threshold: float,
    operating_pr_curve_points: list[dict[str, float]] | None = None,
    target: float,
    population_base_rate: float | None,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    """Build the base-rate precision report from saved anchor OOF probabilities."""
    frame = _classifier_rows(anchor_oof)
    probs = pd.to_numeric(frame["prob_calibrated_oof"], errors="coerce").to_numpy(float)
    labels = pd.to_numeric(frame["human_label"], errors="coerce").to_numpy(int)
    weights = design_weights(frame, normalize=True).to_numpy(float)
    pi = _population_base_rate(labels, weights, population_base_rate)

    # Use operating threshold's PR-curve grid when available to ensure the
    # candidate thresholds are aligned with the operating point selection.
    # Otherwise fall back to deriving from scratch.
    if operating_pr_curve_points is not None:
        candidate_thresholds = [p["threshold"] for p in operating_pr_curve_points]
    else:
        threshold_report = pick_threshold(probs, labels, "max_f1", 0.0)
        candidate_thresholds = [
            p["threshold"] for p in threshold_report["pr_curve_points"]
        ]

    points = [
        _threshold_point("operating", operating_threshold, probs, labels, pi, None),
        _threshold_point("max_f1", max_f1_threshold, probs, labels, pi, None),
        _threshold_point("operating", operating_threshold, probs, labels, pi, weights),
        _threshold_point("max_f1", max_f1_threshold, probs, labels, pi, weights),
    ]

    candidates = [
        _base_rate_point(t, probs, labels, pi, weights) for t in candidate_thresholds
    ]
    eligible = [p for p in candidates if p["base_rate_precision"] >= target]
    unattainable = not eligible
    selected = (
        min(eligible, key=lambda p: p["threshold"])
        if eligible
        else max(
            candidates,
            key=lambda p: (p["base_rate_precision"], p["recall"], p["threshold"]),
        )
    )
    selected = dict(selected)
    selected["ci"] = _bootstrap_ci(
        probs,
        labels,
        pi,
        weights,
        float(selected["threshold"]),
        seed=seed,
        n_resamples=n_resamples,
    )
    return {
        "population_base_rate": pi,
        "population_base_rate_source": "config"
        if population_base_rate is not None
        else "weighted_anchor_fallback",
        "target_precision": float(target),
        "unattainable": unattainable,
        "threshold": float(selected["threshold"]),
        "selected": selected,
        "points": points,
        "candidate_points": candidates,
    }


def _classifier_rows(anchor_oof: pd.DataFrame) -> pd.DataFrame:
    required = {"prob_calibrated_oof", "human_label", "sample_prob"}
    missing = required - set(anchor_oof.columns)
    if missing:
        raise ValueError(f"anchor OOF scores missing columns: {sorted(missing)}")
    frame = anchor_oof.copy()
    if "decision_source" in frame.columns:
        source = frame["decision_source"].astype(str).str.lower()
        classifier = source.str.contains("classifier")
        if classifier.any():
            frame = frame.loc[classifier].copy()
    if frame.empty:
        raise ValueError("No anchor rows available for base-rate precision.")
    return frame.reset_index(drop=True)


def _population_base_rate(
    labels: np.ndarray,
    weights: np.ndarray,
    configured: float | None,
) -> float:
    if configured is not None:
        if not 0.0 < configured < 1.0:
            raise ValueError("evaluation.population_base_rate must be between 0 and 1")
        return float(configured)
    return float(np.average(labels, weights=weights))


def _threshold_point(
    name: str,
    threshold: float,
    probs: np.ndarray,
    labels: np.ndarray,
    pi: float,
    weights: np.ndarray | None,
) -> dict[str, Any]:
    point = _base_rate_point(threshold, probs, labels, pi, weights)
    point["name"] = name
    point["weighted"] = weights is not None
    return point


def _base_rate_point(
    threshold: float,
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    pi: float,
    weights: Sequence[float] | np.ndarray | None,
) -> dict[str, Any]:
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    pred = probs_arr >= threshold
    if weights is None:
        w = np.ones(len(labels_arr), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    pos = labels_arr == 1
    neg = labels_arr == 0
    tpr = _safe_divide(float(w[pos & pred].sum()), float(w[pos].sum()))
    fpr = _safe_divide(float(w[neg & pred].sum()), float(w[neg].sum()))
    return {
        "threshold": float(threshold),
        "tpr": tpr,
        "fpr": fpr,
        "recall": tpr,
        "base_rate_precision": precision_at_base_rate(tpr, fpr, pi),
    }


def _bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    pi: float,
    weights: np.ndarray,
    threshold: float,
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, float | None]:
    if n_resamples <= 0:
        return {"lower": None, "upper": None}
    rng = np.random.default_rng(seed=seed)
    values = []
    n = len(labels)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        values.append(
            _base_rate_point(threshold, probs[idx], labels[idx], pi, weights[idx])[
                "base_rate_precision"
            ],
        )
    if not values:
        return {"lower": None, "upper": None}
    return {
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else float(numerator / denominator)
