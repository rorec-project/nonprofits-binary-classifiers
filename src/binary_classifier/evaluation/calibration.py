"""Post-hoc probability calibration for anchor-set model scores.

This module implements post-hoc calibration methods that transform raw
positive-class probabilities into calibrated probabilities suitable for
population-prevalence estimation (PPI++).  It is consumed by the stage-07
evaluation and prevalence pipelines.

Data provenance
    Calibrators are fit on human-coded anchor labels (``anchor_to_code.csv``)
    and applied to silver-label probabilities produced by the selected
    production slate.

Methodology
    We compare Platt (logistic) scaling and temperature scaling on a
    stratified cross-fit, selecting the winner by mean out-of-fold Brier score
    with log-loss as tie-breaker.  Platt scaling is preferred as the default
    because its intercept can absorb sample-enrichment or prior-shift offsets,
    whereas temperature scaling is deliberately intercept-free and serves as a
    comparison method.  Isotonic regression is excluded from the default
    comparison because it can overfit small calibration sets (Zadrozny &
    Elkan, 2002).

Key citations
    * Platt (1999) — Probabilistic outputs for SVMs.
    * Zadrozny & Elkan (2002) — Transforming classifier scores into accurate
      multiclass probability estimates.
    * Guo et al. (2017) — On calibration of modern neural networks.
    * Desai & Durrett (2020) — Calibration of pre-trained transformers.
    * Silva Filho et al. (2023) — Classifier calibration: a survey.

DOIs
    * Zadrozny & Elkan (2002): https://doi.org/10.1145/775047.775151
    * Guo et al. (2017): https://proceedings.mlr.press/v70/guo17a.html
    * Desai & Durrett (2020): https://aclanthology.org/2020.emnlp-main.21/
    * Silva Filho et al. (2023): https://doi.org/10.1007/s10994-023-06336-7
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

CalibrationMethod = Literal["platt", "temperature"]
CalibrationParams = dict[str, float]
ScoreInput = Sequence[float] | NDArray[Any]
LabelInput = Sequence[int] | NDArray[Any]

_EPS = 1e-12
_VALID_METHODS: set[str] = {"platt", "temperature"}


# ---------------------------------------------------------------------------
# Post-hoc calibration fitting
# ---------------------------------------------------------------------------
# We fit two parametric families on the logit of raw probabilities.  Platt
# scaling (logistic regression on a single feature) has both slope and
# intercept; the intercept is the reason it is the default choice under
# sample-enrichment prior shifts (Platt, 1999; Zadrozny & Elkan, 2002).
# Temperature scaling is a scalar divisor with no intercept, so it can only
# stretch or compress the logit scale and cannot correct a prior shift
# (Guo et al., 2017; Desai & Durrett, 2020).  It is included as a comparison
# method in the cross-fit bake-off.
# ---------------------------------------------------------------------------


def fit_platt(scores: ScoreInput, labels: LabelInput) -> CalibrationParams:
    """Fit Platt scaling on the logit of input probabilities.

    The fitted model is ``sigmoid(a * logit(score) + b)``. The intercept ``b``
    is the parameter that can absorb sample-enrichment or prior-shift offsets,
    which is why Platt scaling is the default calibrator in this pipeline
    (Platt, 1999; Zadrozny & Elkan, 2002).

    Args:
        scores: Raw positive-class probabilities.
        labels: Binary human labels aligned to ``scores``.

    Returns:
        JSON-serializable parameters ``{"a": a, "b": b}``.

    Raises:
        ValueError: If inputs are malformed or contain fewer than two classes.

    """
    score_arr, label_arr = _validated_scores_labels(scores, labels)
    _require_two_classes(label_arr)

    feature = _logit(score_arr).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", random_state=0)
    model.fit(feature, label_arr)
    return {
        "a": float(model.coef_[0, 0]),
        "b": float(model.intercept_[0]),
    }


def fit_temperature(scores: ScoreInput, labels: LabelInput) -> CalibrationParams:
    """Fit scalar temperature scaling on the logit of input probabilities.

    The fitted model is ``sigmoid(logit(score) / T)``. It deliberately has no
    intercept, so it cannot correct a prior shift and is treated as a comparison
    method rather than the default choice under anchor-set prior shifts
    (Guo et al., 2017).

    Args:
        scores: Raw positive-class probabilities.
        labels: Binary human labels aligned to ``scores``.

    Returns:
        JSON-serializable parameters ``{"T": T}``.

    Raises:
        ValueError: If inputs are malformed or contain fewer than two classes.

    """
    score_arr, label_arr = _validated_scores_labels(scores, labels)
    _require_two_classes(label_arr)
    logits = _logit(score_arr)

    def objective(temperature: float) -> float:
        if temperature <= 0 or not math.isfinite(temperature):
            return math.inf
        calibrated = _sigmoid(logits / temperature)
        return compute_log_loss(label_arr, calibrated)

    temperature = _minimize_scalar(objective, lower=0.05, upper=20.0)
    return {"T": float(temperature)}


def apply_calibration(
    scores: ScoreInput,
    method: CalibrationMethod,
    params: Mapping[str, float],
) -> list[float]:
    """Apply fitted calibration parameters to raw probabilities.

    Args:
        scores: Raw positive-class probabilities.
        method: Calibration family, either ``"platt"`` or ``"temperature"``.
        params: Fitted parameter mapping from :func:`fit_platt` or
            :func:`fit_temperature`.

    Returns:
        Calibrated probabilities in the same order as ``scores``.

    Raises:
        ValueError: If scores, method, or parameters are invalid.

    """
    score_arr = _validated_scores(scores)
    logits = _logit(score_arr)
    if method == "platt":
        try:
            a = float(params["a"])
            b = float(params["b"])
        except KeyError as exc:
            raise ValueError("Platt params must contain 'a' and 'b'.") from exc
        calibrated = _sigmoid(a * logits + b)
    elif method == "temperature":
        try:
            temperature = float(params["T"])
        except KeyError as exc:
            raise ValueError("Temperature params must contain 'T'.") from exc
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("Temperature parameter 'T' must be positive and finite.")
        calibrated = _sigmoid(logits / temperature)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}.")
    return calibrated.astype(float).tolist()


# ---------------------------------------------------------------------------
# Cross-fit model selection
# ---------------------------------------------------------------------------
# We stratified-cross-fit each candidate calibrator so that every row is
# calibrated by parameters fit on *other* rows.  The winner is selected by
# mean out-of-fold Brier score (log-loss as tie-breaker) and then refit on
# the full anchor set for deployment.  Isotonic regression is omitted from
# the default slate because it can overfit small calibration sets (Zadrozny
# & Elkan, 2002; Silva Filho et al., 2023).
# ---------------------------------------------------------------------------


def crossfit_calibrate(
    scores: ScoreInput,
    labels: LabelInput,
    folds: int,
    methods: Sequence[CalibrationMethod],
    seed: int,
    *,
    ece_bins: int = 10,
) -> tuple[list[float], dict[str, Any]]:
    """Cross-fit calibration methods and select a deployed winner.

    For each method, every row is predicted only by a calibrator fit on rows from
    other folds (out-of-fold, OOF). The selected winner is the method with the
    lowest mean fold OOF Brier score, with mean fold OOF log-loss as the
    tiebreaker. The selected method is then refit on all anchor rows for
    deployment parameters.

    Isotonic regression is deliberately omitted from the default comparison set
    because it can overfit small calibration sets and produce non-monotonic
    artifacts (Zadrozny & Elkan, 2002; Silva Filho et al., 2023).  If the anchor
    set grows large enough, isotonic can be added as an additional candidate via
    ``methods``.

    Args:
        scores: Raw positive-class probabilities for anchor rows.
        labels: Binary human labels aligned to ``scores``.
        folds: Stratified cross-fit fold count.
        methods: Calibration methods to compare.
        seed: Random seed for deterministic fold assignment.
        ece_bins: Equal-width bin count for ECE and reliability points.

    Returns:
        A tuple ``(winner_oof_calibrated, report)``. The first item is the
        winning method's OOF calibrated probabilities in input order. The report
        is JSON-serializable and contains per-method metrics, the winning method,
        and all-row deployment parameters.

    Raises:
        ValueError: If inputs, fold counts, or method names are invalid.

    """
    score_arr, label_arr = _validated_scores_labels(scores, labels)
    methods_list = _validated_methods(methods)
    folds = _validated_fold_count(folds, label_arr)
    _validate_bin_count(ece_bins)

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(seed),
    )
    splits = list(splitter.split(score_arr, label_arr))

    oof_by_method: dict[str, list[float]] = {}
    method_reports: dict[str, dict[str, Any]] = {}

    for method in methods_list:
        oof = np.empty_like(score_arr, dtype=float)
        fold_reports: list[dict[str, Any]] = []
        for fold_id, (train_idx, heldout_idx) in enumerate(splits):
            params = _fit_method(
                method,
                score_arr[train_idx].tolist(),
                label_arr[train_idx].tolist(),
            )
            calibrated = apply_calibration(
                score_arr[heldout_idx].tolist(),
                method,
                params,
            )
            oof[heldout_idx] = np.asarray(calibrated, dtype=float)
            fold_metrics = calibration_metrics(
                label_arr[heldout_idx],
                calibrated,
                bins=ece_bins,
            )
            fold_metrics["fold"] = fold_id
            fold_metrics["n"] = int(len(heldout_idx))
            fold_reports.append(fold_metrics)

        all_params = _fit_method(method, score_arr.tolist(), label_arr.tolist())
        aggregate = calibration_metrics(label_arr, oof, bins=ece_bins)
        mean_brier = float(np.mean([fold["brier"] for fold in fold_reports]))
        mean_log_loss = float(np.mean([fold["log_loss"] for fold in fold_reports]))
        report = {
            **aggregate,
            "mean_oof_brier": mean_brier,
            "mean_oof_log_loss": mean_log_loss,
            "fold_metrics": fold_reports,
            "all_params": _jsonable_params(all_params),
        }
        oof_by_method[method] = oof.astype(float).tolist()
        method_reports[method] = report

    winner = min(
        methods_list,
        key=lambda method: (
            method_reports[method]["mean_oof_brier"],
            method_reports[method]["mean_oof_log_loss"],
            methods_list.index(method),
        ),
    )
    deployed_params = method_reports[winner]["all_params"]
    report = {
        "methods": method_reports,
        "winner": winner,
        "deployed": {
            "method": winner,
            "params": deployed_params,
            "fitted_on": "anchor",
        },
        "crossfit_folds": folds,
        "ece_bins": int(ece_bins),
    }
    return oof_by_method[winner], report


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
# Proper scoring rules (Brier, log-loss) and reliability diagnostics (ECE,
# reliability curve) are used to compare calibrators and to validate the
# deployed model.  We prefer Brier score for winner selection because it is
# a proper scoring rule that is sensitive to both calibration and sharpness
# (Silva Filho et al., 2023).  ECE is reported for visualization but is not
# used for selection because equal-width binning can be unstable with small
# samples.
# ---------------------------------------------------------------------------


def compute_brier(labels: LabelInput, probabilities: ScoreInput) -> float:
    """Compute binary Brier score.

    Args:
        labels: Binary labels.
        probabilities: Positive-class probabilities aligned to ``labels``.

    Returns:
        Mean squared probability error.

    """
    prob_arr, label_arr = _validated_probabilities_labels(probabilities, labels)
    return float(np.mean((prob_arr - label_arr) ** 2))


def compute_log_loss(labels: LabelInput, probabilities: ScoreInput) -> float:
    """Compute binary log-loss with clipped probabilities.

    Args:
        labels: Binary labels.
        probabilities: Positive-class probabilities aligned to ``labels``.

    Returns:
        Mean negative log-likelihood.

    """
    prob_arr, label_arr = _validated_probabilities_labels(probabilities, labels)
    clipped = np.clip(prob_arr, _EPS, 1.0 - _EPS)
    losses = -(label_arr * np.log(clipped) + (1 - label_arr) * np.log1p(-clipped))
    return float(np.mean(losses))


def expected_calibration_error(
    labels: LabelInput,
    probabilities: ScoreInput,
    *,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error.

    Args:
        labels: Binary labels.
        probabilities: Positive-class probabilities aligned to ``labels``.
        bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        Weighted average absolute difference between bin confidence and accuracy.

    """
    points = reliability_curve(labels, probabilities, bins=bins)
    n = sum(_point_count(point) for point in points)
    if n == 0:
        raise ValueError("At least one probability is required.")
    ece = 0.0
    for point in points:
        count = _point_count(point)
        gap = point["gap"]
        if count == 0 or gap is None:
            continue
        ece += count / n * abs(float(gap))
    return float(ece)


def reliability_curve(
    labels: LabelInput,
    probabilities: ScoreInput,
    *,
    bins: int = 10,
) -> list[dict[str, float | int | None]]:
    """Return serializable equal-width reliability-curve bin points.

    Args:
        labels: Binary labels.
        probabilities: Positive-class probabilities aligned to ``labels``.
        bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        One dictionary per bin with bounds, count, mean prediction, observed
        positive rate, and signed gap ``mean_predicted - observed_fraction``.

    """
    prob_arr, label_arr = _validated_probabilities_labels(probabilities, labels)
    _validate_bin_count(bins)

    bin_ids = np.minimum((prob_arr * bins).astype(int), bins - 1)
    points: list[dict[str, float | int | None]] = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        lower = bin_id / bins
        upper = (bin_id + 1) / bins
        count = int(np.sum(mask))
        if count == 0:
            points.append(
                {
                    "bin": bin_id,
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "mean_predicted": None,
                    "observed_fraction": None,
                    "gap": None,
                },
            )
            continue
        mean_predicted = float(np.mean(prob_arr[mask]))
        observed_fraction = float(np.mean(label_arr[mask]))
        points.append(
            {
                "bin": bin_id,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_predicted": mean_predicted,
                "observed_fraction": observed_fraction,
                "gap": float(mean_predicted - observed_fraction),
            },
        )
    return points


def calibration_metrics(
    labels: LabelInput,
    probabilities: ScoreInput,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Compute serializable calibration metrics for binary probabilities.

    Args:
        labels: Binary labels.
        probabilities: Positive-class probabilities aligned to ``labels``.
        bins: Number of equal-width bins for ECE and reliability points.

    Returns:
        Dictionary with Brier score, log-loss, ECE, and reliability curve.

    """
    return {
        "brier": compute_brier(labels, probabilities),
        "log_loss": compute_log_loss(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities, bins=bins),
        "reliability_curve": reliability_curve(labels, probabilities, bins=bins),
    }


def serialize_calibrator(method: CalibrationMethod, params: Mapping[str, float]) -> str:
    """Serialize calibration method and parameters as JSON.

    Args:
        method: Calibration family.
        params: Fitted calibration parameters.

    Returns:
        Stable JSON string containing ``method`` and ``params``.

    """
    payload = _calibrator_payload(method, params)
    return json.dumps(payload, sort_keys=True)


def deserialize_calibrator(
    payload: str | bytes,
) -> tuple[CalibrationMethod, CalibrationParams]:
    """Deserialize a calibrator JSON payload.

    Args:
        payload: JSON string or bytes from :func:`serialize_calibrator`.

    Returns:
        Tuple of ``(method, params)``.

    Raises:
        ValueError: If the payload has an invalid schema.

    """
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Calibrator payload must be valid JSON.") from exc
    if not isinstance(raw, dict):
        raise ValueError("Calibrator payload must be a JSON object.")
    method_raw = raw.get("method")
    params_raw = raw.get("params")
    if method_raw not in _VALID_METHODS:
        raise ValueError("Calibrator method must be 'platt' or 'temperature'.")
    if not isinstance(params_raw, dict):
        raise ValueError("Calibrator payload must contain a params object.")
    method = cast(CalibrationMethod, method_raw)
    params = _validate_params_for_method(method, params_raw)
    return method, params


def _fit_method(
    method: CalibrationMethod,
    scores: ScoreInput,
    labels: LabelInput,
) -> CalibrationParams:
    if method == "platt":
        return fit_platt(scores, labels)
    if method == "temperature":
        return fit_temperature(scores, labels)
    raise ValueError(f"Unknown calibration method: {method!r}.")


def _validated_scores_labels(
    scores: ScoreInput,
    labels: LabelInput,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    score_arr = _validated_scores(scores)
    label_arr = _validated_labels(labels)
    if len(score_arr) != len(label_arr):
        raise ValueError("scores and labels must have the same length.")
    return score_arr, label_arr


def _validated_probabilities_labels(
    probabilities: ScoreInput,
    labels: LabelInput,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    prob_arr = _validated_scores(probabilities)
    label_arr = _validated_labels(labels)
    if len(prob_arr) != len(label_arr):
        raise ValueError("probabilities and labels must have the same length.")
    return prob_arr, label_arr


def _point_count(point: Mapping[str, float | int | None]) -> int:
    count = point["count"]
    if not isinstance(count, int):
        raise ValueError("Reliability-curve point count must be an integer.")
    return count


def _validated_scores(scores: ScoreInput) -> NDArray[np.float64]:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1:
        raise ValueError("scores must be one-dimensional.")
    if len(arr) == 0:
        raise ValueError("scores must contain at least one value.")
    if not np.isfinite(arr).all():
        raise ValueError("scores must be finite.")
    if ((arr < 0.0) | (arr > 1.0)).any():
        raise ValueError("scores must be probabilities in [0, 1].")
    return arr.astype(float)


def _validated_labels(labels: LabelInput) -> NDArray[np.int_]:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if len(arr) == 0:
        raise ValueError("labels must contain at least one value.")
    numeric = np.asarray(arr, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("labels must be finite 0/1 values.")
    if ((numeric != 0.0) & (numeric != 1.0)).any():
        raise ValueError("labels must be coded as 0/1.")
    return numeric.astype(int)


def _require_two_classes(labels: NDArray[np.int_]) -> None:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Calibration fitting requires both binary classes.")


def _validated_fold_count(folds: int, labels: NDArray[np.int_]) -> int:
    folds = int(folds)
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    if folds > len(labels):
        raise ValueError("folds cannot exceed the number of rows.")
    _require_two_classes(labels)
    counts = np.bincount(labels, minlength=2)
    min_class_count = int(np.min(counts[:2]))
    if folds > min_class_count:
        raise ValueError(
            f"folds cannot exceed the smallest class count ({min_class_count}).",
        )
    return folds


def _validated_methods(methods: Sequence[CalibrationMethod]) -> list[CalibrationMethod]:
    if not methods:
        raise ValueError("At least one calibration method is required.")
    validated: list[CalibrationMethod] = []
    seen: set[str] = set()
    for method in methods:
        if method not in _VALID_METHODS:
            raise ValueError(f"Unknown calibration method: {method!r}.")
        if method in seen:
            raise ValueError(f"Duplicate calibration method: {method!r}.")
        seen.add(method)
        validated.append(method)
    return validated


def _validate_bin_count(bins: int) -> None:
    if int(bins) < 1:
        raise ValueError("bins must be at least 1.")


def _logit(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(probabilities, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return np.clip(out, _EPS, 1.0 - _EPS)


def _minimize_scalar(
    objective: Any,
    *,
    lower: float,
    upper: float,
    tolerance: float = 1e-6,
) -> float:
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        return _golden_section_search(
            objective,
            lower=lower,
            upper=upper,
            tolerance=tolerance,
        )

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": tolerance},
    )
    if not result.success or not math.isfinite(float(result.x)):
        return _golden_section_search(
            objective,
            lower=lower,
            upper=upper,
            tolerance=tolerance,
        )
    return float(result.x)


def _golden_section_search(
    objective: Any,
    *,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inverse_phi_squared = (3.0 - math.sqrt(5.0)) / 2.0
    left = float(lower)
    right = float(upper)
    h = right - left
    if h <= tolerance:
        return (left + right) / 2.0

    n_steps = int(math.ceil(math.log(tolerance / h) / math.log(inverse_phi)))
    c = left + inverse_phi_squared * h
    d = left + inverse_phi * h
    yc = float(objective(c))
    yd = float(objective(d))

    for _ in range(n_steps - 1):
        if yc < yd:
            right = d
            d = c
            yd = yc
            h = inverse_phi * h
            c = left + inverse_phi_squared * h
            yc = float(objective(c))
        else:
            left = c
            c = d
            yc = yd
            h = inverse_phi * h
            d = left + inverse_phi * h
            yd = float(objective(d))
    return float((left + right) / 2.0)


def _calibrator_payload(
    method: CalibrationMethod,
    params: Mapping[str, float],
) -> dict[str, Any]:
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown calibration method: {method!r}.")
    return {
        "method": method,
        "params": _validate_params_for_method(method, params),
    }


def _validate_params_for_method(
    method: CalibrationMethod,
    params: Mapping[str, Any],
) -> CalibrationParams:
    if method == "platt":
        required = ("a", "b")
    elif method == "temperature":
        required = ("T",)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}.")
    out: CalibrationParams = {}
    for key in required:
        if key not in params:
            raise ValueError(f"{method} params must contain {key!r}.")
        value = float(params[key])
        if not math.isfinite(value):
            raise ValueError(f"{method} param {key!r} must be finite.")
        out[key] = value
    if method == "temperature" and out["T"] <= 0:
        raise ValueError("temperature param 'T' must be positive.")
    return out


def _jsonable_params(params: Mapping[str, float]) -> CalibrationParams:
    return {str(key): float(value) for key, value in params.items()}
