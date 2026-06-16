"""Prediction-powered prevalence estimation wrapper."""

import importlib
import logging
from collections.abc import Iterable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

PPIResult: TypeAlias = dict[str, float | int | bool | None]


def ppi_prevalence(
    y_labeled: Iterable[float],
    yhat_labeled: Iterable[float],
    yhat_unlabeled: Iterable[float],
    *,
    alpha: float,
    w: Iterable[float] | None = None,
) -> PPIResult:
    """Estimate population prevalence with prediction-powered inference.

    The estimand is the mean of the binary label ``Y``. ``yhat`` inputs are
    calibrated probabilities for the positive class and are passed directly to
    ``ppi_py``'s mean estimator. The PPI confidence-interval ``alpha`` is always
    passed explicitly because the library default is 0.1.

    Args:
        y_labeled: Human labels for the labeled anchor rows, encoded as 0/1.
        yhat_labeled: Calibrated probabilities for the labeled anchor rows.
        yhat_unlabeled: Calibrated probabilities for the unlabeled population
            or target frame.
        alpha: Significance level for the returned confidence interval.
        w: Optional labeled-row weights, such as inverse-probability design
            weights. When provided, these are passed to ``ppi_py`` as ``w``.

    Returns:
        A dictionary containing the point estimate, confidence bounds, the
        power-tuning ``lam`` (``None`` because ppi_py auto-tunes it internally),
        input row counts, a weighted flag, and the ``alpha`` actually used.

    Raises:
        ValueError: If inputs are empty, non-finite, misaligned, outside the
            expected 0/1 or probability ranges, or have invalid weights.
        ImportError: If ``ppi_py`` is unavailable.

    """
    alpha_value = _validate_alpha(alpha)
    y = _as_1d_float_array(y_labeled, "y_labeled")
    yhat = _as_1d_float_array(yhat_labeled, "yhat_labeled")
    yhat_unlab = _as_1d_float_array(yhat_unlabeled, "yhat_unlabeled")
    _validate_same_length(y, yhat)
    _validate_binary_labels(y)
    _validate_probabilities(yhat, "yhat_labeled")
    _validate_probabilities(yhat_unlab, "yhat_unlabeled")

    weights = None if w is None else _as_weight_array(w, len(y))
    ppi_py = importlib.import_module("ppi_py")

    # Leave ``lam`` unset so ppi_py estimates the optimal PPI++ power-tuning
    # parameter from the data (Angelopoulos et al. 2023, arXiv:2311.01453). This
    # uses ppi_py's documented public behaviour (``lam=None`` -> auto-tune)
    # instead of re-deriving lambda through a private ppi_py internal.
    call_kwargs: dict[str, Any] = {}
    if weights is not None:
        call_kwargs["w"] = weights

    estimate = _as_scalar(
        ppi_py.ppi_mean_pointestimate(y, yhat, yhat_unlab, **call_kwargs),
        "estimate",
    )
    ci_lower, ci_upper = ppi_py.ppi_mean_ci(
        y,
        yhat,
        yhat_unlab,
        alpha=alpha_value,
        **call_kwargs,
    )

    result: PPIResult = {
        "estimate": estimate,
        "ci_lower": _as_scalar(ci_lower, "ci_lower"),
        "ci_upper": _as_scalar(ci_upper, "ci_upper"),
        "lam": None,  # auto-tuned internally by ppi_py
        "alpha": alpha_value,
        "n_labeled": int(len(y)),
        "n_unlabeled": int(len(yhat_unlab)),
        "weighted": weights is not None,
    }
    logger.info(
        "Estimated PPI prevalence on %d labeled and %d unlabeled rows",
        result["n_labeled"],
        result["n_unlabeled"],
    )
    return result


def _validate_alpha(alpha: float) -> float:
    """Validate and normalize a confidence-interval alpha value."""
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    return alpha_value


def _as_1d_float_array(values: Iterable[float], name: str) -> NDArray[np.float64]:
    """Convert an iterable to a non-empty one-dimensional float array."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_weight_array(values: Iterable[float], n_labeled: int) -> NDArray[np.float64]:
    """Convert and validate labeled-row PPI weights."""
    weights = _as_1d_float_array(values, "w")
    if len(weights) != n_labeled:
        raise ValueError("w must have the same length as y_labeled")
    if (weights <= 0).any():
        raise ValueError("w must contain strictly positive weights")
    return weights


def _validate_same_length(
    y: NDArray[np.float64],
    yhat: NDArray[np.float64],
) -> None:
    """Validate paired labels and predictions have the same length."""
    if len(y) != len(yhat):
        raise ValueError("y_labeled and yhat_labeled must have the same length")


def _validate_binary_labels(y: NDArray[np.float64]) -> None:
    """Validate labels are encoded as binary 0/1 values."""
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y_labeled must contain only 0/1 labels")


def _validate_probabilities(values: NDArray[np.float64], name: str) -> None:
    """Validate calibrated probabilities are in the closed unit interval."""
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{name} must contain probabilities in [0, 1]")


def _as_scalar(value: float | NDArray[np.float64], name: str) -> float:
    """Convert a scalar-like value returned by ``ppi_py`` into a Python float."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar")
    scalar = float(arr.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar
