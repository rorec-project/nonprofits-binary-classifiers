"""Quantification cross-checks for prevalence estimation."""

import importlib
import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

_EPS = 1e-12


def emq_prevalence(
    val_posteriors: Iterable[Iterable[float]],
    val_labels: Iterable[float],
    corpus_posteriors: Iterable[Iterable[float]],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> float:
    """Estimate positive-class prevalence with vendored SLD/EMQ.

    This implements the Saerens-Latinne-Decaestecker prior-shift EM loop over
    precomputed binary posterior probabilities. Validation labels define the
    source prior; corpus posteriors are iteratively reweighted to estimate the
    target prior.

    Args:
        val_posteriors: Validation-set posterior matrix with columns ordered as
            negative class then positive class. Used for shape validation.
        val_labels: Validation-set binary labels encoded as 0/1.
        corpus_posteriors: Target corpus posterior matrix with columns ordered
            as negative class then positive class.
        max_iter: Maximum EM iterations.
        tol: L1 convergence tolerance between consecutive prevalence vectors.

    Returns:
        Estimated positive-class prevalence in ``[0, 1]``.

    Raises:
        ValueError: If inputs are empty, non-finite, malformed, outside the
            expected probability/label ranges, or if convergence knobs are
            invalid.

    """
    source_post = _as_binary_posterior_matrix(val_posteriors, "val_posteriors")
    labels = _as_binary_label_array(val_labels)
    target_post = _as_binary_posterior_matrix(corpus_posteriors, "corpus_posteriors")
    _validate_fit_inputs(source_post, labels)
    max_iterations = _validate_max_iter(max_iter)
    tolerance = _validate_tol(tol)

    source_prior = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float64)
    source_prior /= source_prior.sum()
    source_prior = _smooth_prior(source_prior)
    target_prior = source_prior.copy()

    for _ in range(max_iterations):
        previous_prior = target_prior.copy()
        adjusted = target_post * (target_prior / source_prior)
        row_sums = adjusted.sum(axis=1, keepdims=True)
        if (row_sums <= 0.0).any():
            raise ValueError("EMQ reweighting produced zero-probability rows")
        adjusted /= row_sums
        target_prior = adjusted.mean(axis=0)
        if np.abs(target_prior - previous_prior).sum() <= tolerance:
            break
    else:
        logger.warning("EMQ reached max_iter=%d before convergence", max_iterations)

    return _positive_prevalence(target_prior, "EMQ prevalence")


def kdey_prevalence(
    val_posteriors: Iterable[Iterable[float]],
    val_labels: Iterable[float],
    corpus_posteriors: Iterable[Iterable[float]],
) -> float:
    """Estimate positive-class prevalence with QuaPy's KDEyML cross-check.

    Args:
        val_posteriors: Validation-set posterior matrix with columns ordered as
            negative class then positive class.
        val_labels: Validation-set binary labels encoded as 0/1.
        corpus_posteriors: Target corpus posterior matrix with columns ordered
            as negative class then positive class.

    Returns:
        Estimated positive-class prevalence in ``[0, 1]``.

    Raises:
        ValueError: If inputs are invalid.
        ImportError: If QuaPy is unavailable or its KDEyML API is incompatible.

    """
    source_post, labels, target_post = _validated_quantifier_inputs(
        val_posteriors,
        val_labels,
        corpus_posteriors,
    )
    aggregative = _import_quapy_aggregative()
    kdey_cls = getattr(aggregative, "KDEyML", None)
    if kdey_cls is None:
        raise ImportError(
            "QuaPy KDEy prevalence requires quapy.method.aggregative.KDEyML"
        )

    try:
        quantifier = kdey_cls(
            _PosteriorShim(source_post),
            fit_classifier=False,
            val_split=None,
        )
        quantifier.fit(_dummy_features(len(source_post)), labels)
        prevalence = quantifier.aggregate(target_post)
    except Exception as exc:  # pragma: no cover - exercised only on API drift.
        raise ImportError("QuaPy KDEyML API is unavailable or incompatible") from exc

    return _positive_prevalence(prevalence, "QuaPy KDEy prevalence")


class _PosteriorShim(BaseEstimator):
    """Minimal fitted sklearn-style classifier over stored posteriors."""

    def __init__(self, posteriors: NDArray[np.float64]) -> None:
        self.posteriors = posteriors
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = 1
        self.is_fitted_ = True

    def fit(self, _x: NDArray[np.float64], _y: NDArray[np.int64]) -> "_PosteriorShim":
        """Return the already-fitted shim."""
        return self

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return stored posteriors for dummy feature rows."""
        if len(x) != len(self.posteriors):
            raise ValueError("posterior shim received an unexpected number of rows")
        return self.posteriors.copy()


def _validated_quantifier_inputs(
    val_posteriors: Iterable[Iterable[float]],
    val_labels: Iterable[float],
    corpus_posteriors: Iterable[Iterable[float]],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Validate common quantifier inputs."""
    source_post = _as_binary_posterior_matrix(val_posteriors, "val_posteriors")
    labels = _as_binary_label_array(val_labels)
    target_post = _as_binary_posterior_matrix(corpus_posteriors, "corpus_posteriors")
    _validate_fit_inputs(source_post, labels)
    return source_post, labels, target_post


def _as_binary_posterior_matrix(
    values: Iterable[Iterable[float]],
    name: str,
) -> NDArray[np.float64]:
    """Convert and validate a binary posterior matrix."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_rows, 2)")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    if ((arr < 0.0) | (arr > 1.0)).any():
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    if not np.allclose(arr.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} rows must sum to 1")
    return arr


def _as_binary_label_array(values: Iterable[float]) -> NDArray[np.int64]:
    """Convert and validate binary labels."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("val_labels must be one-dimensional")
    if arr.size == 0:
        raise ValueError("val_labels must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError("val_labels must contain only finite values")
    if not np.isin(arr, [0.0, 1.0]).all():
        raise ValueError("val_labels must contain only 0/1 labels")
    return arr.astype(np.int64)


def _validate_fit_inputs(
    source_post: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> None:
    """Validate paired validation posteriors and labels."""
    if len(source_post) != len(labels):
        raise ValueError("val_posteriors and val_labels must have the same length")


def _validate_max_iter(max_iter: int) -> int:
    """Validate an EM iteration cap."""
    max_iterations = int(max_iter)
    if max_iterations <= 0:
        raise ValueError("max_iter must be positive")
    return max_iterations


def _validate_tol(tol: float) -> float:
    """Validate an EM convergence tolerance."""
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be finite and positive")
    return tolerance


def _smooth_prior(prior: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smooth zero priors to keep EM ratios finite."""
    if (prior <= 0.0).any():
        prior = prior + _EPS
        prior = prior / prior.sum()
    return prior


def _positive_prevalence(values: object, name: str) -> float:
    """Extract and validate the positive-class prevalence from a vector."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"{name} must be a two-element prevalence vector")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    estimate = float(arr[1])
    if estimate < -1e-8 or estimate > 1.0 + 1e-8:
        raise ValueError(f"{name} must be in [0, 1]")
    return min(1.0, max(0.0, estimate))


def _dummy_features(n_rows: int) -> NDArray[np.float64]:
    """Build dummy feature rows for precomputed-posterior quantifier fitting."""
    return np.arange(n_rows, dtype=np.float64).reshape(-1, 1)


def _import_quapy_aggregative() -> Any:
    """Import QuaPy's aggregative quantifier module."""
    try:
        return importlib.import_module("quapy.method.aggregative")
    except Exception as exc:  # pragma: no cover - depends on optional install state.
        raise ImportError(
            "QuaPy prevalence cross-checks require optional dependency quapy"
        ) from exc
