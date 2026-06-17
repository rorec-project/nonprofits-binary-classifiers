"""Quantification cross-checks for prevalence estimation.

This module provides two quantification-based prevalence cross-checks that are
run on the HIGH+MEDIUM tier as diagnostics.  They are **not** the primary
estimator but are reported in the stage-09 JSON report for downstream review.

1. **EMQ (Expectation-Maximization Quantification)** implements the
   Saerens-Latinne-Decaestecker (SLD) prior-shift EM loop
   (Saerens et al., 2002, https://doi.org/10.1162/089976602753284446)
   over precomputed binary posterior probabilities.  It iteratively reweights
   validation posteriors to estimate the target prior under the assumption of
   prior (label) shift.

2. **KDEy** uses QuaPy's ``KDEyML`` quantifier (Moreo et al., 2021 via QuaPy)
   which fits kernel-density estimates to the per-class posterior distributions
   and aggregates them over the target corpus.

Quantification methods are surveyed in González et al. (2017,
https://doi.org/10.1145/3117807), Bella et al. (2010,
https://doi.org/10.1109/ICDM.2010.75), and Esuli et al. (2023,
https://link.springer.com/book/10.1007/978-3-031-20467-8).  A large
comparative evaluation is provided by Schumacher et al. (2025,
https://www.jmlr.org/papers/v26/21-0241.html).  González, Moreo & Sebastiani
(2024, http://nmis.isti.cnr.it/sebastiani/Publications/DMKD2024a.pdf) caution
that prior-shift quantifiers can fail under covariate or concept shift, so
these cross-checks are used only as robustness signals, not as primary
estimates.

References
----------

* Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the Outputs of
  a Classifier to New a Priori Probabilities: A Simple Procedure.
  *Neural Computation*, 14(1), 21--41.
  https://doi.org/10.1162/089976602753284446
* Bella, A., Ferri, C., Hernández-Orallo, J., & Ramírez-Quintana, M. J. (2010).
  Quantification via Probability Estimators. *ICDM*.
  https://doi.org/10.1109/ICDM.2010.75
* González, P. et al. (2017). A Review on Quantification Learning.
  *ACM Computing Surveys*. https://doi.org/10.1145/3117807
* Esuli, A., Fabris, A., Moreo, A., & Sebastiani, F. (2023). *Learning to
  Quantify*. Springer. https://link.springer.com/book/10.1007/978-3-031-20467-8
* Schumacher, T., Strohmaier, M., & Lemmerich, F. (2025). A Comparative
  Evaluation of Quantification Methods. *JMLR*, 26(21), 1--42.
  https://www.jmlr.org/papers/v26/21-0241.html
* González, P., Moreo, A., & Sebastiani, F. (2024). Binary Quantification and
  Dataset Shift. *DMKD*.
  http://nmis.isti.cnr.it/sebastiani/Publications/DMKD2024a.pdf
* Moreo, A., Esuli, A., & Sebastiani, F. (2021). QuaPy: A Python-based open
  source framework for quantification. *ACM SIGIR*.
"""

import importlib
import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# EMQ / SLD prior-shift quantifier
# ---------------------------------------------------------------------------
# The EMQ loop (Saerens et al., 2002) iteratively reweights validation-set
# posteriors to match the target corpus prior.  It assumes prior (label) shift
# but not covariate or concept shift.  A small epsilon is added to zero priors
# to keep the reweighting ratios finite.
# ---------------------------------------------------------------------------


def emq_prevalence(
    val_posteriors: Iterable[Iterable[float]],
    val_labels: Iterable[float],
    corpus_posteriors: Iterable[Iterable[float]],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> float:
    """Estimate positive-class prevalence with vendored SLD/EMQ.

    This implements the Saerens-Latinne-Decaestecker prior-shift EM loop over
    precomputed binary posterior probabilities (Saerens et al., 2002,
    https://doi.org/10.1162/089976602753284446).  Validation labels define the
    source prior; corpus posteriors are iteratively reweighted to estimate the
    target prior.

    Args:
        val_posteriors: Validation-set posterior matrix with columns ordered as
            negative class then positive class.  Used for shape validation.
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


# ---------------------------------------------------------------------------
# KDEy quantifier via QuaPy
# ---------------------------------------------------------------------------
# KDEyML fits kernel-density estimates to the per-class posterior distributions
# observed on the validation set and aggregates them over the target corpus
# (Moreo et al., 2021 via QuaPy).  This is a non-parametric alternative to EMQ
# that does not assume prior shift.
# ---------------------------------------------------------------------------


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
    """Minimal fitted sklearn-style classifier over stored posteriors.

    QuaPy's KDEyML expects a fitted sklearn classifier.  This shim stores
    precomputed posteriors and returns them from ``predict_proba``, avoiding
    re-fitting a real model when we already have calibrated probabilities.
    """

    def __init__(self, posteriors: NDArray[np.float64]) -> None:
        """Initialize with a precomputed posterior matrix.

        Args:
            posteriors: Array of shape (n_samples, n_classes) with posterior
                probabilities.
        """
        self.posteriors = posteriors
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = 1
        self.is_fitted_ = True

    def fit(self, _x: NDArray[np.float64], _y: NDArray[np.int64]) -> "_PosteriorShim":
        """Return the already-fitted shim."""
        return self

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return stored posteriors for dummy feature rows.

        Args:
            x: Dummy feature array whose length must match the stored posteriors.

        Returns:
            Copy of the stored posteriors.

        Raises:
            ValueError: If the number of rows does not match.
        """
        if len(x) != len(self.posteriors):
            raise ValueError("posterior shim received an unexpected number of rows")
        return self.posteriors.copy()


def _validated_quantifier_inputs(
    val_posteriors: Iterable[Iterable[float]],
    val_labels: Iterable[float],
    corpus_posteriors: Iterable[Iterable[float]],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Validate common quantifier inputs.

    Args:
        val_posteriors: Validation-set posterior matrix.
        val_labels: Validation-set binary labels.
        corpus_posteriors: Target corpus posterior matrix.

    Returns:
        Validated arrays ready for quantifier fitting and aggregation.

    """
    source_post = _as_binary_posterior_matrix(val_posteriors, "val_posteriors")
    labels = _as_binary_label_array(val_labels)
    target_post = _as_binary_posterior_matrix(corpus_posteriors, "corpus_posteriors")
    _validate_fit_inputs(source_post, labels)
    return source_post, labels, target_post


def _as_binary_posterior_matrix(
    values: Iterable[Iterable[float]],
    name: str,
) -> NDArray[np.float64]:
    """Convert and validate a binary posterior matrix.

    Args:
        values: Iterable of rows, each row an iterable of two probabilities.
        name: Column name for error messages.

    Returns:
        A 2-D float array of shape (n_rows, 2) where each row sums to 1.

    Raises:
        ValueError: If the input is empty, not 2-D, has rows that do not sum to
            1, or contains non-finite or out-of-range probabilities.

    """
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
    """Convert and validate binary labels.

    Args:
        values: Iterable of binary labels.

    Returns:
        A 1-D int64 array of 0/1 values.

    Raises:
        ValueError: If the input is empty, not 1-D, non-finite, or contains
            values other than 0/1.

    """
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
    """Validate paired validation posteriors and labels.

    Args:
        source_post: Validation posterior matrix.
        labels: Validation binary labels.

    Raises:
        ValueError: If the lengths do not match.

    """
    if len(source_post) != len(labels):
        raise ValueError("val_posteriors and val_labels must have the same length")


def _validate_max_iter(max_iter: int) -> int:
    """Validate an EM iteration cap.

    Args:
        max_iter: Proposed maximum number of iterations.

    Returns:
        The validated integer.

    Raises:
        ValueError: If the value is not positive.

    """
    max_iterations = int(max_iter)
    if max_iterations <= 0:
        raise ValueError("max_iter must be positive")
    return max_iterations


def _validate_tol(tol: float) -> float:
    """Validate an EM convergence tolerance.

    Args:
        tol: Proposed L1 tolerance.

    Returns:
        The validated float.

    Raises:
        ValueError: If the value is not finite and positive.

    """
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be finite and positive")
    return tolerance


def _smooth_prior(prior: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smooth zero priors to keep EM ratios finite.

    A small epsilon is added to any zero prior mass and the vector is
    renormalized.  This prevents division by zero in the EM reweighting step.

    Args:
        prior: A 2-element prevalence vector.

    Returns:
        The smoothed and renormalized vector.

    """
    if (prior <= 0.0).any():
        prior = prior + _EPS
        prior = prior / prior.sum()
    return prior


def _positive_prevalence(values: object, name: str) -> float:
    """Extract and validate the positive-class prevalence from a vector.

    Args:
        values: A 2-element prevalence vector (negative, positive).
        name: Identifier for error messages.

    Returns:
        The positive-class prevalence clipped to [0, 1].

    Raises:
        ValueError: If the input is not a 2-element finite vector or if the
            positive value is outside [0, 1].

    """
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
    """Build dummy feature rows for precomputed-posterior quantifier fitting.

    QuaPy expects feature arrays even when the posterior shim ignores them.
    This returns an (n_rows, 1) array of integers cast to float.

    Args:
        n_rows: Number of dummy rows.

    Returns:
        A 2-D float array of shape (n_rows, 1).

    """
    return np.arange(n_rows, dtype=np.float64).reshape(-1, 1)


def _import_quapy_aggregative() -> Any:
    """Import QuaPy's aggregative quantifier module.

    Returns:
        The ``quapy.method.aggregative`` module.

    Raises:
        ImportError: If QuaPy is not installed.

    """
    try:
        return importlib.import_module("quapy.method.aggregative")
    except Exception as exc:  # pragma: no cover - depends on optional install state.
        raise ImportError(
            "QuaPy prevalence cross-checks require optional dependency quapy"
        ) from exc
