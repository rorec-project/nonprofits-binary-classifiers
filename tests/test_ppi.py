"""Tests for prediction-powered prevalence estimation."""

import numpy as np
import pytest

from binary_classifier.config import BinaryClassifierConfig

pytest.importorskip("ppi_py")

from binary_classifier.prevalence.ppi import ppi_prevalence


@pytest.mark.parametrize("seed", range(5))
def test_ppi_ci_covers_known_population_prevalence_across_seeds(
    seed: int,
    tiny_config: BinaryClassifierConfig,
) -> None:
    """Synthetic PPI CIs cover the known population prevalence across seeds."""
    rng = np.random.default_rng(seed)
    n_population = 5_000
    true_prevalence = 0.37
    n_positive = int(n_population * true_prevalence)
    labels = np.concatenate(
        [
            np.ones(n_positive, dtype=float),
            np.zeros(n_population - n_positive, dtype=float),
        ],
    )
    rng.shuffle(labels)
    labeled_idx = rng.choice(n_population, size=500, replace=False)

    result = ppi_prevalence(
        labels[labeled_idx],
        labels[labeled_idx],
        labels,
        alpha=tiny_config.prevalence.alpha,
    )

    assert result["ci_lower"] <= true_prevalence <= result["ci_upper"]
    assert result["alpha"] == tiny_config.prevalence.alpha
    assert result["n_labeled"] == 500
    assert result["n_unlabeled"] == n_population
    assert result["weighted"] is False
    assert result["lam"] is None  # ppi_py auto-tunes lambda internally


def test_ppi_weighted_and_unweighted_estimates_differ_with_skewed_weights() -> None:
    """Skewed labeled-row weights change the PPI prevalence estimate."""
    y_labeled = np.array([1, 1, 0, 0, 0, 0], dtype=float)
    yhat_labeled = np.array([0.9, 0.8, 0.2, 0.2, 0.2, 0.2], dtype=float)
    yhat_unlabeled = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.2], dtype=float)
    weights = np.array([10, 10, 1, 1, 1, 1], dtype=float)

    unweighted = ppi_prevalence(
        y_labeled,
        yhat_labeled,
        yhat_unlabeled,
        alpha=0.05,
    )
    weighted = ppi_prevalence(
        y_labeled,
        yhat_labeled,
        yhat_unlabeled,
        alpha=0.05,
        w=weights,
    )

    assert unweighted["weighted"] is False
    assert weighted["weighted"] is True
    assert weighted["estimate"] != pytest.approx(unweighted["estimate"])


def test_ppi_records_configured_alpha(tiny_config: BinaryClassifierConfig) -> None:
    """The wrapper records the configured alpha, not ``ppi_py``'s default."""
    result = ppi_prevalence(
        [1.0, 0.0, 1.0, 0.0],
        [0.8, 0.2, 0.7, 0.3],
        [0.75, 0.25, 0.6, 0.4],
        alpha=tiny_config.prevalence.alpha,
    )

    assert result["alpha"] == tiny_config.prevalence.alpha
