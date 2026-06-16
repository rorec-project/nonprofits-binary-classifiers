"""Tests for prevalence quantification cross-checks."""

import numpy as np
import pytest
from numpy.typing import NDArray

from binary_classifier.prevalence.quantify import emq_prevalence


def test_emq_recovers_known_prior_shift_on_synthetic_posteriors() -> None:
    """Vendored EMQ recovers a target prior under synthetic prior shift."""
    val_posteriors, val_labels, corpus_posteriors = _prior_shift_problem()

    estimate = emq_prevalence(val_posteriors, val_labels, corpus_posteriors)

    assert estimate == pytest.approx(0.6, abs=1e-3)


@pytest.mark.parametrize(
    "val_posteriors, val_labels, corpus_posteriors, match",
    [
        ([[0.2, 0.8, 0.0]], [1], [[0.2, 0.8]], "shape"),
        ([[0.2, 0.8]], [1, 0], [[0.2, 0.8]], "same length"),
        ([[0.2, 0.8]], [2], [[0.2, 0.8]], "0/1"),
        ([[0.2, 0.8]], [1], [[0.2, 0.9]], "sum to 1"),
    ],
)
def test_emq_validates_inputs(
    val_posteriors: list[list[float]],
    val_labels: list[int],
    corpus_posteriors: list[list[float]],
    match: str,
) -> None:
    """Invalid posterior or label inputs fail clearly."""
    with pytest.raises(ValueError, match=match):
        emq_prevalence(val_posteriors, val_labels, corpus_posteriors)


def _prior_shift_problem() -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """Build a binary prior-shift problem with exact class-conditional counts."""
    validation_prior = 0.3
    high_posterior = _posterior(validation_prior, p_score_given_pos=0.8, p_score_given_neg=0.2)
    low_posterior = _posterior(validation_prior, p_score_given_pos=0.2, p_score_given_neg=0.8)

    val_labels = np.concatenate(
        [
            np.ones(300, dtype=np.int64),
            np.zeros(700, dtype=np.int64),
        ],
    )
    val_posteriors = np.vstack(
        [
            np.tile(high_posterior, (240, 1)),
            np.tile(low_posterior, (60, 1)),
            np.tile(high_posterior, (140, 1)),
            np.tile(low_posterior, (560, 1)),
        ],
    )
    corpus_posteriors = np.vstack(
        [
            np.tile(high_posterior, (480, 1)),
            np.tile(low_posterior, (120, 1)),
            np.tile(high_posterior, (80, 1)),
            np.tile(low_posterior, (320, 1)),
        ],
    )
    return val_posteriors, val_labels, corpus_posteriors


def _posterior(
    prior: float,
    *,
    p_score_given_pos: float,
    p_score_given_neg: float,
) -> NDArray[np.float64]:
    """Convert a class-conditional score probability into a source posterior."""
    positive = prior * p_score_given_pos
    negative = (1.0 - prior) * p_score_given_neg
    pos_posterior = positive / (positive + negative)
    return np.array([1.0 - pos_posterior, pos_posterior], dtype=np.float64)
