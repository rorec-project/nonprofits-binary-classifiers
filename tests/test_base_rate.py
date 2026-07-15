"""Tests for base-rate-adjusted precision diagnostics."""

import pandas as pd
import pytest

from binary_classifier.evaluation.base_rate import (
    base_rate_report,
    precision_at_base_rate,
)


def test_precision_at_base_rate_hand_computed() -> None:
    assert precision_at_base_rate(0.8, 0.2, 0.1) == pytest.approx(
        0.08 / (0.08 + 0.18)
    )


def test_base_rate_report_has_weighted_unweighted_and_ci() -> None:
    frame = pd.DataFrame(
        {
            "EIN2": ["1", "2", "3", "4", "5", "6"],
            "prob_calibrated_oof": [0.9, 0.8, 0.7, 0.4, 0.2, 0.1],
            "human_label": [1, 0, 1, 0, 1, 0],
            "sample_prob": [0.5, 0.25, 0.5, 0.25, 0.5, 0.25],
        }
    )

    report = base_rate_report(
        frame,
        operating_threshold=0.2,
        max_f1_threshold=0.7,
        target=0.20,
        population_base_rate=0.20,
        seed=42,
        n_resamples=20,
    )

    assert report["unattainable"] is False
    assert report["threshold"] == pytest.approx(0.1)
    assert {point["weighted"] for point in report["points"]} == {False, True}
    assert report["selected"]["ci"]["lower"] is not None
    assert report["selected"]["ci"]["upper"] is not None


def test_base_rate_report_flags_unattainable_target() -> None:
    frame = pd.DataFrame(
        {
            "prob_calibrated_oof": [0.9, 0.8, 0.7, 0.6],
            "human_label": [1, 0, 1, 0],
            "sample_prob": [1.0, 1.0, 1.0, 1.0],
        }
    )

    report = base_rate_report(
        frame,
        operating_threshold=0.6,
        max_f1_threshold=0.6,
        target=1.01,
        population_base_rate=0.1,
        seed=42,
        n_resamples=5,
    )

    assert report["unattainable"] is True
    assert report["selected"]["base_rate_precision"] < 1.01
