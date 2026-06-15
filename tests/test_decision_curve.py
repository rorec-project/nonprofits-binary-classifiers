"""Tests for T3.3 decision-curve utilities."""

import pytest

from binary_classifier.evaluation.decision_curve import net_benefit


def test_net_benefit_matches_manual_computation() -> None:
    points = net_benefit(
        y_true=[1, 0, 1, 0],
        y_prob=[0.9, 0.8, 0.4, 0.1],
        thresholds=[0.5, 0.75],
    )

    assert points == [
        {
            "threshold": 0.5,
            "net_benefit": pytest.approx(0.0),
            "treat_all_net_benefit": pytest.approx(0.0),
            "treat_none_net_benefit": 0.0,
            "tp": 1,
            "fp": 1,
            "n": 4,
        },
        {
            "threshold": 0.75,
            "net_benefit": pytest.approx(-0.5),
            "treat_all_net_benefit": pytest.approx(-1.0),
            "treat_none_net_benefit": 0.0,
            "tp": 1,
            "fp": 1,
            "n": 4,
        },
    ]


def test_net_benefit_rejects_boundary_thresholds() -> None:
    with pytest.raises(ValueError, match="open interval"):
        net_benefit(y_true=[1, 0], y_prob=[0.8, 0.2], thresholds=[0.0, 0.5])
