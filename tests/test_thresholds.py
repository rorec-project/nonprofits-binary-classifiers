"""Tests for T3.3 threshold selection utilities."""

import pytest

from binary_classifier.evaluation.thresholds import pick_threshold


def test_precision_floor_uses_highest_recall_eligible_threshold() -> None:
    result = pick_threshold(
        probs=[0.9, 0.8, 0.7, 0.4, 0.2],
        labels=[1, 0, 1, 0, 1],
        policy="precision_floor",
        precision_floor=0.65,
    )

    assert result["threshold"] == pytest.approx(0.7)
    assert result["achieved_precision"] == pytest.approx(2 / 3)
    assert result["achieved_recall"] == pytest.approx(2 / 3)
    assert result["max_f1_threshold"] == pytest.approx(0.2)
    assert result["floor_unattainable"] is False
    expected_points = [
        {"threshold": 0.2, "precision": 3 / 5, "recall": 1.0, "f1": 0.75},
        {"threshold": 0.4, "precision": 0.5, "recall": 2 / 3, "f1": 4 / 7},
        {"threshold": 0.7, "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3},
        {"threshold": 0.8, "precision": 0.5, "recall": 1 / 3, "f1": 0.4},
        {"threshold": 0.9, "precision": 1.0, "recall": 1 / 3, "f1": 0.5},
    ]
    assert len(result["pr_curve_points"]) == len(expected_points)
    for actual, expected in zip(result["pr_curve_points"], expected_points):
        assert actual["threshold"] == pytest.approx(expected["threshold"])
        assert actual["precision"] == pytest.approx(expected["precision"])
        assert actual["recall"] == pytest.approx(expected["recall"])
        assert actual["f1"] == pytest.approx(expected["f1"])


def test_precision_floor_fallback_when_floor_unattainable() -> None:
    result = pick_threshold(
        probs=[0.9, 0.8, 0.7],
        labels=[0, 1, 1],
        policy="precision_floor",
        precision_floor=0.9,
    )

    assert result["threshold"] == pytest.approx(0.7)
    assert result["achieved_precision"] == pytest.approx(2 / 3)
    assert result["achieved_recall"] == pytest.approx(1.0)
    assert result["floor_unattainable"] is True


def test_max_f1_policy_selects_max_f1_threshold() -> None:
    result = pick_threshold(
        probs=[0.9, 0.8, 0.7, 0.4, 0.2],
        labels=[1, 0, 1, 0, 1],
        policy="max_f1",
        precision_floor=0.95,
    )

    assert result["threshold"] == pytest.approx(0.2)
    assert result["achieved_precision"] == pytest.approx(3 / 5)
    assert result["achieved_recall"] == pytest.approx(1.0)
    assert result["floor_unattainable"] is False
