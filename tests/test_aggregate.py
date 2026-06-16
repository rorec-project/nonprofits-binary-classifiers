"""Tests for label aggregation dispatch behavior."""

import pandas as pd
import pytest

from binary_classifier.annotate.aggregate import aggregate_labels


def test_dawid_skene_small_n_raises_value_error() -> None:
    """Dawid-Skene raises ValueError when label matrix is too small."""
    df = pd.DataFrame(
        [
            {"EIN2": "00-1", "source_id": "m__p1", "label": 1, "confidence": 0.9},
        ]
    )

    with pytest.raises(ValueError, match="requires at least 2 tasks and 2 annotators"):
        aggregate_labels(df, method="dawid_skene")


def test_crowdlab_raises_not_implemented() -> None:
    """CROWDLAB raises NotImplementedError when pred_probs are unavailable."""
    df = pd.DataFrame(
        [
            {"EIN2": "00-1", "source_id": "m__p1", "label": 1, "confidence": 0.9},
        ]
    )

    with pytest.raises(
        NotImplementedError,
        match="requires pred_probs from a trained classifier",
    ):
        aggregate_labels(df, method="crowdlab")


def test_aggregate_labels_majority_unchanged() -> None:
    """Majority dispatch still returns the existing silver-label fields."""
    df = pd.DataFrame(
        [
            {"EIN2": "00-1", "source_id": "m__p1", "label": 1, "confidence": 0.8},
            {"EIN2": "00-1", "source_id": "m__p2", "label": 1, "confidence": 0.6},
            {"EIN2": "00-1", "source_id": "m__p3", "label": 0, "confidence": 0.9},
        ]
    )

    aggregated = aggregate_labels(df, method="majority")

    assert list(aggregated.columns) == [
        "EIN2",
        "silver_label",
        "silver_confidence",
        "num_votes",
        "num_abstain",
        "agreement",
        "tie",
    ]
    row = aggregated.iloc[0]
    assert row["EIN2"] == "00-1"
    assert row["silver_label"] == 1
    assert row["silver_confidence"] == pytest.approx(0.7)
    assert row["num_votes"] == 3
    assert row["num_abstain"] == 0
    assert row["agreement"] == pytest.approx(2 / 3)
    assert not row["tie"]
