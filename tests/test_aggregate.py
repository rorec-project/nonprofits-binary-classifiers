"""Tests for label aggregation dispatch behavior."""

import pandas as pd
import pytest

from binary_classifier.annotate.aggregate import aggregate_labels


@pytest.mark.parametrize(
    ("method", "message"),
    [
        (
            "dawid_skene",
            "unverified for correlated LLM ensembles; majority vote is the default",
        ),
        (
            "crowdlab",
            "requires pred_probs from a trained classifier",
        ),
    ],
)
def test_quarantined_aggregators_raise_not_implemented(
    method: str,
    message: str,
) -> None:
    """Alternative aggregation arms fail explicitly while quarantined."""
    df = pd.DataFrame(
        [
            {"EIN2": "00-1", "source_id": "m__p1", "label": 1, "confidence": 0.9},
        ]
    )

    with pytest.raises(NotImplementedError, match=message):
        aggregate_labels(df, method=method)


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
