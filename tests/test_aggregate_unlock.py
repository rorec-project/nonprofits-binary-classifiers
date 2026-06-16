"""Tests for unlocked aggregation comparison arms."""

import logging

import numpy as np
import pandas as pd
import pytest

from binary_classifier.annotate.aggregate import (
    aggregate_crowdlab,
    aggregate_dawid_skene,
    aggregate_labels,
    majority_vote,
)
from binary_classifier.config import AggregationConfig, BinaryClassifierConfig, load_config


SILVER_COLUMNS = [
    "EIN2",
    "silver_label",
    "silver_confidence",
    "num_votes",
    "num_abstain",
    "agreement",
    "tie",
]


def _store(n: int = 20) -> pd.DataFrame:
    rows = []
    sources = ["model_a__prompt_1", "model_b__prompt_1", "model_c__prompt_1"]
    for i in range(n):
        true_label = i % 2
        for j, source_id in enumerate(sources):
            if i == 0 and j == 2:
                continue
            label = true_label if j < 2 else 1 - true_label
            if i == 1 and j == 1:
                label = np.nan
            rows.append(
                {
                    "EIN2": f"00-{i:03d}",
                    "source_id": source_id,
                    "source_type": "model",
                    "label": label,
                    "confidence": 0.9 if label == true_label else 0.55,
                },
            )
    return pd.DataFrame(rows)


def _pred_probs(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        p1 = 0.85 if i % 2 else 0.15
        rows.append({"EIN2": f"00-{i:03d}", "p0": 1 - p1, "p1": p1})
    return pd.DataFrame(rows)


def test_aggregation_config_defaults_and_yaml_load() -> None:
    """Aggregation config keeps production aggregation majority-only."""
    assert BinaryClassifierConfig().aggregation.method == "majority"
    assert BinaryClassifierConfig().aggregation.comparison_arms == []

    cfg = load_config("config/religious_missions.yaml")

    assert cfg.aggregation.method == "majority"
    assert cfg.aggregation.comparison_arms == []


def test_aggregation_config_rejects_non_majority_production_method() -> None:
    """Dawid-Skene/CROWDLAB are comparison arms, not stage-04 methods."""
    with pytest.raises(ValueError):
        AggregationConfig(method="dawid_skene")

    cfg = AggregationConfig(comparison_arms=["dawid_skene", "crowdlab"])

    assert cfg.method == "majority"
    assert cfg.comparison_arms == ["dawid_skene", "crowdlab"]


def test_crowdlab_pivot_nan_handling_and_prediction_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CROWDLAB treats explicit/missing votes as NaN and drops missing probs."""
    caplog.set_level(logging.WARNING)
    store = _store(n=4)
    probs = _pred_probs(n=3)

    aggregated = aggregate_crowdlab(store, pred_probs=probs)

    first = aggregated.loc[aggregated["EIN2"] == "00-000"].iloc[0]
    second = aggregated.loc[aggregated["EIN2"] == "00-001"].iloc[0]

    assert len(aggregated) == 3
    assert first["num_votes"] == 2
    assert first["num_abstain"] == 1
    assert second["num_votes"] == 2
    assert second["num_abstain"] == 1
    assert "Dropping 1 EIN2 rows missing classifier predictions" in caplog.text


def test_unlocked_arms_match_majority_vote_schema() -> None:
    """Comparison arms are drop-in schema replacements for majority vote."""
    store = _store()

    assert list(majority_vote(store).columns) == SILVER_COLUMNS
    assert list(aggregate_dawid_skene(store).columns) == SILVER_COLUMNS
    assert list(aggregate_crowdlab(store, pred_probs=_pred_probs()).columns) == (
        SILVER_COLUMNS
    )


def test_crowdlab_without_pred_probs_preserves_quarantine() -> None:
    """CROWDLAB still fails explicitly when classifier probabilities are absent."""
    with pytest.raises(NotImplementedError, match="requires pred_probs"):
        aggregate_labels(_store(), method="crowdlab")


def test_dawid_skene_small_n_end_to_end() -> None:
    """Dawid-Skene runs end-to-end on a small fabricated annotation store."""
    aggregated = aggregate_labels(_store(), method="dawid_skene")

    assert len(aggregated) == 20
    assert set(aggregated["silver_label"].dropna().astype(int)).issubset({0, 1})
    assert aggregated["silver_confidence"].dropna().between(0, 1).all()


def test_crowdlab_small_n_end_to_end() -> None:
    """CROWDLAB runs end-to-end on a small fabricated annotation store."""
    aggregated = aggregate_labels(_store(), method="crowdlab", pred_probs=_pred_probs())

    assert len(aggregated) == 20
    assert set(aggregated["silver_label"].dropna().astype(int)).issubset({0, 1})
    assert aggregated["silver_confidence"].dropna().between(0, 1).all()
