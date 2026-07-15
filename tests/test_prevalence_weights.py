"""Tests for prevalence design weights and EIN2 alignment."""

import logging

import pandas as pd
import pytest

from binary_classifier.config import BinaryClassifierConfig, load_config
from binary_classifier.prevalence.weights import (
    align_labels_predictions,
    design_weights,
)


def test_prevalence_config_defaults_and_yaml_blocks() -> None:
    """Root config exposes prevalence defaults and YAML overrides load."""
    cfg = BinaryClassifierConfig()

    assert cfg.prevalence.alpha == 0.05
    assert cfg.prevalence.cross_checks == ["emq"]
    assert cfg.prevalence.use_design_weights is True
    assert cfg.prevalence.per_ntee is True
    assert cfg.prevalence.ntee_min_n == 10
    assert cfg.prevalence.low_tier_sensitivity is True

    production = load_config("config/religious_missions.yaml")
    smoke = load_config("config/smoke.yaml")

    assert production.prevalence.cross_checks == ["emq"]
    assert smoke.prevalence.alpha == 0.05


def test_design_weights_inverse_probability_and_normalization() -> None:
    """Weights are inverse sample probabilities, normalized to mean one."""
    manifest = pd.DataFrame(
        {"sample_prob": [0.5, 0.25]},
        index=pd.Index(["first", "second"], name="row"),
    )

    weights = design_weights(manifest)
    raw_weights = design_weights(manifest, normalize=False)

    assert list(weights.index) == ["first", "second"]
    assert weights.name == "design_weight"
    assert raw_weights.tolist() == pytest.approx([2.0, 4.0])
    assert weights.tolist() == pytest.approx([2.0 / 3.0, 4.0 / 3.0])
    assert weights.mean() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "manifest, match",
    [
        (pd.DataFrame({"other": [0.5]}), "sample_prob"),
        (pd.DataFrame({"sample_prob": [0.5, None]}), "non-null"),
        (pd.DataFrame({"sample_prob": [0.5, "bad"]}), "numeric"),
        (pd.DataFrame({"sample_prob": [0.5, 0.0]}), "strictly positive"),
        (pd.DataFrame({"sample_prob": [0.5, -0.1]}), "strictly positive"),
    ],
)
def test_design_weights_validate_sample_prob(
    manifest: pd.DataFrame,
    match: str,
) -> None:
    """Invalid sampling probabilities fail loudly."""
    with pytest.raises(ValueError, match=match):
        design_weights(manifest)


def test_align_labels_predictions_strips_ein2_and_adds_weights(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Alignment joins on stripped string EIN2s and warns on missing keys."""
    labeled = pd.DataFrame(
        {
            "EIN2": [" 002 ", "001", None, ""],
            "human_label": [1, 0, 1, 0],
            "sample_prob": [0.5, 0.25, 0.1, 0.1],
        },
    )
    predictions = pd.DataFrame(
        {
            "EIN2": ["001 ", "002", "999"],
            "prediction": [0.2, 0.8, 0.9],
        },
    )

    with caplog.at_level(logging.WARNING):
        aligned = align_labels_predictions(labeled, predictions)

    assert list(aligned.columns) == [
        "EIN2",
        "human_label",
        "prediction",
        "design_weight",
    ]
    assert aligned["EIN2"].tolist() == ["002", "001"]
    assert aligned["human_label"].tolist() == [1, 0]
    assert aligned["prediction"].tolist() == pytest.approx([0.8, 0.2])
    assert aligned["design_weight"].tolist() == pytest.approx([2.0 / 3.0, 4.0 / 3.0])
    assert "Dropped 2 rows with missing EIN2 from labeled_df" in caplog.text


def test_align_labels_predictions_warns_on_unmatched_predictions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Labeled rows without a prediction are dropped and counted."""
    labeled = pd.DataFrame(
        {
            "EIN2": ["001", "002"],
            "human_label": [1, 0],
            "sample_prob": [0.5, 0.25],
        },
    )
    predictions = pd.DataFrame({"EIN2": ["001"], "prediction": [0.7]})

    with caplog.at_level(logging.WARNING):
        aligned = align_labels_predictions(labeled, predictions)

    assert aligned["EIN2"].tolist() == ["001"]
    assert aligned["design_weight"].tolist() == pytest.approx([1.0])
    assert "Dropped 1 labeled rows without matching predictions by EIN2" in caplog.text


def test_align_labels_predictions_rejects_duplicate_ein2() -> None:
    """Duplicate normalized join keys are rejected before merging."""
    labeled = pd.DataFrame(
        {
            "EIN2": ["001", " 001 "],
            "human_label": [1, 0],
            "sample_prob": [0.5, 0.25],
        },
    )
    predictions = pd.DataFrame({"EIN2": ["001"], "prediction": [0.7]})

    with pytest.raises(ValueError, match="duplicate EIN2"):
        align_labels_predictions(labeled, predictions)
