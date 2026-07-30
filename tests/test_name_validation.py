"""Tests for paired names-arm transfer validation."""

import json

import pandas as pd
import pytest

from binary_classifier.names.validation import run_name_validation


def test_run_name_validation_compares_uncontaminated_paired_variants(
    tiny_config,
    tiny_registry,
) -> None:
    """The emitted report compares equal positive-rate name methods per variant."""
    tiny_config.evaluation.bootstrap_resamples = 10
    _write_validation_inputs(tiny_registry)

    run_name_validation(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.names_validation.read_text())
    assert report["comparison_population"] == {
        "n_paired_before_contamination_exclusion": 5,
        "n_contaminated_excluded": 1,
        "n_evaluated": 4,
        "n_missing_mission_score_excluded": 1,
    }
    assert set(report["variants"]) == {"suffix_stripped", "suffix_retaining"}
    primary = report["variants"]["suffix_stripped"]
    assert primary["matched_operating_point"] == {
        "method": "top_k_by_raw_score",
        "positive_count": 2,
        "positive_rate": 0.5,
        "tie_breaker": "EIN2 ascending",
    }
    assert primary["transfer_metrics"]["precision"] == 1.0
    assert primary["transfer_metrics"]["recall"] == 1.0
    assert primary["lexicon_metrics"]["precision"] == 0.5
    assert primary["lexicon_metrics"]["recall"] == 0.5
    assert primary["hypothesis"]["transfer_beats_lexicon_on_precision"] is True
    assert primary["agreement"]["score_pearson_correlation"] == 1.0
    assert primary["calibration_against_mission_labels"]["ece"] == 0.15
    assert primary["calibration_against_mission_scores"]["mean_squared_error"] == 0.0
    ablation = report["variants"]["suffix_retaining"]
    assert ablation["transfer_metrics"]["precision"] == 0.0
    assert ablation["calibration_against_mission_labels"]["ece"] == pytest.approx(0.75)
    assert ablation["calibration_against_mission_scores"]["mean_squared_error"] == pytest.approx(
        0.37
    )
    assert "cannot validate" in report["limitations"][0]


def test_run_name_validation_rejects_incomplete_variant_pair(
    tiny_config,
    tiny_registry,
) -> None:
    """A partial score artifact cannot masquerade as a two-variant comparison."""
    _write_validation_inputs(tiny_registry)
    scores = pd.read_parquet(tiny_registry.names_scores)
    scores.loc[scores["input_variant"] == "suffix_retaining"].to_parquet(
        tiny_registry.names_scores,
        index=False,
    )

    with pytest.raises(ValueError, match="both required input variants"):
        run_name_validation(tiny_config, tiny_registry)


def _write_validation_inputs(registry) -> None:
    pd.DataFrame(
        {
            "EIN2": ["A", "B", "C", "D", "X", "R"],
            "has_mission": [True, True, True, True, True, True],
            "is_manifest_contaminated": [False, False, False, False, True, False],
        },
    ).to_parquet(registry.names_panel_cleaned, index=False)
    pd.DataFrame(
        {
            "EIN2": ["A", "B", "C", "D", "X", "R"],
            "pred_label": [1, 1, 0, 0, 1, 0],
            "prob_calibrated": [0.9, 0.8, 0.2, 0.1, 0.9, None],
        },
    ).to_parquet(registry.predictions_full_parquet, index=False)
    scores = []
    for variant, probabilities in {
        "suffix_stripped": [0.9, 0.8, 0.2, 0.1, 0.99, 0.1],
        "suffix_retaining": [0.4, 0.3, 0.9, 0.8, 0.99, 0.1],
    }.items():
        for ein2, probability, lexicon_label in zip(
            ["A", "B", "C", "D", "X", "R"],
            probabilities,
            [1, 0, 1, 0, 1, 0],
            strict=True,
        ):
            scores.append(
                {
                    "EIN2": ein2,
                    "population": "panel_501c3",
                    "input_variant": variant,
                    "prob_raw": probability,
                    "lexicon_rule_label": lexicon_label,
                },
            )
    pd.DataFrame(scores).to_parquet(registry.names_scores, index=False)
