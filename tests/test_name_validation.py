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
    gold = report["bmf_only_gold"]
    assert gold["target_population"] == "bmf_only"
    assert gold["n_gold"] == 1
    assert gold["variants"]["suffix_stripped"]["design_weighted_accuracy"] == 1.0


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


def test_run_name_validation_rejects_template_text_changed_after_draw(
    tiny_config,
    tiny_registry,
) -> None:
    """The gate rejects coding against text other than the drawn BMF-only name."""
    _write_validation_inputs(tiny_registry)
    template = pd.read_csv(tiny_registry.names_gold_coding_template)
    template.loc[0, "text"] = "Altered name"
    template.to_csv(tiny_registry.names_gold_coding_template, index=False)

    with pytest.raises(ValueError, match="text and split"):
        run_name_validation(tiny_config, tiny_registry)


def test_run_name_validation_reports_external_flags_and_gold_offset(
    tiny_config,
    tiny_registry,
) -> None:
    """The stage artifact reports IRS diagnostics and an uncorrected gold offset."""
    tiny_config.names.diagnostic_threshold = 0.5
    tiny_config.names.base_rate_shift_ratio_tolerance = 0.25
    _write_external_validation_inputs(tiny_registry, flat_model_rate=False)

    run_name_validation(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.names_validation.read_text())
    external = report["external_flag_validation"]
    panel = external["populations"]["panel_scoped"]["suffix_stripped"]
    bmf_only = external["populations"]["bmf_only"]["suffix_stripped"]
    assert external["status"] == "available"
    assert external["populations"]["panel_scoped"]["total_rows"] == 4
    assert external["populations"]["panel_scoped"]["bmf_covered_rows"] == 4
    assert external["populations"]["panel_scoped"]["bmf_coverage_rate"] == 1.0
    assert panel["flag_base_rate"] == 0.25
    assert panel["model_positive_rate"] == 0.25
    assert bmf_only["flag_base_rate"] == 0.5
    assert bmf_only["model_positive_rate"] == 0.5
    assert panel["metrics"]["precision"] == 1.0
    offset = external["construct_offset"]
    assert offset["source_population"] == (
        "uncontaminated panel prompt_dev + validation overlap"
    )
    assert offset["flag_construct"] == "IRS religious auspice"
    assert offset["target_construct"] == "observable religious purpose"
    assert offset["n_overlap"] == 4
    assert offset["offset"] == 0.25
    assert offset["flag_vs_human_purpose_metrics"]["confusion_matrix"] == {
        "tn": 3,
        "fp": 1,
        "fn": 0,
        "tp": 0,
    }
    assert offset["correction_applied"] is False
    assert "must not be read as purpose prevalence" in offset["interpretation"]
    shift = external["base_rate_shift"]
    assert shift["absolute_flag_rate_difference"] == 0.25
    assert shift["absolute_model_positive_rate_difference"] == 0.25
    assert shift["verdict"] == "PASS"
    assert external["base_rate_shift"]["failure_interpretation"]


def test_run_name_validation_fails_flat_external_flag_shift(
    tiny_config,
    tiny_registry,
) -> None:
    """A model rate that does not track the flag shift emits an explicit FAIL."""
    _write_external_validation_inputs(tiny_registry, flat_model_rate=True)

    run_name_validation(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.names_validation.read_text())
    shift = report["external_flag_validation"]["base_rate_shift"]
    assert shift["flag_rate_ratio"] == 2.0
    assert shift["model_positive_rate_ratio"] == 1.0
    assert shift["verdict"] == "FAIL"
    assert "input shape rather than measuring religion" in shift["interpretation"]


def test_run_name_validation_reports_unavailable_external_flags_without_bmf_panel_rows(
    tiny_config,
    tiny_registry,
) -> None:
    """Missing BMF enrichment does not block independent names validation."""
    _write_external_validation_inputs(tiny_registry, flat_model_rate=False)
    panel = pd.read_parquet(tiny_registry.names_panel_cleaned)
    panel["has_bmf"] = False
    panel["is_external_religious_flag"] = pd.NA
    panel.to_parquet(tiny_registry.names_panel_cleaned, index=False)

    run_name_validation(tiny_config, tiny_registry)

    external = json.loads(tiny_registry.names_validation.read_text())["external_flag_validation"]
    assert external == {
        "status": "unavailable",
        "reason": "No panel_scoped rows have BMF enrichment for external flags.",
        "populations": {
            "bmf_only": {
                "total_rows": 4,
                "bmf_covered_rows": 4,
                "bmf_coverage_rate": 1.0,
            },
            "panel_scoped": {
                "total_rows": 4,
                "bmf_covered_rows": 0,
                "bmf_coverage_rate": 0.0,
            },
        },
    }


def test_run_name_validation_excludes_bmf_uncovered_panel_rows_from_flag_rates(
    tiny_config,
    tiny_registry,
) -> None:
    """External flags use only panel rows enriched from the BMF."""
    _write_external_validation_inputs(tiny_registry, flat_model_rate=False)
    panel = pd.read_parquet(tiny_registry.names_panel_cleaned)
    panel.loc[3, "has_bmf"] = False
    panel["is_external_religious_flag"] = panel["is_external_religious_flag"].astype(
        "boolean"
    )
    panel.loc[3, "is_external_religious_flag"] = pd.NA
    panel.to_parquet(tiny_registry.names_panel_cleaned, index=False)

    run_name_validation(tiny_config, tiny_registry)

    external = json.loads(tiny_registry.names_validation.read_text())["external_flag_validation"]
    panel_report = external["populations"]["panel_scoped"]
    assert panel_report["total_rows"] == 4
    assert panel_report["bmf_covered_rows"] == 3
    assert panel_report["bmf_coverage_rate"] == 0.75
    assert panel_report["suffix_stripped"]["n"] == 3


def _write_validation_inputs(registry) -> None:
    _write_completed_name_gold_template(registry)
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
            "prob_raw": [0.9, 0.8, 0.2, 0.1, 0.9, 0.2],
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
                    "population": "panel_scoped",
                    "input_variant": variant,
                    "prob_raw": probability,
                    "lexicon_rule_label": lexicon_label,
                },
            )
    pd.DataFrame(scores).to_parquet(registry.names_scores, index=False)
    _append_gold_scores(registry)


def _write_external_validation_inputs(registry, *, flat_model_rate: bool) -> None:
    _write_completed_name_gold_template(registry)
    ein2s = ["A", "B", "C", "D", "E", "F", "G", "H"]
    populations = ["panel_scoped"] * 4 + ["bmf_only"] * 4
    flags = [True, False, False, False, True, True, False, False]
    has_mission = [True] * 4 + [False] * 4
    pd.DataFrame(
        {
            "EIN2": ein2s[:4],
            "has_mission": has_mission[:4],
            "is_manifest_contaminated": [False] * 4,
            "has_bmf": [True] * 4,
            "is_external_religious_flag": flags[:4],
        },
    ).to_parquet(registry.names_panel_cleaned, index=False)
    pd.DataFrame(
        {
            "EIN2": ein2s[4:],
            "population": ["bmf_only"] * 4,
            "has_bmf": [True] * 4,
            "is_external_religious_flag": flags[4:],
        },
    ).to_parquet(registry.names_bmf_only_cleaned, index=False)

    model_scores = [0.9, 0.1, 0.2, 0.3]
    if flat_model_rate:
        bmf_scores = [0.9, 0.1, 0.2, 0.3]
    else:
        bmf_scores = [0.9, 0.8, 0.2, 0.3]
    scores = []
    for variant in ("suffix_stripped", "suffix_retaining"):
        for ein2, population, probability, flag in zip(
            ein2s,
            populations,
            model_scores + bmf_scores,
            flags,
            strict=True,
        ):
            scores.append(
                {
                    "EIN2": ein2,
                    "population": population,
                    "input_variant": variant,
                    "prob_raw": probability,
                    "lexicon_rule_label": int(flag),
                },
            )
    pd.DataFrame(scores).to_parquet(registry.names_scores, index=False)
    _append_gold_scores(registry)
    pd.DataFrame(
        {
            "EIN2": ein2s[:4],
            "split": ["prompt_dev", "prompt_dev", "validation", "validation"],
            "text": ["a", "b", "c", "d"],
            "human_label": [0, 0, 0, 0],
        },
    ).to_csv(registry.gold_coding_template, index=False)
    pd.DataFrame(
        {"EIN2": ein2s[:4], "pred_label": [1, 1, 0, 0], "prob_calibrated": [0.9] * 4},
    ).to_parquet(registry.predictions_full_parquet, index=False)


def _write_completed_name_gold_template(registry) -> None:
    registry.names_gold_coding_template.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "EIN2": ["NAMES_GOLD"],
            "split": ["names_gold"],
            "text": ["Example name"],
            "human_label": [0],
        }
    ).to_csv(registry.names_gold_coding_template, index=False)
    pd.DataFrame(
        {
            "EIN2": ["NAMES_GOLD"],
            "name_raw": ["Example name"],
            "population": ["bmf_only"],
            "inclusion_probability": [1.0],
            "sampling_cell": ["ntee_x_only|none"],
        }
    ).to_csv(
        registry.names_gold_manifest,
        index=False,
    )
    registry.names_gold_coding_instructions.write_text(
        "Use the unchanged mission construct: positive means observable religious or "
        "spiritual purpose, tradition, or motivation as a core driver of the work. "
        "A saint name alone is not religious. Faith-founded identity without religious "
        "purpose is not religious. Enter only 0 or 1 in human_label.\n"
    )


def _append_gold_scores(registry) -> None:
    scores = pd.read_parquet(registry.names_scores)
    scores = pd.concat(
        [
            scores,
            pd.DataFrame(
                [
                    {
                        "EIN2": "NAMES_GOLD",
                        "population": "bmf_only",
                        "input_variant": "suffix_stripped",
                        "prob_raw": 0.1,
                        "lexicon_rule_label": 0,
                    },
                    {
                        "EIN2": "NAMES_GOLD",
                        "population": "bmf_only",
                        "input_variant": "suffix_retaining",
                        "prob_raw": 0.1,
                        "lexicon_rule_label": 0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    scores.to_parquet(registry.names_scores, index=False)
