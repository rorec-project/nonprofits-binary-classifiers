"""Tests for stage 09 prevalence estimation."""

import json
import math

import pandas as pd
import pytest

from binary_classifier.prevalence.composite import (
    composite,
    rogan_gladen,
    rogan_gladen_variance,
)
from binary_classifier.data.ntee_labels import load_ntee_labels
from binary_classifier.prevalence.estimate import (
    _RAW_MULTIPLICITY_COLUMN,
    _estimate_hm,
    _estimate_low,
    _estimate_low_rules,
    _ntee_descriptives,
    _z_value,
    run_prevalence,
)


def test_rogan_gladen_matches_hand_computation() -> None:
    """Rogan-Gladen point and delta variance match hand calculations."""
    p_obs = 0.32
    sens = 0.80
    spec = 0.90
    expected = (p_obs + spec - 1.0) / (sens + spec - 1.0)

    estimate = rogan_gladen(p_obs, sens, spec)

    assert estimate == pytest.approx(expected)

    p_obs_var = 0.001
    sens_var = 0.002
    spec_var = 0.003
    denominator = sens + spec - 1.0
    numerator = p_obs + spec - 1.0
    expected_variance = (
        (1.0 / denominator) ** 2 * p_obs_var
        + (-numerator / denominator**2) ** 2 * sens_var
        + ((sens - p_obs) / denominator**2) ** 2 * spec_var
    )

    assert rogan_gladen_variance(
        p_obs,
        sens,
        spec,
        p_obs_var,
        sens_var,
        spec_var,
    ) == pytest.approx(expected_variance)


def test_composite_variance_formula() -> None:
    """Composite prevalence applies stratum shares to means and variances."""
    estimate, variance = composite(
        {
            "HIGH_MEDIUM": (0.40, 0.010, 0.75),
            "LOW": (0.20, 0.020, 0.25),
        }
    )

    assert estimate == pytest.approx(0.35)
    assert variance == pytest.approx(0.75**2 * 0.010 + 0.25**2 * 0.020)


def test_run_prevalence_writes_report_and_ntee_suppression(
    tiny_config,
    tiny_registry,
) -> None:
    """Run stage 09 on fabricated artifacts and validate output schemas."""
    tiny_config.prevalence.ntee_min_n = 10
    tiny_config.prevalence.cross_checks = ["emq"]
    _write_prevalence_inputs(tiny_registry)

    run_prevalence(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.prevalence_report.read_text())
    by_ntee = pd.read_csv(tiny_registry.prevalence_by_ntee)

    true_prevalence = 0.38
    assert report["stage"] == "09_prevalence"
    assert report["alpha"] == tiny_config.prevalence.alpha
    assert set(report["estimand_statements"]) == {"HIGH_MEDIUM", "LOW", "composite"}
    assert report["n"] == {
        "total": 100,
        "total_raw_ein2": 100,
        "anchor": 50,
        "anchor_high_medium": 40,
        "anchor_low": 10,
        "HIGH_MEDIUM": 80,
        "HIGH_MEDIUM_raw_ein2": 80,
        "LOW": 20,
        "LOW_raw_ein2": 20,
    }
    assert report["composite"]["ci_lower"] <= true_prevalence <= report["composite"]["ci_upper"]
    assert report["hm"]["weighted_ppi"]["estimate"] == pytest.approx(0.40, abs=0.05)
    assert report["hm"]["unweighted_ppi"]["estimate"] == pytest.approx(0.40, abs=0.05)
    assert report["low"]["rogan_gladen"]["estimate"] == pytest.approx(0.30)
    assert report["cross_checks"]["emq"]["status"] == "ok"

    assert list(by_ntee.columns) == [
        "ntee_major_group",
        "n_anchor",
        "estimator",
        "estimate",
        "ci_lower",
        "ci_upper",
        "suppressed",
    ]
    suppressed = by_ntee.loc[by_ntee["ntee_major_group"] == "B"].iloc[0]
    unsuppressed = by_ntee.loc[by_ntee["ntee_major_group"] == "A"].iloc[0]
    assert bool(suppressed["suppressed"])
    assert suppressed["estimator"] == "not estimated"
    assert math.isnan(float(suppressed["estimate"]))
    assert not bool(unsuppressed["suppressed"])
    assert report["per_ntee"]["n_suppressed"] == 1


def _write_prevalence_inputs(registry) -> None:
    predictions = _predictions_frame()
    anchor = _anchor_frame(predictions)
    predictions.to_parquet(registry.predictions_parquet, index=False)
    anchor[
        ["EIN2", "prob_raw", "prob_calibrated_oof", "human_label", "tier", "sample_prob"]
    ].to_parquet(registry.anchor_oof_scores, index=False)
    anchor[
        ["EIN2", "stratum", "tier", "ntee_major_group", "sample_prob", "split"]
    ].to_csv(registry.anchor_manifest, index=False)
    registry.rule_validation.write_text(
        json.dumps(
            {
                "scope": "anchor_low_cells",
                "counts": {"tp": 5, "fp": 0, "tn": 5, "fn": 0},
                "metrics": {
                    "sensitivity": {
                        "value": 1.0,
                        "ci": {"lower": 0.8, "upper": 1.0},
                        "numerator": 5,
                        "denominator": 5,
                    },
                    "specificity": {
                        "value": 1.0,
                        "ci": {"lower": 0.8, "upper": 1.0},
                        "numerator": 5,
                        "denominator": 5,
                    },
                },
            }
        )
    )


def _predictions_frame() -> pd.DataFrame:
    rows = []
    for i in range(100):
        tier = "HIGH" if i < 40 else "MEDIUM" if i < 80 else "LOW"
        label = int(i < 32 or 80 <= i < 86)
        is_low = tier == "LOW"
        ntee = "A" if i < 90 else "B"
        probability = math.nan if is_low else 0.9 if label else 0.1
        rows.append(
            {
                "EIN2": f"E{i:04d}",
                "pred_label": label,
                "pred_label_maxf1": label,
                "pred_label_baserate": label,
                "prob_raw": probability,
                "prob_calibrated": probability,
                "decision_source": (
                    "rule_strong_positive"
                    if is_low and label
                    else "rule_short_negative"
                    if is_low
                    else "classifier"
                ),
                "tier": tier,
                "Q": 1.0 if is_low else 3.5,
                "ntee_major_group": ntee,
                "model_id": "fixture",
                "checkpoint_sha256": "fixture",
                "calibrator_method": "platt",
                "threshold": 0.5,
                "inference_date": "2026-06-16T00:00:00+00:00",
                "pipeline_version": "test",
                "config_hash": "test",
            }
        )
    return pd.DataFrame(rows)


def _anchor_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    anchor_rows = []
    for i in list(range(16)) + list(range(32, 56)) + list(range(80, 90)):
        row = predictions.iloc[i].to_dict()
        prob = 0.9 if row["pred_label"] == 1 else 0.1
        anchor_rows.append(
            {
                "EIN2": row["EIN2"],
                "prob_raw": prob,
                "prob_calibrated_oof": prob,
                "human_label": row["pred_label"],
                "tier": row["tier"],
                "sample_prob": 0.5,
                "ntee_major_group": row["ntee_major_group"],
                "stratum": row["tier"],
                "split": "anchor",
            }
        )
    return pd.DataFrame(anchor_rows)


def test_estimate_low_falls_back_to_uncorrected_when_rule_unvalidated() -> None:
    """A null rule sensitivity (zero-denominator at stage 07) must not crash.

    Stage 07 writes ``value: null`` for a rule metric whose denominator is zero
    (anchor has LOW rows but the rule layer covered none). Stage 09 must skip the
    Rogan-Gladen correction and fall back to the uncorrected observed LOW rate
    rather than raising on ``float(None)``.
    """
    low_predictions = pd.DataFrame({"pred_label": [1, 0, 1, 0, 1]})
    rule_validation = {
        "metrics": {
            "sensitivity": {"value": None, "ci": {"lower": None, "upper": None}},
            "specificity": {"value": 0.9, "ci": {"lower": 0.8, "upper": 0.95}},
        }
    }

    result = _estimate_low_rules(
        low_predictions,
        rule_validation,
        alpha=0.05,
        z_value=_z_value(0.05),
        include_sensitivity=True,
    )

    assert result.corrected is False
    assert result.sensitivity_band is None
    # Uncorrected estimate equals the observed LOW rule-label rate (3/5).
    assert result.observed_prevalence == pytest.approx(0.6)
    assert result.estimate.estimate == pytest.approx(0.6)


def test_hm_unlabeled_probabilities_are_expanded_by_raw_multiplicity() -> None:
    """PPI corpus mean targets raw EIN2 organizations, not unique text rows."""
    anchor = pd.DataFrame(
        {
            "human_label": [0, 1],
            "prob_calibrated_oof": [0.0, 1.0],
            "sample_prob": [0.5, 0.5],
        }
    )
    corpus = pd.DataFrame(
        {
            "prob_calibrated": [0.2, 0.8],
            _RAW_MULTIPLICITY_COLUMN: [3, 1],
        }
    )

    estimates = _estimate_hm(anchor, corpus, alpha=0.05, z_value=_z_value(0.05))

    expanded_mean = ([0.2, 0.2, 0.2, 0.8])
    assert estimates["unweighted_ppi"].estimate == pytest.approx(
        sum(expanded_mean) / len(expanded_mean)
    )


def test_low_decomposition_routes_classifier_to_ppi_and_rules_to_rg() -> None:
    """LOW classifier rows use PPI and rule rows use RG before raw-share recombination."""
    anchor_low = pd.DataFrame(
        {
            "human_label": [0, 1, 1],
            "prob_calibrated_oof": [0.0, 1.0, 0.0],
            "sample_prob": [0.5, 0.5, 0.5],
            "decision_source": [
                "low_via_classifier",
                "low_via_classifier",
                "rule_strong_positive",
            ],
        }
    )
    low_predictions = pd.DataFrame(
        {
            "pred_label": [0, 1, 0],
            "prob_calibrated": [0.2, math.nan, math.nan],
            "decision_source": [
                "low_via_classifier",
                "rule_strong_positive",
                "rule_short_negative",
            ],
            _RAW_MULTIPLICITY_COLUMN: [3, 1, 2],
        }
    )
    rule_validation = {
        "metrics": {
            "sensitivity": {"value": 1.0, "ci": {"lower": 1.0, "upper": 1.0}},
            "specificity": {"value": 1.0, "ci": {"lower": 1.0, "upper": 1.0}},
        }
    }

    decomposed = _estimate_low(
        anchor_low,
        low_predictions,
        rule_validation,
        alpha=0.05,
        z_value=_z_value(0.05),
        include_sensitivity=False,
    )
    old_all_rg = _estimate_low_rules(
        low_predictions,
        rule_validation,
        alpha=0.05,
        z_value=_z_value(0.05),
        include_sensitivity=False,
    )

    low_sub = decomposed.sub_strata or {}
    assert low_sub["low_via_classifier"]["estimator"] == "ppi"
    classifier_anchor_ppi = _estimate_hm(
        anchor_low.loc[anchor_low["decision_source"].eq("low_via_classifier")],
        low_predictions.loc[
            low_predictions["decision_source"].eq("low_via_classifier")
        ],
        alpha=0.05,
        z_value=_z_value(0.05),
    )["weighted_ppi"]
    assert low_sub["low_via_classifier"]["estimate"]["estimate"] == pytest.approx(
        classifier_anchor_ppi.estimate
    )
    assert low_sub["rule"]["estimator"] == "rogan_gladen"
    assert low_sub["low_via_classifier"]["share"] + low_sub["rule"]["share"] == pytest.approx(1.0)
    expected = (
        low_sub["low_via_classifier"]["share"]
        * low_sub["low_via_classifier"]["estimate"]["estimate"]
        + low_sub["rule"]["share"] * low_sub["rule"]["estimate"]["estimate"]
    )
    assert decomposed.estimate.estimate == pytest.approx(expected)
    assert decomposed.estimate.estimate != pytest.approx(old_all_rg.estimate.estimate)


def test_run_prevalence_uses_predictions_full_raw_tier_shares(
    tiny_config,
    tiny_registry,
) -> None:
    """Stage 09 prefers per-organization predictions_full for raw tier shares."""
    tiny_config.prevalence.per_ntee = False
    _write_prevalence_inputs(tiny_registry)
    predictions = pd.read_parquet(tiny_registry.predictions_parquet)
    duplicate = predictions.iloc[[0]].copy()
    duplicate["EIN2"] = "RAW_DUPLICATE"
    predictions_full = pd.concat([predictions, duplicate], ignore_index=True)
    predictions_full.to_parquet(tiny_registry.predictions_full_parquet, index=False)

    run_prevalence(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.prevalence_report.read_text())
    assert report["n"]["total_raw_ein2"] == 101
    assert report["n"]["HIGH_MEDIUM_raw_ein2"] == 81
    assert report["tier_shares"]["HIGH_MEDIUM"] == pytest.approx(81 / 101)
    assert sum(report["tier_shares"].values()) == pytest.approx(1.0)


def _ntee_descriptives_fixture() -> pd.DataFrame:
    """A small predictions frame covering scored, rule, and missing-NTEE rows."""
    rows = [
        # Group A: 3 classifier-scored rows (2 positive) + 1 rule-routed row.
        {
            "EIN2": "A0",
            "ntee_major_group": "A",
            "decision_source": "classifier",
            "prob_raw": 0.8,
            "prob_calibrated": 0.9,
            "pred_label": 1,
            "pred_label_maxf1": 1,
            "pred_label_baserate": 0,
        },
        {
            "EIN2": "A1",
            "ntee_major_group": "A",
            "decision_source": "classifier",
            "prob_raw": 0.2,
            "prob_calibrated": 0.1,
            "pred_label": 0,
            "pred_label_maxf1": 0,
            "pred_label_baserate": 0,
        },
        {
            "EIN2": "A2",
            "ntee_major_group": "A",
            "decision_source": "low_via_classifier",
            "prob_raw": 0.7,
            "prob_calibrated": 0.8,
            "pred_label": 1,
            "pred_label_maxf1": 1,
            "pred_label_baserate": 1,
        },
        {
            "EIN2": "A3",
            "ntee_major_group": "A",
            "decision_source": "rule_strong_positive",
            "prob_raw": math.nan,
            "prob_calibrated": math.nan,
            "pred_label": 1,
            "pred_label_maxf1": 1,
            "pred_label_baserate": 1,
        },
        # Missing-NTEE ("?") rows: must fold into Z.
        {
            "EIN2": "Q0",
            "ntee_major_group": "?",
            "decision_source": "rule_short_negative",
            "prob_raw": math.nan,
            "prob_calibrated": math.nan,
            "pred_label": 0,
            "pred_label_maxf1": 0,
            "pred_label_baserate": 0,
        },
        {
            "EIN2": "Q1",
            "ntee_major_group": "?",
            "decision_source": "rule_short_negative",
            "prob_raw": math.nan,
            "prob_calibrated": math.nan,
            "pred_label": 0,
            "pred_label_maxf1": 0,
            "pred_label_baserate": 0,
        },
        # Native Z row.
        {
            "EIN2": "Z0",
            "ntee_major_group": "Z",
            "decision_source": "classifier",
            "prob_raw": 0.5,
            "prob_calibrated": 0.6,
            "pred_label": 1,
            "pred_label_maxf1": 0,
            "pred_label_baserate": 0,
        },
    ]
    return pd.DataFrame(rows)


def test_ntee_descriptives_covers_all_26_letters_with_no_missing_marker() -> None:
    """The descriptives frame has 26 A-Z rows and no '?' row."""
    out = _ntee_descriptives(_ntee_descriptives_fixture())

    assert list(out.columns) == [
        "ntee_major_group",
        "label",
        "n",
        "n_scored",
        "mean_prob_raw",
        "mean_prob_calibrated",
        "n_pred_label",
        "share_pred_label",
        "n_pred_label_maxf1",
        "share_pred_label_maxf1",
        "n_pred_label_baserate",
        "share_pred_label_baserate",
    ]
    assert sorted(out["ntee_major_group"]) == [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    assert "?" not in set(out["ntee_major_group"])


def test_ntee_descriptives_folds_missing_ntee_into_z() -> None:
    """'?' and native 'Z' rows combine into a single Z group."""
    out = _ntee_descriptives(_ntee_descriptives_fixture())
    z_row = out.loc[out["ntee_major_group"] == "Z"].iloc[0]

    # 2 folded "?" rows (unscored) + 1 native Z row (scored).
    assert int(z_row["n"]) == 3
    assert int(z_row["n_scored"]) == 1
    assert z_row["mean_prob_calibrated"] == pytest.approx(0.6)


def test_ntee_descriptives_n_scored_matches_decision_source_identity() -> None:
    """n_scored equals n minus that group's rule-routed rows, per group."""
    fixture = _ntee_descriptives_fixture()
    out = _ntee_descriptives(fixture)

    fixture_folded = fixture.copy()
    fixture_folded["ntee_major_group"] = fixture_folded["ntee_major_group"].replace(
        "?", "Z"
    )
    is_rule_routed = fixture_folded["decision_source"].str.startswith("rule_")
    expected_scored = (
        fixture_folded.assign(_scored=~is_rule_routed)
        .groupby("ntee_major_group")["_scored"]
        .sum()
    )

    for ntee, expected in expected_scored.items():
        actual = out.loc[out["ntee_major_group"] == ntee, "n_scored"].iloc[0]
        assert actual == expected, ntee


def test_ntee_descriptives_shares_use_full_group_as_denominator() -> None:
    """Label shares divide by n (all rows), not n_scored."""
    out = _ntee_descriptives(_ntee_descriptives_fixture())
    a_row = out.loc[out["ntee_major_group"] == "A"].iloc[0]

    assert int(a_row["n"]) == 4
    assert int(a_row["n_pred_label"]) == 3
    assert a_row["share_pred_label"] == pytest.approx(3 / 4)


def test_ntee_label_table_covers_every_group_present_in_fixture() -> None:
    """The packaged label table has an entry for every NTEE letter used."""
    labels = load_ntee_labels()
    present = set(_ntee_descriptives_fixture()["ntee_major_group"]) - {"?"}

    assert present <= set(labels["ntee_major_group"])
    assert labels.loc[labels["ntee_major_group"] == "Z", "label"].iloc[0] == (
        "Unknown or missing NTEE"
    )


def test_run_prevalence_ntee_descriptives_flag_independent_of_per_ntee(
    tiny_config,
    tiny_registry,
) -> None:
    """The ntee_descriptives flag gates its artifact without affecting per_ntee."""
    _write_prevalence_inputs(tiny_registry)
    tiny_config.prevalence.per_ntee = False
    tiny_config.prevalence.ntee_descriptives = True

    run_prevalence(tiny_config, tiny_registry)

    assert tiny_registry.ntee_descriptives.exists()
    by_ntee = pd.read_csv(tiny_registry.prevalence_by_ntee)
    assert by_ntee.empty

    tiny_config.prevalence.per_ntee = True
    tiny_config.prevalence.ntee_descriptives = False
    tiny_registry.ntee_descriptives.unlink()

    run_prevalence(tiny_config, tiny_registry)

    assert not tiny_registry.ntee_descriptives.exists()
    by_ntee = pd.read_csv(tiny_registry.prevalence_by_ntee)
    assert not by_ntee.empty
