"""Tests for expanding deduplicated stage-08 predictions to raw EIN2 rows."""

from __future__ import annotations

import pandas as pd
import pytest

from binary_classifier.inference import predict as predict_mod


def test_missing_mission_sentinel_is_not_empty_string() -> None:
    normalized = predict_mod._normalize_mission_key(pd.Series([None, ""]))

    assert predict_mod._MISSING_MISSION_SENTINEL != ""
    assert normalized.tolist() == [predict_mod._MISSING_MISSION_SENTINEL, ""]


def test_expand_back_labels_nan_and_empty_text_rows(tiny_config) -> None:
    raw = pd.DataFrame(
        [
            _mission_row("R001", None),
            _mission_row("R002", ""),
            _mission_row("R003", None),
        ]
    )
    deduped = pd.DataFrame([_mission_row("D001", None), _mission_row("D002", "")])
    predictions = pd.DataFrame(
        [_prediction_row("D001", 1), _prediction_row("D002", 0)],
        columns=predict_mod._PREDICTION_COLUMNS,
    )

    full = predict_mod._expand_predictions_to_raw(
        tiny_config,
        raw=raw,
        deduped=deduped,
        predictions=predictions,
    )

    assert full["EIN2"].tolist() == ["R001", "R002", "R003"]
    assert full["pred_label"].tolist() == [1, 0, 1]
    assert int(full["pred_label"].isna().sum()) == 0


def test_expand_back_many_to_one_collision_raises(tiny_config) -> None:
    raw = pd.DataFrame([_mission_row("R001", "same")])
    deduped = pd.DataFrame([_mission_row("D001", "same"), _mission_row("D002", "same")])
    predictions = pd.DataFrame(
        [_prediction_row("D001", 1), _prediction_row("D002", 0)],
        columns=predict_mod._PREDICTION_COLUMNS,
    )

    with pytest.raises(pd.errors.MergeError, match="many-to-one"):
        predict_mod._expand_predictions_to_raw(
            tiny_config,
            raw=raw,
            deduped=deduped,
            predictions=predictions,
        )


def test_predictions_full_completeness_assertion_fires_on_dropped_ein() -> None:
    full = pd.DataFrame([{"EIN2": "R001", "pred_label": 1}])

    with pytest.raises(ValueError, match="1 rows for 2 input EIN2s"):
        predict_mod._validate_predictions_full(pd.Series(["R001", "R002"]), full)


def test_expand_back_does_not_mutate_deduped_predictions(tiny_config) -> None:
    raw = pd.DataFrame([_mission_row("R001", "alpha")])
    deduped = pd.DataFrame([_mission_row("D001", "alpha")])
    predictions = pd.DataFrame(
        [_prediction_row("D001", 1)],
        columns=predict_mod._PREDICTION_COLUMNS,
    )
    before = predictions.copy(deep=True)

    predict_mod._expand_predictions_to_raw(
        tiny_config,
        raw=raw,
        deduped=deduped,
        predictions=predictions,
    )

    pd.testing.assert_frame_equal(predictions, before)


def _mission_row(ein2: str, text: str | None) -> dict:
    return {
        "EIN2": ein2,
        "LONGEST_MISSION": text,
        "mission_text": text,
        "ntee_major_group": "P",
        "is_truncated": False,
        "NTEE_IRS": "P20",
        "data_source": "synthetic",
    }


def _prediction_row(ein2: str, label: int) -> dict:
    return {
        "EIN2": ein2,
        "pred_label": label,
        "pred_label_maxf1": label,
        "pred_label_baserate": label,
        "prob_raw": 0.8 if label else 0.2,
        "prob_calibrated": 0.8 if label else 0.2,
        "decision_source": "classifier",
        "tier": "HIGH",
        "Q": 5.0,
        "ntee_major_group": "P",
        "model_id": "stub",
        "checkpoint_sha256": "stub-sha",
        "calibrator_method": "platt",
        "calibrator_params_hash": "0" * 64,
        "threshold": 0.5,
        "threshold_maxf1": 0.7,
        "threshold_baserate": 0.9,
        "inference_date": "2026-01-01T00:00:00+00:00",
        "pipeline_version": "test",
        "config_hash": "1" * 64,
    }
