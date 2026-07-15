"""Tests for anchor calibration utilities."""

import json
import math

import numpy as np
import pytest

from binary_classifier.evaluation import calibration


def test_platt_absorbs_synthetic_prior_shift_better_than_temperature() -> None:
    """Platt's intercept handles enrichment-induced prior shift; temperature cannot."""
    rng = np.random.default_rng(2025)
    latent = rng.normal(size=4_000)
    target_prob = _probabilities_with_mean(latent, mean=0.13)
    raw_scores = _sigmoid(
        _logit(target_prob) + _logit(np.array([0.30]))[0] - _logit(np.array([0.13]))[0],
    )
    labels = rng.binomial(1, target_prob).astype(int)

    _oof, report = calibration.crossfit_calibrate(
        raw_scores,
        labels,
        folds=5,
        methods=["platt", "temperature"],
        seed=17,
        ece_bins=10,
    )

    platt_brier = report["methods"]["platt"]["mean_oof_brier"]
    temperature_brier = report["methods"]["temperature"]["mean_oof_brier"]
    assert labels.mean() == pytest.approx(0.13, abs=0.01)
    assert platt_brier < temperature_brier
    assert report["winner"] == "platt"


def test_metrics_have_hand_checked_values() -> None:
    """Brier, log-loss, ECE, and reliability bins are deterministic."""
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.8, 0.9]

    metrics = calibration.calibration_metrics(labels, probabilities, bins=2)

    expected_log_loss = -sum(
        [
            math.log(1 - probabilities[0]),
            math.log(1 - probabilities[1]),
            math.log(probabilities[2]),
            math.log(probabilities[3]),
        ],
    ) / 4
    assert metrics["brier"] == pytest.approx(0.055)
    assert metrics["log_loss"] == pytest.approx(expected_log_loss)
    assert metrics["ece"] == pytest.approx(0.2)
    assert metrics["reliability_curve"] == [
        {
            "bin": 0,
            "lower": 0.0,
            "upper": 0.5,
            "count": 2,
            "mean_predicted": pytest.approx(0.25),
            "observed_fraction": 0.0,
            "gap": pytest.approx(0.25),
        },
        {
            "bin": 1,
            "lower": 0.5,
            "upper": 1.0,
            "count": 2,
            "mean_predicted": pytest.approx(0.85),
            "observed_fraction": 1.0,
            "gap": pytest.approx(-0.15),
        },
    ]


def test_crossfit_never_scores_rows_with_in_fold_fit(monkeypatch) -> None:
    """Each held-out score is calibrated by parameters fit on other rows."""
    train_sets = []
    scored_sets = []

    def fake_fit(scores, labels):
        del labels
        train_sets.append(tuple(scores))
        return {"a": float(len(train_sets)), "b": 0.0}

    def fake_apply(scores, method, params):
        del method
        train_scores = set(train_sets[int(params["a"]) - 1])
        heldout_scores = set(scores)
        assert train_scores.isdisjoint(heldout_scores)
        scored_sets.append(tuple(scores))
        return [0.5] * len(scores)

    monkeypatch.setattr(calibration, "fit_platt", fake_fit)
    monkeypatch.setattr(calibration, "apply_calibration", fake_apply)

    scores = np.linspace(0.05, 0.95, 12)
    labels = np.array([0, 1] * 6)
    oof, report = calibration.crossfit_calibrate(
        scores,
        labels,
        folds=3,
        methods=["platt"],
        seed=123,
        ece_bins=5,
    )

    assert oof == [0.5] * len(scores)
    assert len(scored_sets) == 3
    assert len(train_sets) == 4  # three folds plus the all-anchor deployment refit
    assert report["deployed"] == {
        "method": "platt",
        "params": {"a": 4.0, "b": 0.0},
        "fitted_on": "anchor",
    }


def test_calibrator_and_crossfit_outputs_round_trip_through_json() -> None:
    """Fitted params and cross-fit reports are JSON-serializable artifacts."""
    scores = [0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95]
    labels = [0, 0, 0, 1, 0, 1, 1, 1]
    params = calibration.fit_platt(scores, labels)

    payload = calibration.serialize_calibrator("platt", params)
    method, loaded_params = calibration.deserialize_calibrator(payload)
    assert method == "platt"
    assert loaded_params == pytest.approx(params)
    assert calibration.apply_calibration(scores, method, loaded_params) == pytest.approx(
        calibration.apply_calibration(scores, "platt", params),
    )

    oof, report = calibration.crossfit_calibrate(
        scores,
        labels,
        folds=2,
        methods=["platt", "temperature"],
        seed=42,
        ece_bins=4,
    )
    round_tripped_oof = json.loads(json.dumps(oof))
    round_tripped_report = json.loads(json.dumps(report))
    assert round_tripped_oof == pytest.approx(oof)
    assert round_tripped_report["winner"] in {"platt", "temperature"}
    assert round_tripped_report["deployed"]["method"] == round_tripped_report["winner"]


def _probabilities_with_mean(latent: np.ndarray, *, mean: float) -> np.ndarray:
    lower = -10.0
    upper = 10.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        probabilities = _sigmoid(1.6 * latent + midpoint)
        if probabilities.mean() > mean:
            upper = midpoint
        else:
            lower = midpoint
    return _sigmoid(1.6 * latent + (lower + upper) / 2)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    return np.log(probabilities / (1.0 - probabilities))
