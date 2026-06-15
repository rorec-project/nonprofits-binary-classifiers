"""Evaluation utilities for calibration and model assessment."""

from binary_classifier.evaluation.calibration import (
    apply_calibration,
    calibration_metrics,
    compute_brier,
    compute_log_loss,
    crossfit_calibrate,
    deserialize_calibrator,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
    reliability_curve,
    serialize_calibrator,
)

__all__ = [
    "apply_calibration",
    "calibration_metrics",
    "compute_brier",
    "compute_log_loss",
    "crossfit_calibrate",
    "deserialize_calibrator",
    "expected_calibration_error",
    "fit_platt",
    "fit_temperature",
    "reliability_curve",
    "serialize_calibrator",
]
