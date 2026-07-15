"""Tests for frozen-safe calibrator regeneration."""

import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "regen_calibrator.py"
_SPEC = importlib.util.spec_from_file_location("regen_calibrator_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
regen_calibrator = _MODULE.regen_calibrator


def test_regen_calibrator_touches_only_calibrator(tiny_config, tiny_registry) -> None:
    tiny_config.evaluation.threshold_policy = "precision_floor"
    tiny_config.evaluation.precision_floor = 0.65
    pd.DataFrame(
        {
            "EIN2": ["1", "2", "3", "4", "5"],
            "prob_calibrated_oof": [0.9, 0.8, 0.7, 0.4, 0.2],
            "human_label": [1, 0, 1, 0, 1],
            "sample_prob": [1.0] * 5,
        }
    ).to_parquet(tiny_registry.anchor_oof_scores, index=False)
    tiny_registry.calibrator_path.write_text(
        json.dumps({"method": "platt", "params": {"a": 1.0, "b": 0.0}}) + "\n"
    )
    tiny_registry.test_evaluation.write_text('{"do_not_touch": true}\n')
    test_before = tiny_registry.test_evaluation.read_bytes()
    anchor_before = tiny_registry.anchor_oof_scores.read_bytes()

    payload = regen_calibrator(tiny_config, tiny_registry)

    assert tiny_registry.test_evaluation.read_bytes() == test_before
    assert tiny_registry.anchor_oof_scores.read_bytes() == anchor_before
    assert payload["threshold"] == pytest.approx(0.7)
    assert payload["max_f1_threshold"] == pytest.approx(0.2)
    assert payload["achieved_precision"] == pytest.approx(2 / 3)
    assert payload["achieved_recall"] == pytest.approx(2 / 3)
    assert len(payload["pr_curve_points"]) == 5
