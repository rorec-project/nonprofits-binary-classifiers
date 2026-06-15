"""Tests for stage 07 evaluation."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from binary_classifier.evaluation import evaluate as evaluate_mod
from binary_classifier.evaluation.evaluate import run_evaluation


class TextPredictor:
    """Tiny deterministic predictor keyed off religious words."""

    def predict_proba(self, texts):
        p1 = np.asarray(
            [0.9 if "church" in str(text).lower() else 0.1 for text in texts],
            dtype=float,
        )
        return np.column_stack([1.0 - p1, p1])


def test_run_evaluation_happy_path_and_ordering(
    monkeypatch,
    tiny_config,
    tiny_registry,
) -> None:
    """Write calibration artifacts before the internal frozen-test reader runs."""
    tiny_config.evaluation.crossfit_folds = 2
    tiny_config.evaluation.calibration_methods = ["platt"]
    tiny_config.evaluation.bootstrap_resamples = 20
    tiny_config.evaluation.ece_bins = 2
    tiny_config.evaluation.acceptance.min_pr_auc = 0.5
    tiny_config.evaluation.acceptance.min_minority_f1_ci_lower = 0.0
    tiny_config.evaluation.acceptance.max_ece = 1.0

    missions = _write_gate_artifacts(tiny_config, tiny_registry)
    monkeypatch.setattr(evaluate_mod, "load_missions", lambda cfg: missions)
    original_reader = evaluate_mod._read_frozen_test_labels
    calls = []

    def reader(registry, mission_frame):
        calls.append("test_read")
        assert registry.calibrator_path.exists()
        assert registry.anchor_oof_scores.exists()
        return original_reader(registry, mission_frame)

    monkeypatch.setattr(evaluate_mod, "_read_frozen_test_labels", reader)

    run_evaluation(tiny_config, tiny_registry, predictor=TextPredictor())

    assert calls == ["test_read"]
    assert tiny_registry.calibrator_path.exists()
    assert tiny_registry.anchor_oof_scores.exists()
    assert tiny_registry.rule_validation.exists()
    assert tiny_registry.test_evaluation.exists()

    calibrator = json.loads(tiny_registry.calibrator_path.read_text())
    assert calibrator["fitted_on"] == "anchor"
    assert calibrator["threshold_policy"] == "precision_floor"
    scores = pd.read_parquet(tiny_registry.anchor_oof_scores)
    assert list(scores.columns) == [
        "EIN2",
        "prob_raw",
        "prob_calibrated_oof",
        "human_label",
        "tier",
        "sample_prob",
    ]
    report = json.loads(tiny_registry.test_evaluation.read_text())
    assert report["acceptance"]["passed"] is True
    assert report["metadata"]["checkpoint_sha256"]


def test_run_evaluation_one_shot_refusal(
    monkeypatch,
    tiny_config,
    tiny_registry,
) -> None:
    """An existing frozen-test report blocks any re-run."""
    tiny_config.evaluation.crossfit_folds = 2
    tiny_config.evaluation.calibration_methods = ["platt"]
    tiny_config.evaluation.acceptance.max_ece = 1.0
    missions = _write_gate_artifacts(tiny_config, tiny_registry)
    monkeypatch.setattr(evaluate_mod, "load_missions", lambda cfg: missions)
    tiny_registry.test_evaluation.write_text("{}\n")

    with pytest.raises(RuntimeError, match="delete it explicitly to re-run"):
        run_evaluation(tiny_config, tiny_registry, predictor=TextPredictor())


def test_run_evaluation_acceptance_failure_raises(
    monkeypatch,
    tiny_config,
    tiny_registry,
) -> None:
    """Acceptance failures are loud after writing the audit report."""
    tiny_config.evaluation.crossfit_folds = 2
    tiny_config.evaluation.calibration_methods = ["platt"]
    tiny_config.evaluation.bootstrap_resamples = 20
    tiny_config.evaluation.ece_bins = 2
    tiny_config.evaluation.acceptance.min_pr_auc = 1.01
    tiny_config.evaluation.acceptance.min_minority_f1_ci_lower = 0.0
    tiny_config.evaluation.acceptance.max_ece = 1.0
    missions = _write_gate_artifacts(tiny_config, tiny_registry)
    monkeypatch.setattr(evaluate_mod, "load_missions", lambda cfg: missions)

    with pytest.raises(RuntimeError, match="EVALUATION GATE FAILED"):
        run_evaluation(tiny_config, tiny_registry, predictor=TextPredictor())

    assert tiny_registry.test_evaluation.exists()
    report = json.loads(tiny_registry.test_evaluation.read_text())
    assert report["acceptance"]["passed"] is False


def _write_gate_artifacts(tiny_config, registry) -> pd.DataFrame:
    checkpoint = registry.checkpoints_dir / "tiny" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"tiny checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    relpath = checkpoint.relative_to(registry.models_dir)
    registry.selected_model.write_text(
        json.dumps(
            {
                "checkpoint_relpath": str(relpath),
                "checkpoint_sha256": checkpoint_sha,
                "tokenizer_id": "tiny-tokenizer",
                "encoder_id": "tiny-encoder",
            }
        )
    )
    registry.test_unlock.write_text(
        json.dumps(
            {
                "confirmed": True,
                "checkpoint": str(relpath),
                "checkpoint_sha256": checkpoint_sha,
                "acceptance": tiny_config.evaluation.acceptance.model_dump(),
                "rationale": "unit test fixture",
            }
        )
    )

    anchor_rows = []
    mission_rows = []
    for i in range(8):
        label = i % 2
        ein2 = f"A{i:04d}"
        text = f"church worship mission {i}" if label else f"food pantry {i}"
        anchor_rows.append(
            {"EIN2": ein2, "tier": "LOW", "text": text, "human_label": label}
        )
        mission_rows.append(_mission_row(ein2, text))
    pd.DataFrame(anchor_rows).to_csv(registry.anchor_coding_template, index=False)
    pd.DataFrame(
        {
            "EIN2": [row["EIN2"] for row in anchor_rows],
            "stratum": ["LOW|A"] * len(anchor_rows),
            "tier": ["LOW"] * len(anchor_rows),
            "ntee_major_group": ["A"] * len(anchor_rows),
            "sample_prob": [0.5] * len(anchor_rows),
            "split": ["anchor"] * len(anchor_rows),
        }
    ).to_csv(registry.anchor_manifest, index=False)

    test_rows = []
    for i in range(6):
        label = i % 2
        ein2 = f"T{i:04d}"
        text = f"church outreach {i}" if label else f"arts education {i}"
        test_rows.append(
            {"EIN2": ein2, "split": "test", "text": text, "human_label": label}
        )
        mission_rows.append(_mission_row(ein2, text))
    pd.DataFrame(test_rows).to_csv(registry.gold_coding_template, index=False)
    return pd.DataFrame(mission_rows)


def _mission_row(ein2: str, text: str) -> dict[str, object]:
    return {
        "EIN2": ein2,
        "mission_text": text,
        "ntee_major_group": "A",
        "is_truncated": False,
        "NTEE_IRS": "A20",
        "data_source": "synthetic",
    }
