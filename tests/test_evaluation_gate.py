"""Tests for T3.1: G3 frozen-test unlock gate."""

import json

import pandas as pd

from binary_classifier.qc.preflight import validate_gates


def _write_stage_07_prereqs(registry) -> None:
    """Write valid G1 test labels and G4 anchor labels."""
    registry.gold_coding_template.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"EIN2": "G0001", "split": "test", "text": "t", "human_label": 1},
            {"EIN2": "G0002", "split": "test", "text": "t", "human_label": 0},
        ],
    ).to_csv(registry.gold_coding_template, index=False)

    registry.anchor_coding_template.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"EIN2": "A0001", "tier": "high", "text": "t", "human_label": 1},
            {"EIN2": "A0002", "tier": "low", "text": "t", "human_label": 0},
        ],
    ).to_csv(registry.anchor_coding_template, index=False)


def _write_selected_model(registry, sha: str = "selected-sha") -> None:
    """Write the selected-model artifact consumed by G3."""
    registry.selected_model.parent.mkdir(parents=True, exist_ok=True)
    registry.selected_model.write_text(
        json.dumps(
            {
                "checkpoint_relpath": "checkpoints/example/model.safetensors",
                "checkpoint_sha256": sha,
            },
        ),
    )


def _write_unlock(
    cfg,
    registry,
    *,
    confirmed: bool = True,
    sha: str = "selected-sha",
    acceptance: dict | None = None,
) -> None:
    """Write the human test-unlock artifact consumed by G3."""
    registry.test_unlock.parent.mkdir(parents=True, exist_ok=True)
    registry.test_unlock.write_text(
        json.dumps(
            {
                "confirmed": confirmed,
                "checkpoint": "checkpoints/example/model.safetensors",
                "checkpoint_sha256": sha,
                "acceptance": acceptance or cfg.evaluation.acceptance.model_dump(),
                "rationale": "unit-test unlock",
            },
        ),
    )


def test_missing_test_unlock_flags_g3(tiny_config, tiny_registry) -> None:
    _write_stage_07_prereqs(tiny_registry)

    problems = validate_gates(tiny_config, tiny_registry, {"07"})

    assert len(problems) == 1
    assert problems[0].startswith("G3")
    assert "no confirmed test unlock" in problems[0]


def test_unconfirmed_test_unlock_flags_g3(tiny_config, tiny_registry) -> None:
    _write_stage_07_prereqs(tiny_registry)
    _write_selected_model(tiny_registry)
    _write_unlock(tiny_config, tiny_registry, confirmed=False)

    problems = validate_gates(tiny_config, tiny_registry, {"07"})

    assert len(problems) == 1
    assert problems[0].startswith("G3")
    assert "not confirmed" in problems[0]


def test_acceptance_drift_flags_g3(tiny_config, tiny_registry) -> None:
    _write_stage_07_prereqs(tiny_registry)
    _write_selected_model(tiny_registry)
    acceptance = tiny_config.evaluation.acceptance.model_dump()
    acceptance["min_pr_auc"] = 0.99
    _write_unlock(tiny_config, tiny_registry, acceptance=acceptance)

    problems = validate_gates(tiny_config, tiny_registry, {"07"})

    assert len(problems) == 1
    assert problems[0].startswith("G3")
    assert "acceptance snapshot differs" in problems[0]
    assert "min_pr_auc" in problems[0]


def test_checkpoint_sha_mismatch_flags_g3(tiny_config, tiny_registry) -> None:
    _write_stage_07_prereqs(tiny_registry)
    _write_selected_model(tiny_registry, sha="selected-sha")
    _write_unlock(tiny_config, tiny_registry, sha="unlock-sha")

    problems = validate_gates(tiny_config, tiny_registry, {"07"})

    assert len(problems) == 1
    assert problems[0].startswith("G3")
    assert "checkpoint_sha256 does not match" in problems[0]


def test_confirmed_matching_test_unlock_passes_g3(tiny_config, tiny_registry) -> None:
    _write_stage_07_prereqs(tiny_registry)
    _write_selected_model(tiny_registry)
    _write_unlock(tiny_config, tiny_registry)

    assert validate_gates(tiny_config, tiny_registry, {"07"}) == []
