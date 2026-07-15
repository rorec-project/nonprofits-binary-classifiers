"""Tests for the standalone stage-06 training CLI."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from binary_classifier.train import trainer as trainer_mod


def _load_train_cli():
    """Load ``scripts/06_train.py`` despite the numeric filename."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "06_train.py"
    spec = importlib.util.spec_from_file_location("stage06_train_cli", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("argv", "expected_sweep", "expected_final"),
    [
        (["06_train.py"], True, False),
        (["06_train.py", "--sweep"], True, False),
        (["06_train.py", "--final"], False, True),
        (["06_train.py", "--sweep", "--final"], True, True),
    ],
)
def test_stage_06_cli_sweep_final_flags(
    monkeypatch,
    argv,
    expected_sweep,
    expected_final,
) -> None:
    """``--final`` alone means final-only while no flags keep the default sweep."""
    module = _load_train_cli()
    cfg = SimpleNamespace(training=SimpleNamespace())
    registry = object()
    calls = []

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(module, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        module,
        "PathRegistry",
        SimpleNamespace(from_config=lambda cfg_arg: registry),
    )
    monkeypatch.setattr(
        module,
        "run_training",
        lambda cfg_arg, registry_arg, **kwargs: calls.append(
            (cfg_arg, registry_arg, kwargs)
        ),
    )

    module.main()

    assert len(calls) == 1
    cfg_arg, registry_arg, kwargs = calls[0]
    assert cfg_arg is cfg
    assert registry_arg is registry
    assert kwargs["sweep"] is expected_sweep
    assert kwargs["final"] is expected_final


def test_run_training_final_only_requires_existing_selection_before_loading(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """Final-only refit should fail before loading data when no sweep report exists."""

    def fail_load_inputs(*args, **kwargs):
        raise AssertionError("training inputs should not be loaded")

    monkeypatch.setattr(trainer_mod, "_load_training_inputs", fail_load_inputs)

    with pytest.raises(FileNotFoundError, match="Run stage 06 sweep first"):
        trainer_mod.run_training(tiny_config, tiny_registry, sweep=False, final=True)


def test_run_training_final_only_requires_usable_recommendation_before_loading(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """A malformed selection report should fail before any training work."""
    tiny_registry.selection_report.parent.mkdir(parents=True, exist_ok=True)
    tiny_registry.selection_report.write_text(json.dumps({"recommendation": None}))

    def fail_load_inputs(*args, **kwargs):
        raise AssertionError("training inputs should not be loaded")

    monkeypatch.setattr(trainer_mod, "_load_training_inputs", fail_load_inputs)

    with pytest.raises(ValueError, match="usable recommendation"):
        trainer_mod.run_training(tiny_config, tiny_registry, sweep=False, final=True)


def test_run_training_sweep_and_final_chains_after_selection_report(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """A fresh sweep+final call writes selection before final-only refit."""
    tiny_config.training.arms = []
    tiny_config.aggregation.comparison_arms = []
    frame = pd.DataFrame(
        {
            "EIN2": ["00-1", "00-2", "00-3", "00-4"],
            "text": ["church", "food", "ministry", "housing"],
            "p_pos": [0.9, 0.1, 0.8, 0.2],
            "hard_label": [1, 0, 1, 0],
            "ntee_major_group": ["X", "P", "X", "P"],
        },
    )
    validation = pd.DataFrame(
        {"EIN2": ["v1", "v2"], "text": ["church", "food"], "human_label": [1, 0]},
    )
    recommendation = {
        "encoder_id": tiny_config.training.encoders[0].id,
        "targets": "soft",
        "arm": "default",
    }
    report = {"recommendation": recommendation, "selected_model_skeleton": {}}
    calls: list[tuple[bool, bool, bool]] = []

    monkeypatch.setattr(
        trainer_mod,
        "_load_training_inputs",
        lambda cfg, registry: (frame, validation),
    )
    monkeypatch.setattr(
        trainer_mod,
        "split_dev",
        lambda source, fraction, seed: (source, source),
    )

    def fake_build_run_matrix(*args, **kwargs):
        calls.append(
            (
                bool(kwargs["sweep"]),
                bool(kwargs["final"]),
                kwargs["recommendation"] is not None,
            ),
        )
        phase = "sweep" if kwargs["sweep"] else "final"
        return [SimpleNamespace(phase=phase, cell_key="cell")]

    def fake_write_selection_report(*args, **kwargs):
        tiny_registry.selection_report.parent.mkdir(parents=True, exist_ok=True)
        tiny_registry.selection_report.write_text(json.dumps(report))
        return report

    monkeypatch.setattr(trainer_mod, "build_run_matrix", fake_build_run_matrix)
    monkeypatch.setattr(trainer_mod, "execute_run_matrix", lambda *a, **k: [])
    monkeypatch.setattr(trainer_mod, "write_selection_report", fake_write_selection_report)
    monkeypatch.setattr(trainer_mod, "print_selected_model_skeleton", lambda report: None)

    trainer_mod.run_training(tiny_config, tiny_registry, sweep=True, final=True)

    assert calls == [(True, False, False), (False, True, True)]
