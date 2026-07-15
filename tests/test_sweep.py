"""Tests for stage-06 run-matrix and selection helpers."""

import json
from datetime import UTC, datetime

import pandas as pd

from binary_classifier.config import BinaryClassifierConfig, EncoderArm
from binary_classifier.train import sweep


def test_default_run_matrix_counts_and_final_resume(tiny_registry) -> None:
    """Default config enumerates the documented PR-2 matrix counts."""
    cfg = BinaryClassifierConfig()
    recommendation = {
        "encoder_id": "microsoft/deberta-v3-base",
        "targets": "soft",
        "arm": "default",
        "representative_seed": 44,
    }

    specs = sweep.build_run_matrix(cfg, final=True, recommendation=recommendation)

    assert sum(spec.phase == "baseline" for spec in specs) == 2
    assert sum(spec.phase == "curve" for spec in specs) == 2
    assert sum(spec.phase == "sweep" for spec in specs) == 12
    final_specs = [spec for spec in specs if spec.phase == "final"]
    assert len(final_specs) == 5

    for spec in specs:
        if spec.phase == "sweep" and spec.cell_key == ("microsoft/deberta-v3-base", "soft", "default"):
            path = sweep.run_metrics_path(tiny_registry, spec)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_row(spec, 0.7, 0.6)))

    pending_finals = [
        spec for spec in final_specs if not sweep.run_metrics_path(tiny_registry, spec).exists()
    ]
    assert len(pending_finals) == 2


def test_resume_skips_metrics_json(monkeypatch, tiny_config, tiny_registry) -> None:
    """A metrics.json sentinel prevents re-running that spec."""
    spec = sweep.build_run_matrix(tiny_config)[2]
    path = sweep.run_metrics_path(tiny_registry, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _row(spec, 0.8, 0.7)
    path.write_text(json.dumps(expected))

    def fail_finetune(*args, **kwargs):
        raise AssertionError("resume should not call finetune")

    monkeypatch.setattr(sweep, "finetune", fail_finetune)
    rows = sweep.execute_run_matrix(
        tiny_config,
        tiny_registry,
        [spec],
        _frame(12),
        _frame(4, start=20),
        _validation_frame(),
    )

    assert rows == [expected]


def test_selection_report_clear_win_and_schema(tiny_config, tiny_registry) -> None:
    """A PR-AUC advantage larger than seed SD beats the simpler arm."""
    rows = []
    for seed, score in zip([42, 43, 44], [0.70, 0.71, 0.69], strict=True):
        rows.append(_row(_spec("soft", "default", seed), score, 0.60))
    for seed, score in zip([42, 43, 44], [0.84, 0.85, 0.83], strict=True):
        rows.append(_row(_spec("hard", "hard", seed), score, 0.75))

    report = sweep.write_selection_report(
        rows,
        tiny_registry.selection_report,
        cfg=tiny_config,
        registry=tiny_registry,
    )

    assert tiny_registry.selection_report.exists()
    assert report["schema_version"] == 1
    assert report["recommendation"]["targets"] == "hard"
    assert report["recommendation"]["arm"] == "hard"
    assert report["summaries"][0]["validation_pr_auc"]["ci_lower"] is not None
    assert set(report["selected_model_skeleton"]) == {
        "checkpoint_relpath",
        "checkpoint_sha256",
        "tokenizer_id",
        "encoder_id",
        "targets",
        "recipe_snapshot",
        "selected_at",
    }


def test_selection_tie_goes_to_simpler(tiny_config, tiny_registry) -> None:
    """A small advantage inside across-seed SD keeps the simpler soft arm."""
    rows = []
    for seed, score in zip([42, 43, 44], [0.65, 0.70, 0.75], strict=True):
        rows.append(_row(_spec("soft", "default", seed), score, 0.60))
    for seed, score in zip([42, 43, 44], [0.72, 0.72, 0.72], strict=True):
        rows.append(_row(_spec("hard", "hard", seed), score, 0.62))

    report = sweep.write_selection_report(
        rows,
        tiny_registry.selection_report,
        cfg=tiny_config,
        registry=tiny_registry,
    )

    assert report["recommendation"]["targets"] == "soft"
    assert report["tie_rule_verdicts"][-1]["reason"] == "tie_to_simpler"


def _spec(targets: str, arm: str, seed: int) -> sweep.RunSpec:
    encoder = EncoderArm(id="microsoft/deberta-v3-base", arm="primary")
    return sweep.RunSpec(
        kind="encoder",
        phase="sweep",
        run_id=f"deberta-{targets}-{arm}-{seed}",
        seed=seed,
        train_fraction=1.0,
        model=encoder.id,
        targets=targets,
        arm=arm,
        encoder=encoder,
    )


def _row(spec: sweep.RunSpec, pr_auc: float, minority_f1: float) -> dict:
    bundle = {"pr_auc": pr_auc, "f1": minority_f1}
    return {
        "run_id": spec.run_id,
        "model": spec.model,
        "targets": spec.targets,
        "arm": spec.arm,
        "train_fraction": spec.train_fraction,
        "n_train": 4,
        "seed": spec.seed,
        "dev": bundle,
        "validation": bundle,
        "wall_seconds": 0.0,
        "precision": "fp32",
        "device": "cpu",
        "git_sha": "unknown",
        "config_hash": "unknown",
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _frame(n_rows: int, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EIN2": [f"e{i}" for i in range(start, start + n_rows)],
            "text": [f"mission text {i}" for i in range(start, start + n_rows)],
            "ntee_major_group": ["A"] * n_rows,
            "p_pos": [0.1 if i % 2 == 0 else 0.9 for i in range(n_rows)],
            "hard_label": [i % 2 for i in range(n_rows)],
        }
    )


def _validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EIN2": ["v0", "v1"],
            "text": ["validation zero", "validation one"],
            "human_label": [0, 1],
        }
    )
