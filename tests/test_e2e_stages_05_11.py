"""Slow synthetic E2E route for stages 05 onward.

Later PRs extend this file through stages 09–11.
"""

import json
import runpy
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from binary_classifier.annotate.schema import AnnotationStore, BinaryLabel, LabelRecord, SourceType
from binary_classifier.data import anchor as anchor_mod
from binary_classifier.evaluation import evaluate as evaluate_mod
from binary_classifier.evaluation.evaluate import run_evaluation
from binary_classifier.inference import predict as predict_mod
from binary_classifier.inference.predict import run_inference
from binary_classifier.prevalence.estimate import run_prevalence
from binary_classifier.qc.aggregation_compare import run_aggregation_compare
from binary_classifier.train import data as train_data_mod
from binary_classifier.train import sweep
from binary_classifier.train.trainer import run_training


@pytest.mark.slow
def test_e2e_stages_05_to_08_with_finetune_stub(
    monkeypatch,
    tiny_config,
    tiny_registry,
) -> None:
    """Fabricate stage-04 artifacts, then run stages 05-08 offline."""
    tiny_config.anchor.n = 8
    tiny_config.anchor.min_stratum_frame = 1
    tiny_config.training.baselines = ["tfidf_logreg"]
    tiny_config.training.encoders = [Encoder := tiny_config.training.encoders[0].model_copy(update={"id": "fake/encoder"})]
    tiny_config.training.sweep_seeds = [42]
    tiny_config.training.final_seeds = [42, 43]
    tiny_config.training.curve_fractions = [1.0]
    tiny_config.training.arms = []
    tiny_config.evaluation.crossfit_folds = 2
    tiny_config.evaluation.calibration_methods = ["platt"]
    tiny_config.evaluation.bootstrap_resamples = 20
    tiny_config.evaluation.ece_bins = 2
    tiny_config.evaluation.acceptance.min_pr_auc = 0.5
    tiny_config.evaluation.acceptance.min_minority_f1_ci_lower = 0.0
    tiny_config.evaluation.acceptance.max_ece = 1.0
    tiny_config.prevalence.cross_checks = ["emq"]
    tiny_config.prevalence.ntee_min_n = 2
    tiny_config.q_thresholds.MEDIUM = 0.0

    missions = _missions_frame()
    monkeypatch.setattr(anchor_mod, "load_missions", lambda cfg: missions)
    monkeypatch.setattr(train_data_mod, "load_missions", lambda cfg: missions)
    monkeypatch.setattr(evaluate_mod, "load_missions", lambda cfg: missions)
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    monkeypatch.setattr(sweep, "finetune", _fake_finetune)

    _write_stage04_training_artifacts(tiny_registry)
    _write_coded_gold(tiny_registry)
    _write_confirmed_slate(tiny_registry)
    _write_stage01_manifests(tiny_registry)

    anchor_mod.build_anchor(tiny_config, tiny_registry, force=True)
    anchor_template = pd.read_csv(tiny_registry.anchor_coding_template)
    anchor_template["human_label"] = anchor_template["text"].str.contains("church").astype(int)
    anchor_template.to_csv(tiny_registry.anchor_coding_template, index=False)

    run_training(tiny_config, tiny_registry, final=True)

    assert tiny_registry.anchor_manifest.exists()
    assert tiny_registry.learning_curve_results.exists()
    assert tiny_registry.selection_report.exists()
    report = json.loads(tiny_registry.selection_report.read_text())
    assert report["recommendation"]["encoder_id"] == Encoder.id
    assert report["selected_model_skeleton"]["checkpoint_relpath"]

    selected_model = report["selected_model_skeleton"]
    tiny_registry.selected_model.write_text(json.dumps(selected_model))
    tiny_registry.test_unlock.write_text(
        json.dumps(
            {
                "confirmed": True,
                "checkpoint": selected_model["checkpoint_relpath"],
                "checkpoint_sha256": selected_model["checkpoint_sha256"],
                "acceptance": tiny_config.evaluation.acceptance.model_dump(),
                "rationale": "slow smoke fixture",
            }
        )
    )

    run_evaluation(tiny_config, tiny_registry, predictor=_TextPredictor())

    assert tiny_registry.calibrator_path.exists()
    assert tiny_registry.test_evaluation.exists()

    run_inference(tiny_config, tiny_registry, predictor=_TextPredictor())

    assert tiny_registry.predictions_parquet.exists()
    assert tiny_registry.monitor_scores.exists()
    predictions = pd.read_parquet(tiny_registry.predictions_parquet)
    assert set(predictions["EIN2"].astype(str)) == set(missions["EIN2"].astype(str))
    monitor_scores = json.loads(tiny_registry.monitor_scores.read_text())
    assert monitor_scores["metadata"]["n_monitor"] == 8

    run_prevalence(tiny_config, tiny_registry)

    assert tiny_registry.prevalence_report.exists()
    assert tiny_registry.prevalence_by_ntee.exists()

    stage10 = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "10_visualize.py"),
    )
    stage10["load_missions"] = lambda cfg: missions
    stage10["run_visualization"](tiny_config, tiny_registry)

    assert (tiny_registry.figures_dir / "documentation_curve.png").exists()
    assert (tiny_registry.figures_dir / "prevalence_forest.png").exists()
    for figure_name in (
        "precision_recall_curve.png",
        "frozen_test_confusion_matrices.png",
        "score_distribution_by_tier_label.png",
        "prevalence_decomposition.png",
        "rule_validation_intervals.png",
        "quantification_sensitivity.png",
        "subgroup_performance.png",
    ):
        path = tiny_registry.figures_dir / figure_name
        assert path.exists(), figure_name
        assert path.stat().st_size > 0

    run_aggregation_compare(tiny_config, tiny_registry)

    assert tiny_registry.aggregation_compare.exists()
    compare_report = json.loads(tiny_registry.aggregation_compare.read_text())
    assert set(compare_report["arms"]) == {"majority"}
    assert compare_report["arms"]["majority"]["n_scored"] == 8
    assert "Diagnostic only" in compare_report["sensitivity"]["message"]

    with pytest.raises(RuntimeError, match="delete it explicitly to re-run"):
        run_evaluation(tiny_config, tiny_registry, predictor=_TextPredictor())


def _missions_frame() -> pd.DataFrame:
    rows = []
    for i in range(48):
        is_silver = i < 24
        prefix = "S" if is_silver else "A"
        label = i % 2
        text = (
            f"church prayer worship mission {i}"
            if label
            else f"food shelter arts community mission {i}"
        )
        rows.append(
            {
                "EIN2": f"{prefix}{i:04d}",
                "mission_text": text,
                "ntee_major_group": "A",
                "is_truncated": False,
                "NTEE_IRS": "A20",
                "data_source": "synthetic",
            }
        )
    return pd.DataFrame(rows)


def _write_stage04_training_artifacts(registry) -> None:
    silver_rows = []
    records = []
    for i in range(24):
        ein2 = f"S{i:04d}"
        label = i % 2
        silver_rows.append(
            {
                "EIN2": ein2,
                "silver_label": label,
                "silver_confidence": 0.9,
                "num_votes": 3,
                "num_abstain": 0,
                "agreement": 1.0,
                "tie": False,
            }
        )
        for source in range(3):
            records.append(
                LabelRecord(
                    EIN2=ein2,
                    source_id=f"m{source}:p0",
                    source_type=SourceType.LLM_PROMPT,
                    model_id=f"m{source}",
                    prompt_id="p0",
                    temperature=0.0,
                    seed=42,
                    binary_label=BinaryLabel.RELIGIOUS if label else BinaryLabel.NONRELIGIOUS,
                    label=float(label),
                    confidence=0.9,
                )
            )
    registry.silver_labels.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(silver_rows).to_csv(registry.silver_labels, index=False)

    for i in range(24, 32):
        ein2 = f"A{i:04d}"
        label = i % 2
        for source in range(3):
            records.append(
                LabelRecord(
                    EIN2=ein2,
                    source_id=f"m{source}:p0",
                    source_type=SourceType.LLM_PROMPT,
                    model_id=f"m{source}",
                    prompt_id="p0",
                    temperature=0.0,
                    seed=42,
                    binary_label=BinaryLabel.RELIGIOUS
                    if label
                    else BinaryLabel.NONRELIGIOUS,
                    label=float(label),
                    confidence=0.9,
                )
            )

    AnnotationStore(registry.annotation_store).append_many(records)


def _write_coded_gold(registry) -> None:
    rows = []
    for i in range(8):
        mission_i = i + 24
        rows.append(
            {
                "EIN2": f"A{mission_i:04d}",
                "split": "validation",
                "text": (
                    f"church prayer worship mission {mission_i}"
                    if mission_i % 2
                    else f"food shelter arts community mission {mission_i}"
                ),
                "human_label": mission_i % 2,
            }
        )
    for i in range(8):
        mission_i = i + 32
        rows.append(
            {
                "EIN2": f"A{mission_i:04d}",
                "split": "test",
                "text": (
                    f"church prayer worship mission {mission_i}"
                    if mission_i % 2
                    else f"food shelter arts community mission {mission_i}"
                ),
                "human_label": mission_i % 2,
            }
        )
    registry.gold_coding_template.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(registry.gold_coding_template, index=False)


def _write_confirmed_slate(registry) -> None:
    registry.production_slate.write_text(
        json.dumps(
            {
                "confirmed": True,
                "models": [{"id": "gpt-4o-mini", "provider": "openai"}],
                "selected": [],
            }
        )
    )


def _write_stage01_manifests(registry) -> None:
    registry.silver_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"EIN2": [f"S{i:04d}" for i in range(24)]}).to_csv(
        registry.silver_manifest,
        index=False,
    )
    pd.DataFrame({"EIN2": [f"A{i:04d}" for i in range(24, 40)]}).to_csv(
        registry.gold_manifest,
        index=False,
    )
    pd.DataFrame({"EIN2": [f"A{i:04d}" for i in range(40, 48)]}).to_csv(
        registry.monitor_manifest,
        index=False,
    )


def _fake_finetune(
    cfg,
    registry,
    encoder,
    train_df,
    dev_df,
    *,
    targets,
    arm,
    train_fraction,
    seed,
    run_root=None,
) -> dict:
    del cfg, dev_df, run_root
    checkpoint_dir = registry.checkpoints_dir / "fake__encoder" / arm / f"s{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"fake model")
    spec = sweep.RunSpec(
        kind="encoder",
        phase="sweep",
        run_id=f"fake-{targets}-{arm}-{seed}",
        seed=seed,
        train_fraction=train_fraction,
        model=encoder.id,
        targets=targets,
        arm=arm,
        encoder=encoder,
    )
    bundle = {"pr_auc": 0.8, "f1": 0.7}
    row = {
        "run_id": spec.run_id,
        "model": spec.model,
        "targets": spec.targets,
        "arm": spec.arm,
        "train_fraction": spec.train_fraction,
        "n_train": len(train_df),
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
    return row


class _TextPredictor:
    def predict_proba(self, texts):
        scores = [0.9 if "church" in str(text).lower() else 0.1 for text in texts]
        return [[1.0 - score, score] for score in scores]
