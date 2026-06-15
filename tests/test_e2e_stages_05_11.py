"""Slow synthetic E2E route for stages 05 onward.

Later PRs extend this file through stages 07–11.
"""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from binary_classifier.annotate.schema import AnnotationStore, BinaryLabel, LabelRecord, SourceType
from binary_classifier.data import anchor as anchor_mod
from binary_classifier.train import data as train_data_mod
from binary_classifier.train import sweep
from binary_classifier.train.trainer import run_training


@pytest.mark.slow
def test_e2e_stages_05_to_06_with_finetune_stub(
    monkeypatch,
    tiny_config,
    tiny_registry,
) -> None:
    """Fabricate stage-04 artifacts, run stage 05, then run stage 06 offline."""
    tiny_config.anchor.n = 8
    tiny_config.anchor.min_stratum_frame = 1
    tiny_config.training.baselines = ["tfidf_logreg"]
    tiny_config.training.encoders = [Encoder := tiny_config.training.encoders[0].model_copy(update={"id": "fake/encoder"})]
    tiny_config.training.sweep_seeds = [42]
    tiny_config.training.final_seeds = [42, 43]
    tiny_config.training.curve_fractions = [1.0]
    tiny_config.training.arms = []

    missions = _missions_frame()
    monkeypatch.setattr(anchor_mod, "load_missions", lambda cfg: missions)
    monkeypatch.setattr(train_data_mod, "load_missions", lambda cfg: missions)
    monkeypatch.setattr(sweep, "finetune", _fake_finetune)

    _write_stage04_training_artifacts(tiny_registry)
    _write_coded_gold(tiny_registry)
    _write_confirmed_slate(tiny_registry)
    _write_stage01_manifests(tiny_registry)

    anchor_mod.build_anchor(tiny_config, tiny_registry, force=True)
    anchor_template = pd.read_csv(tiny_registry.anchor_coding_template)
    anchor_template["human_label"] = [i % 2 for i in range(len(anchor_template))]
    anchor_template.to_csv(tiny_registry.anchor_coding_template, index=False)

    run_training(tiny_config, tiny_registry, final=True)

    assert tiny_registry.anchor_manifest.exists()
    assert tiny_registry.learning_curve_results.exists()
    assert tiny_registry.selection_report.exists()
    report = json.loads(tiny_registry.selection_report.read_text())
    assert report["recommendation"]["encoder_id"] == Encoder.id
    assert report["selected_model_skeleton"]["checkpoint_relpath"]


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
    AnnotationStore(registry.annotation_store).append_many(records)


def _write_coded_gold(registry) -> None:
    rows = []
    for i in range(8):
        rows.append(
            {
                "EIN2": f"G{i:04d}",
                "split": "validation",
                "text": f"validation text {i}",
                "human_label": i % 2,
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
    pd.DataFrame({"EIN2": [f"G{i:04d}" for i in range(8)]}).to_csv(
        registry.gold_manifest,
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
