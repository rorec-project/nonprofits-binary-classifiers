"""Offline chain coverage for the isolated names transfer arm."""

import json
import runpy
from pathlib import Path

import pandas as pd

from binary_classifier.config import NamesExpectedCounts
from binary_classifier.names import (
    build_name_frame,
    draw_name_gold,
    run_name_validation,
    score_names,
)
from binary_classifier.names.cleaner import clean_names


def test_names_chain_writes_isolated_artifacts_and_preserves_missions(
    monkeypatch,
    tiny_config,
    tiny_registry,
) -> None:
    """Run N1, N2, N3, N6, and N4 without touching missions outputs."""
    _configure_tiny_name_gold(tiny_config)
    _write_name_inputs(tiny_registry)
    _write_mission_scoring_inputs(tiny_registry)
    before = _mission_artifact_snapshot(tiny_registry)

    frozen_test = tiny_registry.test_evaluation.resolve()
    original_open = Path.open

    def guard_frozen_test_open(path: Path, *args, **kwargs):
        if path.resolve() == frozen_test:
            raise AssertionError("names chain accessed the frozen test artifact")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guard_frozen_test_open)
        build_name_frame(tiny_config, tiny_registry)
        clean_names(tiny_config, tiny_registry)
        score_names(tiny_config, tiny_registry, predictor=_TextPredictor())
        draw_name_gold(tiny_config, tiny_registry)

        name_gold = pd.read_csv(tiny_registry.names_gold_coding_template)
        name_gold["human_label"] = (
            name_gold["text"]
            .str.contains(
                "Church|Baptist",
                regex=True,
            )
            .astype(int)
        )
        name_gold.to_csv(tiny_registry.names_gold_coding_template, index=False)
        run_name_validation(tiny_config, tiny_registry)

    _assert_name_artifacts(tiny_registry)
    assert _mission_artifact_snapshot(tiny_registry) == before

    orchestrator = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"),
    )
    assert all(
        not module_name.startswith("binary_classifier.names")
        for module_name, _ in orchestrator["_STAGE_MODULES"].values()
    )


def _configure_tiny_name_gold(cfg) -> None:
    cfg.names.expected_counts = NamesExpectedCounts(
        panel_has_mission=2,
        panel_name_only=1,
        panel_no_name_no_mission=0,
        panel_name_only_flagged=0,
        bmf_only=4,
        bmf_only_flagged=3,
    )
    cfg.names.gold_sample_size = 4
    cfg.names.gold_stratum_quotas = {
        "ntee_x_only": 1,
        "church_foundation_only": 1,
        "both_external_flags": 1,
        "neither_external_flag": 1,
    }
    cfg.names.gold_conflict_quotas = {
        "saint_name": 0,
        "faith_heritage": 0,
        "non_christian_tradition": 0,
        "non_english_name": 0,
    }
    cfg.evaluation.bootstrap_resamples = 10


def _write_name_inputs(registry) -> None:
    registry.panel_final_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "FIRST BAPTIST CHURCH, INC.",
                "BEST_NAME_CASED": "First Baptist Church",
            },
            {
                "EIN2": "P002",
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "COMMUNITY FOOD BANK LLC",
                "BEST_NAME_CASED": "Community Food Bank",
            },
            {
                "EIN2": "P003",
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "NAME ONLY OUTREACH INC.",
                "BEST_NAME_CASED": "Name Only Outreach",
            },
        ]
    ).to_parquet(registry.panel_final_parquet, index=False)
    pd.DataFrame(
        {
            "EIN2": ["P001", "P002", "P003"],
            "BEST_NAME_BARE_CASED": [
                "First Baptist Church",
                "Community Food Bank",
                "Name Only Outreach",
            ],
        }
    ).to_parquet(registry.panel_filled_gaps_parquet, index=False)
    pd.DataFrame(
        {
            "EIN2": ["P001", "P002"],
            "LONGEST_MISSION": ["Church services", "Food assistance"],
        }
    ).to_parquet(registry.missions_parquet, index=False)
    pd.DataFrame(
        [
            _bmf_row("P001", "FIRST BAPTIST CHURCH", "P20", 0),
            _bmf_row("P002", "COMMUNITY FOOD BANK", "P20", 0),
            _bmf_row("P003", "NAME ONLY OUTREACH", "P20", 0),
            _bmf_row("B001", "FAITH CHURCH", "X20", 0),
            _bmf_row("B002", "COMMUNITY FOUNDATION", "A20", 10),
            _bmf_row("B003", "BAPTIST COMMUNITY CENTER", "X20", 10),
            _bmf_row("B004", "SAINT MARY HOSPITAL", "P20", 0),
        ]
    ).to_parquet(registry.bmf_parquet, index=False)


def _bmf_row(
    ein2: str, name: str, ntee: str, foundation_code: int
) -> dict[str, object]:
    return {
        "EIN2": ein2,
        "ORG_NAME_CURRENT": name,
        "NTEE_IRS": ntee,
        "BMF_FOUNDATION_CODE": foundation_code,
    }


def _write_mission_scoring_inputs(registry) -> None:
    registry.selected_model.write_text(
        json.dumps(
            {
                "encoder_id": "stub-model",
                "checkpoint_sha256": "stub-sha",
                "checkpoint_relpath": "unused/model.safetensors",
            }
        )
    )
    registry.calibrator_path.write_text(
        json.dumps({"threshold": 0.4, "max_f1_threshold": 0.6})
    )
    registry.base_rate_precision.write_text(json.dumps({"threshold": 0.8}))
    registry.predictions_full_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "EIN2": ["P001", "P002"],
            "pred_label": [1, 0],
            "prob_calibrated": [0.9, 0.1],
        }
    ).to_parquet(registry.predictions_full_parquet, index=False)
    registry.test_evaluation.write_text('{"frozen": true}\n')


def _mission_artifact_snapshot(registry) -> dict[Path, bytes]:
    roots = (registry.interim_dir, registry.processed_dir)
    return {
        path.relative_to(root): path.read_bytes()
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "names" not in path.relative_to(root).parts
    }


def _assert_name_artifacts(registry) -> None:
    panel_frame = pd.read_parquet(registry.names_panel_frame)
    bmf_only_frame = pd.read_parquet(registry.names_bmf_only_frame)
    panel_cleaned = pd.read_parquet(registry.names_panel_cleaned)
    bmf_only_cleaned = pd.read_parquet(registry.names_bmf_only_cleaned)
    scores = pd.read_parquet(registry.names_scores)
    name_gold = pd.read_csv(registry.names_gold_coding_template)
    audit = json.loads(registry.names_divergence_audit.read_text())
    validation = json.loads(registry.names_validation.read_text())

    assert set(panel_frame["EIN2"]) == {"P001", "P002", "P003"}
    assert set(panel_cleaned["EIN2"]) == set(panel_frame["EIN2"])
    assert set(bmf_only_frame["EIN2"]) == {"B001", "B002", "B003", "B004"}
    assert set(bmf_only_cleaned["EIN2"]) == set(bmf_only_frame["EIN2"])
    assert len(scores) == 14
    assert set(scores["EIN2"]) == set(panel_frame["EIN2"]) | set(bmf_only_frame["EIN2"])
    assert set(scores["input_variant"]) == {"suffix_stripped", "suffix_retaining"}
    assert set(name_gold["EIN2"]) == set(bmf_only_frame["EIN2"])
    assert name_gold["human_label"].isin([0, 1]).all()
    assert audit["panel_rows_audited"] == len(panel_frame)
    assert validation["comparison_population"]["n_evaluated"] == 2
    assert registry.names_gold_manifest.exists()
    assert registry.names_gold_coding_instructions.exists()


class _TextPredictor:
    def predict_proba(self, texts):
        scores = [
            0.9 if "Church" in str(text) or "Baptist" in str(text) else 0.1
            for text in texts
        ]
        return [[1.0 - score, score] for score in scores]
