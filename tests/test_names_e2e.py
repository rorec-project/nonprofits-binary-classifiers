"""Offline chain coverage for the isolated names transfer arm."""

import builtins
import io
import json
import os
import runpy
from pathlib import Path

import pandas as pd

from binary_classifier.config import NamesExpectedCounts
from binary_classifier.names import (
    build_name_frame,
    draw_name_gold,
    run_name_diagnostics,
    run_name_validation,
    score_names,
)
from binary_classifier.names.cleaner import clean_names
from binary_classifier.names.probes import PROBES, PROBE_SET_VERSION


def test_names_chain_writes_isolated_artifacts_and_preserves_missions(
    monkeypatch,
    tiny_config,
    tiny_registry,
) -> None:
    """Run N1-N6 names stages without touching missions outputs."""
    # ── Synthetic isolated fixture ──────────────────────────────────────────
    # Seed only the names inputs and the read-only mission artifacts N3/N4 need.
    _configure_tiny_name_gold(tiny_config)
    _write_name_inputs(tiny_registry)
    _write_mission_scoring_inputs(tiny_registry)
    mission_artifacts_before = _mission_artifact_snapshot(tiny_registry)

    # ── Names transfer chain ────────────────────────────────────────────────
    # Any Python-level frozen-test open is a hard failure, not merely unchanged
    # output, because the report is governed by one-shot evaluation semantics.
    frozen_test = tiny_registry.test_evaluation.resolve()
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_open = Path.open

    def guard_frozen_test_path(path: object) -> None:
        try:
            path_value = os.fspath(path)
        except TypeError:
            return
        if isinstance(path_value, bytes):
            path_value = os.fsdecode(path_value)
        if Path(path_value).resolve() == frozen_test:
            raise AssertionError("names chain accessed the frozen test artifact")

    def guard_builtin_open(path, *args, **kwargs):
        guard_frozen_test_path(path)
        return original_builtin_open(path, *args, **kwargs)

    def guard_io_open(path, *args, **kwargs):
        guard_frozen_test_path(path)
        return original_io_open(path, *args, **kwargs)

    def guard_os_open(path, *args, **kwargs):
        guard_frozen_test_path(path)
        return original_os_open(path, *args, **kwargs)

    def guard_frozen_test_open(path: Path, *args, **kwargs):
        guard_frozen_test_path(path)
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", guard_builtin_open)
        guard.setattr(io, "open", guard_io_open)
        guard.setattr(os, "open", guard_os_open)
        guard.setattr(Path, "open", guard_frozen_test_open)
        build_name_frame(tiny_config, tiny_registry)
        clean_names(tiny_config, tiny_registry)
        score_names(tiny_config, tiny_registry, predictor=_TextPredictor())
        run_name_diagnostics(
            tiny_config,
            tiny_registry,
            predictor=_TextPredictor(),
        )
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
    assert _mission_artifact_snapshot(tiny_registry) == mission_artifacts_before

    # ── Orchestrator boundary ───────────────────────────────────────────────
    # Names stay independently runnable and must not enter the missions chain.
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
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "HERITAGE ACADEMY, INC.",
                "BEST_NAME_CASED": "Heritage Academy",
                "BEST_DBA_CASED": "Grace Bible Church",
                "HAS_DBA": True,
            },
            {
                "EIN2": "P002",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "COMMUNITY CHURCH FOOD BANK LLC",
                "BEST_NAME_CASED": "Community Church Food Bank",
                "BEST_DBA_CASED": "Community Food Bank",
                "HAS_DBA": True,
            },
            {
                "EIN2": "P003",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "NAME ONLY OUTREACH INC.",
                "BEST_NAME_CASED": "Name Only Outreach",
                "BEST_DBA_CASED": None,
                "HAS_DBA": False,
            },
        ]
    ).to_parquet(registry.panel_final_parquet, index=False)
    pd.DataFrame(
        {
            "EIN2": ["P001", "P002", "P003"],
            "BEST_NAME_BARE_CASED": [
                "Heritage Academy",
                "Community Church Food Bank",
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
    gold_manifest = pd.read_csv(registry.names_gold_manifest)
    name_gold = pd.read_csv(registry.names_gold_coding_template)
    audit = json.loads(registry.names_divergence_audit.read_text())
    probes = json.loads(registry.names_probe_diagnostics.read_text())
    dba_cases = pd.read_parquet(registry.names_dba_case_study)
    dba_report = json.loads(registry.names_dba_case_study_report.read_text())
    validation = json.loads(registry.names_validation.read_text())

    assert set(panel_frame["EIN2"]) == {"P001", "P002", "P003"}
    assert panel_frame["EIN2"].is_unique
    assert panel_frame["population"].eq("panel_scoped").all()
    assert panel_frame["panel_scope"].notna().all()
    assert set(panel_cleaned["EIN2"]) == set(panel_frame["EIN2"])
    assert panel_cleaned["EIN2"].is_unique
    assert {"EIN2", "name_raw", "name_cleaned"}.issubset(panel_cleaned.columns)
    assert set(bmf_only_frame["EIN2"]) == {"B001", "B002", "B003", "B004"}
    assert bmf_only_frame["EIN2"].is_unique
    assert bmf_only_frame["population"].eq("bmf_only").all()
    assert bmf_only_frame["panel_scope"].isna().all()
    assert set(bmf_only_cleaned["EIN2"]) == set(bmf_only_frame["EIN2"])
    assert bmf_only_cleaned["EIN2"].is_unique
    assert {"EIN2", "name_raw", "name_cleaned"}.issubset(bmf_only_cleaned.columns)
    assert len(scores) == 14
    assert set(scores["EIN2"]) == set(panel_frame["EIN2"]) | set(bmf_only_frame["EIN2"])
    assert set(scores["input_variant"]) == {"suffix_stripped", "suffix_retaining"}
    assert not scores.duplicated(["EIN2", "input_variant"]).any()
    assert set(gold_manifest["EIN2"]) == set(bmf_only_frame["EIN2"])
    assert gold_manifest["EIN2"].is_unique
    assert {
        "EIN2",
        "population",
        "name_raw",
        "inclusion_probability",
        "sampling_cell",
    }.issubset(gold_manifest.columns)
    assert set(name_gold["EIN2"]) == set(bmf_only_frame["EIN2"])
    assert name_gold["human_label"].isin([0, 1]).all()
    assert audit["panel_rows_audited"] == len(panel_frame)
    assert {"blocking_divergences", "nonblocking_divergences"}.issubset(audit)
    assert probes["diagnostic_only"] is True
    assert probes["interpretation"] == "diagnosis_not_accuracy"
    assert probes["probe_set_version"] == PROBE_SET_VERSION
    assert len(probes["records"]) == len(PROBES)
    required_probe_fields = {
        "probe_id",
        "category",
        "pair_id",
        "text_raw",
        "text_cleaned",
        "prob_raw",
        "lexicon_rule_label",
        "model_id",
        "checkpoint_sha256",
        "inference_date",
        "config_hash",
    }
    assert all(required_probe_fields.issubset(record) for record in probes["records"])
    assert {record["probe_id"] for record in probes["records"]} == {
        probe[0] for probe in PROBES
    }
    assert set(dba_cases["EIN2"]) == {"P001", "P002"}
    assert {
        "EIN2",
        "population",
        "legal_name_cleaned",
        "dba_name_cleaned",
        "token_direction",
        "legal_name_prob_raw",
        "dba_name_prob_raw",
        "diagnostic_only",
        "production_input_variant",
    }.issubset(dba_cases.columns)
    assert set(dba_cases["token_direction"]) == {
        "dba_adds_religious_token",
        "legal_name_adds_religious_token",
    }
    assert dba_report["diagnostic_only"] is True
    assert dba_report["production_input_variant"] is False
    assert dba_report["dba_having_organizations"] == 2
    assert dba_report["dba_adds_religious_token_count"] == 1
    assert dba_report["legal_name_adds_religious_token_count"] == 1
    assert validation["comparison_population"]["n_evaluated"] == 2
    assert {"variants", "bmf_only_gold", "external_flag_validation"}.issubset(
        validation
    )
    assert set(validation["variants"]) == {"suffix_stripped", "suffix_retaining"}
    assert validation["bmf_only_gold"]["n_gold"] == len(gold_manifest)
    assert registry.names_gold_manifest.exists()
    assert registry.names_gold_coding_instructions.exists()


class _TextPredictor:
    def predict_proba(self, texts):
        scores = [
            0.9 if "Church" in str(text) or "Baptist" in str(text) else 0.1
            for text in texts
        ]
        return [[1.0 - score, score] for score in scores]
