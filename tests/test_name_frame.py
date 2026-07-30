"""Tests for the names-arm frame builder."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from binary_classifier.names.frame import build_name_frame


def _load_name_frame_cli():
    path = Path(__file__).resolve().parents[1] / "scripts/names/N1_build_name_frame.py"
    spec = importlib.util.spec_from_file_location("name_frame_cli_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["name_frame_cli_for_test"] = module
    spec.loader.exec_module(module)
    return module


def test_build_name_frame_derives_disjoint_panel_and_bmf_only_frames(
    tiny_registry,
) -> None:
    """The public stage seam writes the two distinct name populations."""
    tiny_registry.panel_final_parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "Mission Charity Inc",
                "BEST_NAME_CASED": "Mission Charity",
            },
            {
                "EIN2": "P002",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "Name Only Charity Inc",
                "BEST_NAME_CASED": "Name Only Charity",
            },
            {
                "EIN2": "P003",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501CX NONPROFIT",
                "F9_00_ORG_NAME_L1": "Out of Scope",
                "BEST_NAME_CASED": "Out of Scope",
            },
        ],
    ).to_parquet(tiny_registry.panel_final_parquet, index=False)
    pd.DataFrame(
        [
            {"EIN2": "P001", "BEST_NAME_BARE_CASED": "Mission Charity"},
            {"EIN2": "P002", "BEST_NAME_BARE_CASED": "Name Only Charity"},
            {"EIN2": "P003", "BEST_NAME_BARE_CASED": "Out of Scope"},
        ],
    ).to_parquet(tiny_registry.panel_filled_gaps_parquet, index=False)
    pd.DataFrame(
        [{"EIN2": "P001", "LONGEST_MISSION": "Serves families"}],
    ).to_parquet(tiny_registry.missions_parquet, index=False)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "ORG_NAME_CURRENT": "MISSION CHARITY",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": "0",
            },
            {
                "EIN2": "P002",
                "ORG_NAME_CURRENT": "NAME ONLY CHARITY",
                "NTEE_IRS": "X20",
                "BMF_FOUNDATION_CODE": "10",
            },
            {
                "EIN2": "P003",
                "ORG_NAME_CURRENT": "OUT OF SCOPE",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": "0",
            },
            {
                "EIN2": "B001",
                "ORG_NAME_CURRENT": "BMF ONLY CHURCH",
                "NTEE_IRS": "X99",
                "BMF_FOUNDATION_CODE": "0",
            },
        ],
    ).to_parquet(tiny_registry.bmf_parquet, index=False)

    build_name_frame(tiny_registry.cfg, tiny_registry)

    panel = pd.read_parquet(tiny_registry.names_panel_frame)
    bmf_only = pd.read_parquet(tiny_registry.names_bmf_only_frame)

    assert panel["EIN2"].tolist() == ["P001", "P002"]
    assert panel["has_mission"].tolist() == [True, False]
    assert panel["is_name_only"].tolist() == [False, True]
    assert panel["is_bmf_only"].tolist() == [False, False]
    assert panel["name_bare"].tolist() == ["Mission Charity", "Name Only Charity"]
    assert panel["name_raw"].tolist() == ["Mission Charity Inc", "Name Only Charity Inc"]
    assert panel["name_cased"].tolist() == ["Mission Charity", "Name Only Charity"]
    assert panel["is_external_religious_flag"].tolist() == [False, True]
    assert bmf_only["EIN2"].tolist() == ["B001"]
    assert bmf_only["is_bmf_only"].tolist() == [True]
    assert bmf_only["is_name_only"].tolist() == [False]
    assert bmf_only["is_ntee_x"].tolist() == [True]


def test_build_name_frame_flags_manifest_membership_and_excludes_missing_names(
    tiny_registry,
    caplog,
) -> None:
    """Missing names are reported while manifest membership remains auditable."""
    tiny_registry.panel_final_parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "TAX_YEAR": 2022,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "Known Charity",
                "BEST_NAME_CASED": "Known Charity",
            },
            {
                "EIN2": "P001",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "Known Charity",
                "BEST_NAME_CASED": "Known Charity",
            },
            {
                "EIN2": "P002",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": " ",
                "BEST_NAME_CASED": "Recovered Charity",
            },
            {
                "EIN2": "P003",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": " ",
                "BEST_NAME_CASED": " ",
            },
        ],
    ).to_parquet(tiny_registry.panel_final_parquet, index=False)
    pd.DataFrame(
        [
            {"EIN2": "P001", "BEST_NAME_BARE_CASED": "Known Charity"},
            {"EIN2": "P002", "BEST_NAME_BARE_CASED": "Recovered Charity"},
            {"EIN2": "P003", "BEST_NAME_BARE_CASED": None},
        ],
    ).to_parquet(tiny_registry.panel_filled_gaps_parquet, index=False)
    pd.DataFrame(columns=["EIN2", "LONGEST_MISSION"]).to_parquet(
        tiny_registry.missions_parquet,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "ORG_NAME_CURRENT": "KNOWN CHARITY",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": 0,
            },
            {
                "EIN2": "P002",
                "ORG_NAME_CURRENT": "",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": 0,
            },
            {
                "EIN2": "P003",
                "ORG_NAME_CURRENT": "",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": 0,
            },
            {
                "EIN2": "B001",
                "ORG_NAME_CURRENT": "BMF ONLY",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": 0,
            },
        ],
    ).to_parquet(tiny_registry.bmf_parquet, index=False)
    tiny_registry.silver_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"EIN2": ["P001"]}).to_csv(tiny_registry.silver_manifest, index=False)
    pd.DataFrame({"EIN2": ["B001"]}).to_csv(tiny_registry.anchor_manifest, index=False)

    with caplog.at_level("INFO"):
        build_name_frame(tiny_registry.cfg, tiny_registry)

    panel = pd.read_parquet(tiny_registry.names_panel_frame)
    bmf_only = pd.read_parquet(tiny_registry.names_bmf_only_frame)
    assert panel["EIN2"].tolist() == ["P001", "P002"]
    assert panel["name_raw"].tolist() == ["Known Charity", "Recovered Charity"]
    assert panel["is_manifest_contaminated"].tolist() == [True, False]
    assert bmf_only["EIN2"].tolist() == ["B001"]
    assert bmf_only["is_manifest_contaminated"].tolist() == [True]
    assert any("Excluded 1 panel_501c3 rows" in record.message for record in caplog.records)


def test_build_name_frame_rejects_duplicate_bmf_ein2(tiny_registry) -> None:
    """The stage fails rather than silently deduplicating BMF organizations."""
    tiny_registry.panel_final_parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        columns=["EIN2", "COMMON_LEVEL1", "BEST_NAME_CASED"],
    ).to_parquet(tiny_registry.panel_final_parquet, index=False)
    pd.DataFrame(
        columns=["EIN2", "BEST_NAME_BARE_CASED"],
    ).to_parquet(tiny_registry.panel_filled_gaps_parquet, index=False)
    pd.DataFrame(columns=["EIN2", "LONGEST_MISSION"]).to_parquet(
        tiny_registry.missions_parquet,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "EIN2": "B001",
                "ORG_NAME_CURRENT": "BMF ONE",
                "NTEE_IRS": "X20",
                "BMF_FOUNDATION_CODE": 10,
            },
            {
                "EIN2": "B001",
                "ORG_NAME_CURRENT": "BMF ONE",
                "NTEE_IRS": "X20",
                "BMF_FOUNDATION_CODE": 10,
            },
        ],
    ).to_parquet(tiny_registry.bmf_parquet, index=False)

    with pytest.raises(ValueError, match="BMF must contain one row per EIN2"):
        build_name_frame(tiny_registry.cfg, tiny_registry)


def test_name_frame_cli_calls_stage_with_loaded_config(monkeypatch) -> None:
    """The CLI stays a thin wrapper around the public frame-builder seam."""
    module = _load_name_frame_cli()
    cfg = object()
    registry = object()
    calls = []

    monkeypatch.setattr(sys, "argv", ["N1_build_name_frame.py", "--config", "names.yaml"])
    monkeypatch.setattr(module, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(module, "load_config", lambda path: cfg)
    monkeypatch.setattr(module, "PathRegistry", lambda path: registry)
    monkeypatch.setattr(
        module,
        "build_name_frame",
        lambda cfg_arg, registry_arg: calls.append((cfg_arg, registry_arg)),
    )

    assert module.main() == 0
    assert calls == [(cfg, registry)]
