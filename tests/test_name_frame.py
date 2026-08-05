"""Tests for the names-arm frame builder."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from pydantic import ValidationError

from binary_classifier.config import NamesConfig, NamesExpectedCounts, load_config
from binary_classifier.names.frame import (
    _collapse_panel,
    _read_collapsed_panel,
    build_name_frame,
)


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
    tiny_registry.cfg.names.panel_scope_values = [
        " 501C3 CHARITY ",
        "501CX NONPROFIT",
    ]
    tiny_registry.cfg.names.expected_counts = NamesExpectedCounts(
        panel_has_mission=1,
        panel_name_only=2,
        panel_no_name_no_mission=0,
        panel_name_only_flagged=1,
        bmf_only=1,
        bmf_only_flagged=1,
    )
    tiny_registry.panel_final_parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "Mission Charity Inc",
                "BEST_NAME_CASED": "Mission Charity",
                "BEST_DBA_CASED": "Grace Church",
                "HAS_DBA": True,
            },
            {
                "EIN2": "P002",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "Name Only Charity Inc",
                "BEST_NAME_CASED": "Name Only Charity",
                "BEST_DBA_CASED": None,
                "HAS_DBA": False,
            },
            {
                "EIN2": "P003",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501CX NONPROFIT",
                "F9_00_ORG_NAME_L1": "Out of Scope",
                "BEST_NAME_CASED": "Out of Scope",
                "BEST_DBA_CASED": "Out Of Scope DBA",
                "HAS_DBA": True,
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
                "NTEE_IRS": "1A",
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

    assert panel["EIN2"].tolist() == ["P001", "P002", "P003"]
    assert panel["has_mission"].tolist() == [True, False, False]
    assert panel["is_name_only"].tolist() == [False, True, True]
    assert panel["is_bmf_only"].tolist() == [False, False, False]
    assert panel["panel_scope"].tolist() == [
        "501C3 CHARITY",
        "501C3 CHARITY",
        "501CX NONPROFIT",
    ]
    assert panel["name_bare"].tolist() == [
        "Mission Charity",
        "Name Only Charity",
        "Out of Scope",
    ]
    assert panel["name_raw"].tolist() == [
        "Mission Charity Inc",
        "Name Only Charity Inc",
        "Out of Scope",
    ]
    assert panel["name_cased"].tolist() == [
        "Mission Charity",
        "Name Only Charity",
        "Out of Scope",
    ]
    assert panel["dba_cased"].iloc[0] == "Grace Church"
    assert pd.isna(panel["dba_cased"].iloc[1])
    assert panel["has_dba"].tolist() == [True, False, True]
    assert panel["is_external_religious_flag"].tolist() == [False, True, False]
    assert panel["is_external_religious_flag"].notna().all()
    assert panel["ntee_major_group"].tolist() == ["?", "X", "P"]
    assert bmf_only["EIN2"].tolist() == ["B001"]
    assert bmf_only["is_bmf_only"].tolist() == [True]
    assert bmf_only["is_name_only"].tolist() == [False]
    assert bmf_only["is_ntee_x"].tolist() == [True]
    assert bmf_only["is_external_religious_flag"].notna().all()
    assert bmf_only["name_raw_source"].tolist() == ["ORG_NAME_CURRENT"]
    assert bmf_only["panel_scope"].isna().all()


@pytest.mark.parametrize(
    "values, message",
    [
        (["A", " A "], "unique"),
        (["A", "   "], "non-empty"),
        ([], "at least 1|non-empty"),
    ],
)
def test_names_config_validates_scope_values(values, message) -> None:
    with pytest.raises(ValidationError, match=message):
        NamesConfig(panel_scope_values=values)


def test_names_config_trims_scope_values() -> None:
    config = NamesConfig(panel_scope_values=[" A ", "B"])

    assert config.panel_scope_values == ["A", "B"]


def test_collapse_panel_uses_longest_raw_name_with_tax_year_tie_break() -> None:
    """Longitudinal raw names use the upstream best-name selection rule."""
    panel = pd.DataFrame(
        [
            {
                "EIN2": "EIN-01-0017496",
                "TAX_YEAR": 2022,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "BEST_NAME_CASED": "Agamenticus Yacht Club",
                "F9_00_ORG_NAME_L1": "AGAMENTICUS YACHT CLUB OF MAINE",
            },
            {
                "EIN2": "EIN-01-0017496",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "BEST_NAME_CASED": "Agamenticus Yacht Club",
                "F9_00_ORG_NAME_L1": "AGAMENTICUS YACHT CLUB INC",
            },
            {
                "EIN2": "EIN-01-0017496",
                "TAX_YEAR": 2024,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "BEST_NAME_CASED": "Agamenticus Yacht Club",
                "F9_00_ORG_NAME_L1": "AGAMENTICUS YACHT CLUB OF MAINE",
            },
        ]
    )

    collapsed = _collapse_panel(panel, selection_column="F9_00_ORG_NAME_L1")

    assert collapsed.to_dict("records") == [
        {
            "EIN2": "EIN-01-0017496",
            "COMMON_LEVEL1": "501C3 CHARITY",
            "BEST_NAME_CASED": "Agamenticus Yacht Club",
            "F9_00_ORG_NAME_L1": "AGAMENTICUS YACHT CLUB OF MAINE",
        }
    ]


def test_collapse_panel_uses_alphabetic_order_after_equal_length_and_tax_year() -> None:
    """Equal length and year candidates use alphabetic order deterministically."""
    panel = pd.DataFrame(
        [
            {"EIN2": "P001", "TAX_YEAR": 2023, "F9_00_ORG_NAME_L1": "ZEBRA"},
            {"EIN2": "P001", "TAX_YEAR": 2023, "F9_00_ORG_NAME_L1": "ALPHA"},
        ]
    )

    collapsed = _collapse_panel(panel, selection_column="F9_00_ORG_NAME_L1")

    assert collapsed["F9_00_ORG_NAME_L1"].tolist() == ["ALPHA"]


def test_read_collapsed_panel_requires_configured_tax_year(tmp_path) -> None:
    """Raw-name selection cannot silently skip its required year tie-break."""
    path = tmp_path / "panel.parquet"
    pd.DataFrame(
        [{"EIN2": "P001", "F9_00_ORG_NAME_L1": "Example Name"}]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="missing required columns: TAX_YEAR"):
        _read_collapsed_panel(
            path,
            ["EIN2", "F9_00_ORG_NAME_L1"],
            selection_column="F9_00_ORG_NAME_L1",
        )


def test_read_collapsed_panel_selects_names_from_unordered_input(tmp_path) -> None:
    """Batch reduction does not require the upstream parquet to sort EIN2 rows."""
    path = tmp_path / "panel.parquet"
    pd.DataFrame(
        [
            {"EIN2": "P002", "TAX_YEAR": 2022, "F9_00_ORG_NAME_L1": "Short"},
            {
                "EIN2": "P001",
                "TAX_YEAR": 2023,
                "F9_00_ORG_NAME_L1": "Only Name",
            },
            {
                "EIN2": "P002",
                "TAX_YEAR": 2023,
                "F9_00_ORG_NAME_L1": "Longer Name",
            },
        ]
    ).to_parquet(path, index=False)

    collapsed = _read_collapsed_panel(
        path,
        ["EIN2", "F9_00_ORG_NAME_L1"],
        selection_column="F9_00_ORG_NAME_L1",
    )

    assert collapsed.to_dict("records") == [
        {"EIN2": "P001", "F9_00_ORG_NAME_L1": "Only Name"},
        {"EIN2": "P002", "F9_00_ORG_NAME_L1": "Longer Name"},
    ]


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
    assert panel["EIN2"].tolist() == ["P001"]
    assert panel["name_raw"].tolist() == ["Known Charity"]
    assert panel["name_raw_source"].tolist() == ["F9_00_ORG_NAME_L1"]
    assert panel["is_manifest_contaminated"].tolist() == [True]
    assert bmf_only["EIN2"].tolist() == ["B001"]
    assert bmf_only["is_manifest_contaminated"].tolist() == [True]
    assert any("Excluded 2 panel_scoped rows" in record.message for record in caplog.records)
    assert any(
        "panel_has_mission=0 panel_name_only=1 panel_no_name_no_mission=2 "
        "panel_name_only_flagged=0 bmf_only=1 bmf_only_flagged=0." in record.message
        for record in caplog.records
    )

    tiny_registry.cfg.names.expected_counts = NamesExpectedCounts(
        panel_has_mission=0,
        panel_name_only=0,
        panel_no_name_no_mission=2,
        panel_name_only_flagged=0,
        bmf_only=1,
        bmf_only_flagged=0,
    )
    with pytest.raises(
        ValueError,
        match="panel_name_only: expected 0, observed 1",
    ):
        build_name_frame(tiny_registry.cfg, tiny_registry)


def test_build_name_frame_retains_panel_ein2_missing_from_bmf(tiny_registry) -> None:
    """BMF enrichment is optional for scoped panel organizations."""
    tiny_registry.panel_final_parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "F9_00_ORG_NAME_L1": "Missing BMF Charity",
                "BEST_NAME_CASED": "Missing BMF Charity",
            },
        ],
    ).to_parquet(tiny_registry.panel_final_parquet, index=False)
    pd.DataFrame(
        [{"EIN2": "P001", "BEST_NAME_BARE_CASED": "Missing BMF Charity"}],
    ).to_parquet(tiny_registry.panel_filled_gaps_parquet, index=False)
    pd.DataFrame(columns=["EIN2", "LONGEST_MISSION"]).to_parquet(
        tiny_registry.missions_parquet,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "EIN2": "B001",
                "ORG_NAME_CURRENT": "BMF ONLY",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": 0,
            },
        ],
    ).to_parquet(tiny_registry.bmf_parquet, index=False)

    build_name_frame(tiny_registry.cfg, tiny_registry)

    panel = pd.read_parquet(tiny_registry.names_panel_frame)
    assert panel["EIN2"].tolist() == ["P001"]
    assert panel["has_bmf"].tolist() == [False]
    assert panel[
        ["NTEE_IRS", "BMF_FOUNDATION_CODE", "is_external_religious_flag"]
    ].isna().all().all()


def test_build_name_frame_rejects_when_scope_matches_no_panel_ein2(
    tiny_registry,
) -> None:
    """BMF coverage is only required for the scoped panel frame using its flags."""
    tiny_registry.panel_final_parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501CX NONPROFIT",
                "F9_00_ORG_NAME_L1": "Out Of Scope Organization",
                "BEST_NAME_CASED": "Out Of Scope Organization",
            },
        ]
    ).to_parquet(tiny_registry.panel_final_parquet, index=False)
    pd.DataFrame(
        [{"EIN2": "P001", "BEST_NAME_BARE_CASED": "Out Of Scope Organization"}],
    ).to_parquet(tiny_registry.panel_filled_gaps_parquet, index=False)
    pd.DataFrame(columns=["EIN2", "LONGEST_MISSION"]).to_parquet(
        tiny_registry.missions_parquet,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "EIN2": "B001",
                "ORG_NAME_CURRENT": "BMF ONLY",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": 0,
            },
        ]
    ).to_parquet(tiny_registry.bmf_parquet, index=False)

    with pytest.raises(ValueError, match="panel scope values .* matched zero panel EIN2s"):
        build_name_frame(tiny_registry.cfg, tiny_registry)

    assert not tiny_registry.names_panel_frame.exists()
    assert not tiny_registry.names_bmf_only_frame.exists()


def test_build_name_frame_rejects_duplicate_bmf_ein2(tiny_registry) -> None:
    """The stage fails rather than silently deduplicating BMF organizations."""
    tiny_registry.panel_final_parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        columns=[
            "EIN2",
            "TAX_YEAR",
            "COMMON_LEVEL1",
            "F9_00_ORG_NAME_L1",
            "BEST_NAME_CASED",
        ],
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


def test_build_name_frame_uses_name_cased_when_raw_f9_field_is_absent(
    tiny_registry,
) -> None:
    """N1 uses the configured fallback when the preferred raw field is absent."""
    tiny_registry.cfg.names.panel_raw_name_columns = [
        "F9_00_ORG_NAME_L1",
        "NAME_CASED",
    ]
    tiny_registry.panel_final_parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "TAX_YEAR": 2023,
                "COMMON_LEVEL1": "501C3 CHARITY",
                "NAME_CASED": "Raw Alternate Name",
                "BEST_NAME_CASED": "Canonical Name",
            },
        ],
    ).to_parquet(tiny_registry.panel_final_parquet, index=False)
    pd.DataFrame(
        [{"EIN2": "P001", "BEST_NAME_BARE_CASED": "Bare Name"}],
    ).to_parquet(tiny_registry.panel_filled_gaps_parquet, index=False)
    pd.DataFrame(columns=["EIN2", "LONGEST_MISSION"]).to_parquet(
        tiny_registry.missions_parquet,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "ORG_NAME_CURRENT": "RAW ALTERNATE NAME",
                "NTEE_IRS": "P20",
                "BMF_FOUNDATION_CODE": 0,
            },
        ],
    ).to_parquet(tiny_registry.bmf_parquet, index=False)

    build_name_frame(tiny_registry.cfg, tiny_registry)

    panel = pd.read_parquet(tiny_registry.names_panel_frame)
    assert panel["name_raw"].tolist() == ["Raw Alternate Name"]
    assert panel["name_raw_source"].tolist() == ["NAME_CASED"]
    assert panel["name_cased"].tolist() == ["Canonical Name"]


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


def test_production_config_has_names_snapshot_counts() -> None:
    """The production names arm pins the snapshot it is designed to reconcile."""
    config_path = Path(__file__).resolve().parents[1] / "config/religious_missions.yaml"

    cfg = load_config(config_path)

    assert cfg.names.expected_counts == NamesExpectedCounts(
        panel_has_mission=560_354,
        panel_name_only=257_599,
        panel_no_name_no_mission=14_731,
        panel_name_only_flagged=26_757,
        bmf_only=2_004_353,
        bmf_only_flagged=396_379,
    )
    assert cfg.names.panel_scope_values == ["501C3 CHARITY"]
    assert cfg.names.panel_raw_name_columns == ["F9_00_ORG_NAME_L1", "NAME_CASED"]
