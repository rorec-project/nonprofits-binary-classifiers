"""Tests for the names-arm BMF-only human-gold stage."""

import logging

import pandas as pd
import pytest

from binary_classifier.names.gold import draw_name_gold
from binary_classifier.names.validation import run_name_validation


def test_draw_name_gold_writes_reproducible_stratified_bmf_only_template(
    tiny_registry,
    caplog,
) -> None:
    """The public stage seam emits the configured BMF-only coding sample."""
    caplog.set_level(logging.INFO, logger="binary_classifier.names.gold")
    tiny_registry.cfg.names.gold_sample_size = 8
    tiny_registry.cfg.names.gold_stratum_quotas = {
        "ntee_x_only": 2,
        "church_foundation_only": 2,
        "both_external_flags": 2,
        "neither_external_flag": 2,
    }
    tiny_registry.cfg.names.gold_conflict_quotas = {
        "saint_name": 1,
        "faith_heritage": 1,
        "non_christian_tradition": 1,
        "non_english_name": 1,
    }
    pd.DataFrame(
        [
            {
                "EIN2": "001",
                "population": "bmf_only",
                "name_raw": "Saint Mary Hospital",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": True,
                "is_church_foundation": False,
            },
            {
                "EIN2": "X002",
                "population": "bmf_only",
                "name_raw": "Faith Heritage Fundacion",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": True,
                "is_church_foundation": False,
            },
            {
                "EIN2": "C001",
                "population": "bmf_only",
                "name_raw": "Masjid Community Center",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": False,
                "is_church_foundation": True,
            },
            {
                "EIN2": "C002",
                "population": "bmf_only",
                "name_raw": "Foundation Two",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": False,
                "is_church_foundation": True,
            },
            {
                "EIN2": "B001",
                "population": "bmf_only",
                "name_raw": "Societe Culturelle",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": True,
                "is_church_foundation": True,
            },
            {
                "EIN2": "B002",
                "population": "bmf_only",
                "name_raw": "Both Flags Two",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": True,
                "is_church_foundation": True,
            },
            {
                "EIN2": "N001",
                "population": "bmf_only",
                "name_raw": "Saint Thomas Community Hospital",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": False,
                "is_church_foundation": False,
            },
            {
                "EIN2": "N002",
                "population": "bmf_only",
                "name_raw": "Community Support",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": False,
                "is_church_foundation": False,
            },
            {
                "EIN2": "EXCLUDED",
                "population": "bmf_only",
                "name_raw": "Excluded Contaminated",
                "is_bmf_only": True,
                "is_manifest_contaminated": True,
                "is_ntee_x": False,
                "is_church_foundation": False,
            },
        ]
    ).to_parquet(tiny_registry.names_bmf_only_frame, index=False)

    draw_name_gold(tiny_registry.cfg, tiny_registry)

    manifest = pd.read_csv(tiny_registry.names_gold_manifest)
    template = pd.read_csv(tiny_registry.names_gold_coding_template)
    assert len(manifest) == 8
    assert set(manifest["EIN2"]) == set(template["EIN2"])
    assert set(manifest["population"]) == {"bmf_only"}
    assert not manifest["is_manifest_contaminated"].any()
    assert manifest["gold_stratum"].value_counts().to_dict() == {
        "ntee_x_only": 2,
        "church_foundation_only": 2,
        "both_external_flags": 2,
        "neither_external_flag": 2,
    }
    assert {
        "saint_name",
        "faith_heritage",
        "non_christian_tradition",
        "non_english_name",
    }.issubset(
        set("|".join(manifest["conflict_categories"].dropna()).split("|"))
    )
    saint_conflicts = manifest[
        manifest["conflict_categories"].str.contains("saint_name")
    ]
    assert not saint_conflicts["is_ntee_x"].any()
    assert not saint_conflicts["is_church_foundation"].any()
    assert all(
        realized >= tiny_registry.cfg.names.gold_conflict_quotas[category]
        for category, realized in {
        category: int(manifest["conflict_categories"].str.contains(category).sum())
        for category in tiny_registry.cfg.names.gold_conflict_quotas
        }.items()
    )
    assert template.columns.tolist() == [
        "EIN2",
        "split",
        "text",
        "human_label",
    ]
    assert template["human_label"].isna().all()
    assert "saint name alone is not religious" in (
        tiny_registry.names_gold_coding_instructions.read_text()
    )
    assert "conflicts=" in caplog.text
    assert "non_english_name" in caplog.text

    first_draw = manifest.copy()
    draw_name_gold(tiny_registry.cfg, tiny_registry)
    tiny_registry.names_gold_manifest.unlink()
    tiny_registry.names_gold_coding_template.unlink()
    draw_name_gold(tiny_registry.cfg, tiny_registry)
    pd.testing.assert_frame_equal(first_draw, pd.read_csv(tiny_registry.names_gold_manifest))


def test_draw_name_gold_rejects_missing_required_conflict_quota(tiny_registry) -> None:
    """Every documented conflict category must be configured for oversampling."""
    tiny_registry.cfg.names.gold_sample_size = 1
    tiny_registry.cfg.names.gold_stratum_quotas = {
        "ntee_x_only": 1,
        "church_foundation_only": 0,
        "both_external_flags": 0,
        "neither_external_flag": 0,
    }
    tiny_registry.cfg.names.gold_conflict_quotas = {"saint_name": 0}
    pd.DataFrame(
        [
            {
                "EIN2": "001",
                "population": "bmf_only",
                "name_raw": "Saint Mary",
                "is_bmf_only": True,
                "is_manifest_contaminated": False,
                "is_ntee_x": True,
                "is_church_foundation": False,
            }
        ]
    ).to_parquet(tiny_registry.names_bmf_only_frame, index=False)

    with pytest.raises(ValueError, match="gold_conflict_quotas"):
        draw_name_gold(tiny_registry.cfg, tiny_registry)


def test_name_validation_requires_completed_name_gold_coding(tiny_registry) -> None:
    """The downstream validation seam cannot consume an uncoded names sample."""
    with pytest.raises(FileNotFoundError, match="Names gold manifest"):
        run_name_validation(tiny_registry.cfg, tiny_registry)
