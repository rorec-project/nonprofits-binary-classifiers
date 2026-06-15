"""Tests for stage 05 anchor sampling."""

from pathlib import Path

import pandas as pd
import pytest

from binary_classifier.config import BinaryClassifierConfig
from binary_classifier.data import anchor as anchor_module
from binary_classifier.data.anchor import build_anchor
from binary_classifier.paths import PathRegistry


def _mission_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ein = 1000
    for tier, group, count in [
        ("HIGH", "A", 30),
        ("HIGH", "B", 30),
        ("MEDIUM", "A", 30),
        ("MEDIUM", "B", 30),
        ("LOW", "A", 40),
        ("LOW", "B", 20),
    ]:
        for i in range(count):
            rows.append(
                {
                    "EIN2": ein,
                    "mission_text": f"{tier} mission {group} {i}",
                    "ntee_major_group": group,
                    "is_truncated": False,
                    "NTEE_IRS": f"{group}01",
                    "data_source": "test",
                },
            )
            ein += 1
    return pd.DataFrame(rows)


def _fake_quality(text: str) -> float:
    tier = text.split()[0]
    return {"HIGH": 5.5, "MEDIUM": 3.5, "LOW": 1.0}[tier]


@pytest.fixture
def anchor_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tiny_config: BinaryClassifierConfig,
    tiny_registry: PathRegistry,
) -> tuple[pd.DataFrame, BinaryClassifierConfig, PathRegistry]:
    frame = _mission_frame()
    tiny_config.anchor.n = 60
    tiny_config.anchor.oversample_low_factor = 2.0
    tiny_config.anchor.min_stratum_frame = 1
    monkeypatch.setattr(anchor_module, "load_missions", lambda cfg: frame.copy())
    monkeypatch.setattr(anchor_module, "compute_quality_score", _fake_quality)
    return frame, tiny_config, tiny_registry


def test_anchor_allocation_weights_and_exclusions(
    anchor_fixture: tuple[pd.DataFrame, BinaryClassifierConfig, PathRegistry],
) -> None:
    frame, cfg, registry = anchor_fixture
    silver_excluded = frame.loc[[0], "EIN2"].iloc[0]
    gold_excluded = frame.loc[[1], "EIN2"].iloc[0]
    pd.DataFrame({"EIN2": [f" {silver_excluded} "]}).to_csv(
        registry.silver_manifest,
        index=False,
    )
    pd.DataFrame({"EIN2": [gold_excluded]}).to_csv(
        registry.gold_manifest,
        index=False,
    )

    build_anchor(cfg, registry)

    manifest = pd.read_csv(registry.anchor_manifest)
    template = pd.read_csv(registry.anchor_coding_template)

    assert len(manifest) == cfg.anchor.n
    assert list(manifest.columns) == [
        "EIN2",
        "stratum",
        "tier",
        "ntee_major_group",
        "sample_prob",
        "split",
    ]
    assert list(template.columns) == ["EIN2", "tier", "text", "human_label"]
    assert set(manifest["tier"]) == {"HIGH", "MEDIUM", "LOW"}
    assert set(manifest["split"]) == {"anchor"}
    assert silver_excluded not in set(manifest["EIN2"])
    assert gold_excluded not in set(manifest["EIN2"])

    frame_with_strata = frame.copy()
    frame_with_strata["tier"] = frame_with_strata["mission_text"].map(
        lambda text: text.split()[0],
    )
    frame_with_strata["stratum"] = (
        frame_with_strata["tier"] + "|" + frame_with_strata["ntee_major_group"]
    )
    frame_with_strata = frame_with_strata[
        ~frame_with_strata["EIN2"].isin({silver_excluded, gold_excluded})
    ]
    frame_counts = frame_with_strata.groupby("stratum").size()
    draw_counts = manifest.groupby("stratum").size()
    for stratum, n_drawn in draw_counts.items():
        probs = manifest.loc[manifest["stratum"] == stratum, "sample_prob"].unique()
        assert len(probs) == 1
        assert probs[0] == pytest.approx(n_drawn / frame_counts.loc[stratum])

    low_frame_share = (frame_with_strata["tier"] == "LOW").mean()
    low_sample_share = (manifest["tier"] == "LOW").mean()
    assert low_sample_share > low_frame_share


def test_anchor_is_deterministic(
    anchor_fixture: tuple[pd.DataFrame, BinaryClassifierConfig, PathRegistry],
) -> None:
    _, cfg, registry = anchor_fixture

    build_anchor(cfg, registry)
    first_manifest = pd.read_csv(registry.anchor_manifest)
    first_template = pd.read_csv(registry.anchor_coding_template)

    build_anchor(cfg, registry)
    second_manifest = pd.read_csv(registry.anchor_manifest)
    second_template = pd.read_csv(registry.anchor_coding_template)

    pd.testing.assert_frame_equal(first_manifest, second_manifest)
    pd.testing.assert_frame_equal(first_template, second_template)


def test_anchor_clobber_protection_and_force(
    anchor_fixture: tuple[pd.DataFrame, BinaryClassifierConfig, PathRegistry],
) -> None:
    _, cfg, registry = anchor_fixture
    build_anchor(cfg, registry)

    template = pd.read_csv(registry.anchor_coding_template)
    template.loc[0, "human_label"] = 1
    template.to_csv(registry.anchor_coding_template, index=False)

    with pytest.raises(RuntimeError, match="contains 1 human labels"):
        build_anchor(cfg, registry)

    build_anchor(cfg, registry, force=True)
    regenerated = pd.read_csv(registry.anchor_coding_template)
    assert regenerated["human_label"].isna().all()


def test_anchor_synthetic_path_end_to_end(
    tmp_path: Path,
    tiny_config: BinaryClassifierConfig,
    tiny_registry: PathRegistry,
) -> None:
    tiny_config.data.allow_synthetic = True
    tiny_config.paths.raw_dir = tmp_path / "missing_raw"
    tiny_config.anchor.n = 60

    build_anchor(tiny_config, tiny_registry)

    manifest = pd.read_csv(tiny_registry.anchor_manifest)
    template = pd.read_csv(tiny_registry.anchor_coding_template)
    assert len(manifest) == 60
    assert len(template) == 60
    assert list(manifest.columns) == [
        "EIN2",
        "stratum",
        "tier",
        "ntee_major_group",
        "sample_prob",
        "split",
    ]
    assert list(template.columns) == ["EIN2", "tier", "text", "human_label"]
