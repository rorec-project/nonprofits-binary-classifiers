"""Tests for stage-01 sampling, gold coding, and monitor manifests."""

import pandas as pd
import pytest

from binary_classifier.config import (
    BinaryClassifierConfig,
    DataConfig,
    PathsConfig,
    QThresholdsConfig,
    SampleSizesConfig,
)
from binary_classifier.data.sample import (
    _write_gold_coding_template,
    build_gold_set,
    build_sample,
    build_silver_pool,
    split_human_sets,
    validate_label_set_disjointness,
)
from binary_classifier.paths import PathRegistry

_TEMPLATE_COLS = ["EIN2", "split", "text", "human_label"]


def _gold_all() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EIN2": ["00-1", "00-2", "00-3"],
            "split": ["prompt_dev", "validation", "test"],
            "mission_text": ["a church", "a charity", "a bank"],
        }
    )


def _human_split_frame(n_rows: int) -> pd.DataFrame:
    """Build a balanced gold-like frame for deterministic split-size tests.

    The rows cycle across four NTEE groups so prompt-dev and monitor can be
    carved stratified while leaving exactly comparable validation/test
    remainders.
    """
    groups = ["A", "B", "C", "D"]
    return pd.DataFrame(
        {
            "EIN2": [f"EIN-{i:03d}" for i in range(n_rows)],
            "mission_text": [f"mission text {i}" for i in range(n_rows)],
            "ntee_major_group": [groups[i % len(groups)] for i in range(n_rows)],
            "tier": "HIGH",
            "quota_cell_rate": 1.0,
            "is_positive_enriched": [i % 2 == 0 for i in range(n_rows)],
        }
    )


def _one_stratum_enrichment_frame() -> pd.DataFrame:
    """Build a fixture where positive and negative draw rates diverge.

    The single NTEE stratum has four religious-lexicon positives and sixteen
    secular negatives, so enriched sampling must write different per-cell
    inclusion probabilities rather than one ``10 / 20`` stratum marginal.
    """
    positive_rows = [
        {
            "EIN2": f"POS-{i:03d}",
            "mission_text": f"church worship services for families {i}",
            "ntee_major_group": "A",
            "Q": 5.5,
        }
        for i in range(4)
    ]
    negative_rows = [
        {
            "EIN2": f"NEG-{i:03d}",
            "mission_text": f"provide arts classes for families {i}",
            "ntee_major_group": "A",
            "Q": 5.5,
        }
        for i in range(16)
    ]
    return pd.DataFrame(positive_rows + negative_rows)


def _full_coverage_frame(per_stratum: int = 20) -> pd.DataFrame:
    """Build a 26-stratum frame with positive, negative, and boundary rows.

    Each NTEE major group gets ``per_stratum`` HIGH rows split across the three
    disjoint gold cells (clear positive, clear negative, boundary), so
    build_gold_set can realize every per-stratum target and the gold set lands
    on ``target_size`` exactly. The boundary slice is sized smaller than the
    boundary quota at gold=450 so the test also exercises the
    shortfall-rolls-into-filler path.
    """
    rows: list[dict] = []
    for group in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for i in range(per_stratum):
            if i < 6:
                text = f"church worship services for families {group}{i}"
            elif i < 12:
                text = f"youth ministry programs for the {group}{i} community"
            else:
                text = f"provide arts classes for families {group}{i}"
            rows.append(
                {
                    "EIN2": f"{group}-{i:03d}",
                    "mission_text": text,
                    "ntee_major_group": group,
                    "Q": 5.5,
                }
            )
    return pd.DataFrame(rows)


def _threshold_boundary_frame() -> pd.DataFrame:
    """Build rows that move tiers under a non-default Q threshold config.

    The Q=4.0 row is MEDIUM under defaults but LOW when MEDIUM is raised to
    4.5; the Q=5.2 row is HIGH under defaults but MEDIUM when HIGH is raised
    to 5.5. That makes ignored config thresholds visible in sampling outputs.
    """
    return pd.DataFrame(
        {
            "EIN2": ["LOWISH-ROW", "MID-ROW", "HIGHISH-ROW"],
            "mission_text": [
                "provide arts classes for families lowish",
                "provide arts classes for families medium",
                "provide arts classes for families highish",
            ],
            "ntee_major_group": ["A", "A", "A"],
            "Q": [4.0, 4.8, 5.2],
        }
    )


def _single_value(series: pd.Series) -> float:
    """Return the sole value when a test expects one probability per cell."""
    values = series.drop_duplicates().tolist()
    assert len(values) == 1
    return values[0]


def test_fresh_write_blank_4col_template(tiny_registry) -> None:
    _write_gold_coding_template(_gold_all(), tiny_registry)
    df = pd.read_csv(tiny_registry.gold_coding_template)
    assert list(df.columns) == _TEMPLATE_COLS
    assert len(df) == 3
    assert df["human_label"].isna().all()


def test_rerun_without_force_preserves_coded_labels(tiny_registry) -> None:
    _write_gold_coding_template(_gold_all(), tiny_registry)
    # Human codes a sentinel value.
    coded = pd.read_csv(tiny_registry.gold_coding_template)
    coded.loc[0, "human_label"] = 1
    coded.to_csv(tiny_registry.gold_coding_template, index=False)

    _write_gold_coding_template(_gold_all(), tiny_registry)  # no force
    after = pd.read_csv(tiny_registry.gold_coding_template)
    assert after.loc[0, "human_label"] == 1


def test_force_regenerates_blank(tiny_registry) -> None:
    _write_gold_coding_template(_gold_all(), tiny_registry)
    coded = pd.read_csv(tiny_registry.gold_coding_template)
    coded.loc[0, "human_label"] = 1
    coded.to_csv(tiny_registry.gold_coding_template, index=False)

    _write_gold_coding_template(_gold_all(), tiny_registry, force=True)
    after = pd.read_csv(tiny_registry.gold_coding_template)
    assert after["human_label"].isna().all()


def test_split_human_sets_monitor_disjoint_from_other_splits() -> None:
    """The held-out monitor slice must not leak into coding/eval splits."""
    prompt_dev, validation, test, monitor = split_human_sets(
        _human_split_frame(80),
        prompt_dev_size=12,
        monitor_size=8,
        seed=42,
    )

    monitor_ein2 = set(monitor["EIN2"])
    assert len(monitor) == 8
    assert set(monitor["split"]) == {"monitor"}
    assert monitor_ein2.isdisjoint(prompt_dev["EIN2"])
    assert monitor_ein2.isdisjoint(validation["EIN2"])
    assert monitor_ein2.isdisjoint(test["EIN2"])


def test_monitor_increment_preserves_validation_and_test_sizes() -> None:
    """Adding monitor rows to gold should not shrink validation/test."""
    _, base_validation, base_test, base_monitor = split_human_sets(
        _human_split_frame(60),
        prompt_dev_size=12,
        monitor_size=0,
        seed=42,
    )
    _, validation, test, monitor = split_human_sets(
        _human_split_frame(68),
        prompt_dev_size=12,
        monitor_size=8,
        seed=42,
    )

    assert base_monitor.empty
    assert len(monitor) == 8
    assert len(validation) == len(base_validation) == 24
    assert len(test) == len(base_test) == 24


def test_silver_inclusion_prob_uses_positive_negative_cell_rates() -> None:
    """Silver weights use the pos/neg cell rate, not the stratum marginal."""
    silver = build_silver_pool(
        _one_stratum_enrichment_frame(),
        target_size=10,
        seed=42,
        thresholds=QThresholdsConfig(),
        floor_groups=set(),
        cap_groups=set(),
    )

    positives = silver[silver["is_positive_enriched"]]
    negatives = silver[~silver["is_positive_enriched"]]

    assert len(positives) == 3
    assert len(negatives) == 7
    assert _single_value(positives["inclusion_prob"]) == pytest.approx(3 / 4)
    assert _single_value(negatives["inclusion_prob"]) == pytest.approx(7 / 16)
    assert _single_value(positives["inclusion_prob"]) != _single_value(
        negatives["inclusion_prob"]
    )
    assert set(silver["inclusion_prob"]) != {10 / 20}


def test_build_silver_pool_excludes_reserved_ein2s() -> None:
    """Silver helper can anti-join a preselected human label frame."""
    silver = build_silver_pool(
        _one_stratum_enrichment_frame(),
        target_size=10,
        seed=42,
        thresholds=QThresholdsConfig(),
        exclude_ein2={"POS-000", " NEG-000 "},
        floor_groups=set(),
        cap_groups=set(),
    )

    assert len(silver) == 10
    assert {"POS-000", "NEG-000"}.isdisjoint(silver["EIN2"])


def test_gold_quota_cell_rate_uses_quota_cell_rates() -> None:
    """Gold's ``quota_cell_rate`` reflects per-cell draw rates, not a marginal.

    ``target_size=52`` puts the single stratum at ``n_target=2`` so the disjoint
    clear-positive and clear-negative cells each draw exactly one row and the
    filler quota is zero (the fixture has no boundary rows). That isolates the
    two cell rates with no filler-cell rows muddying the per-cell rate. The
    column is named ``quota_cell_rate`` (diagnostic), not ``inclusion_prob``,
    because the gold filler top-up makes it an invalid design weight.
    """
    gold = build_gold_set(
        _one_stratum_enrichment_frame(),
        target_size=52,
        seed=42,
        thresholds=QThresholdsConfig(),
    )

    positives = gold[gold["is_positive_enriched"]]
    negatives = gold[~gold["is_positive_enriched"]]

    assert "inclusion_prob" not in gold.columns
    assert len(positives) == 1
    assert len(negatives) == 1
    assert _single_value(positives["quota_cell_rate"]) == pytest.approx(1 / 4)
    assert _single_value(negatives["quota_cell_rate"]) == pytest.approx(1 / 16)
    assert _single_value(positives["quota_cell_rate"]) != _single_value(
        negatives["quota_cell_rate"]
    )
    assert set(gold["quota_cell_rate"]) != {4 / 20}


def test_build_silver_pool_uses_configured_q_thresholds() -> None:
    """Silver sampling must use caller thresholds, not hidden defaults."""
    thresholds = QThresholdsConfig(HIGH=5.5, MEDIUM=4.5)

    default_silver = build_silver_pool(
        _threshold_boundary_frame(),
        target_size=10,
        seed=42,
        thresholds=QThresholdsConfig(),
        floor_groups=set(),
        cap_groups=set(),
    )
    custom_silver = build_silver_pool(
        _threshold_boundary_frame(),
        target_size=10,
        seed=42,
        thresholds=thresholds,
        floor_groups=set(),
        cap_groups=set(),
    )

    tiers_by_ein = custom_silver.set_index("EIN2")["tier"].to_dict()
    assert "LOWISH-ROW" in set(default_silver["EIN2"])
    assert set(custom_silver["EIN2"]) == {"MID-ROW", "HIGHISH-ROW"}
    assert tiers_by_ein["HIGHISH-ROW"] == "MEDIUM"


def test_gold_set_hits_target_size_exactly() -> None:
    """The quota-cell partition realizes the configured gold size exactly.

    The cells partition each stratum (boundary > positive > negative) and filler
    tops up from the disjoint remainder, so no row is drawn twice and no
    de-duplication trims the gold set below ``target_size`` (the bug that left
    the religious-missions gold pool at 444 instead of 450).
    """
    gold = build_gold_set(
        _full_coverage_frame(per_stratum=20),
        target_size=450,
        seed=42,
        thresholds=QThresholdsConfig(),
    )

    assert len(gold) == 450
    assert not gold["EIN2"].duplicated().any()
    assert gold["ntee_major_group"].nunique() == 26


def test_build_gold_set_uses_configured_q_thresholds() -> None:
    """Gold sampling must use caller thresholds for the human-coding frame."""
    thresholds = QThresholdsConfig(HIGH=5.5, MEDIUM=4.5)

    default_gold = build_gold_set(
        _threshold_boundary_frame(),
        target_size=520,
        seed=42,
        thresholds=QThresholdsConfig(),
    )
    custom_gold = build_gold_set(
        _threshold_boundary_frame(),
        target_size=520,
        seed=42,
        thresholds=thresholds,
    )

    tiers_by_ein = custom_gold.set_index("EIN2")["tier"].to_dict()
    assert "LOWISH-ROW" in set(default_gold["EIN2"])
    assert set(custom_gold["EIN2"]) == {"MID-ROW", "HIGHISH-ROW"}
    assert tiers_by_ein["HIGHISH-ROW"] == "MEDIUM"


def test_build_sample_fails_when_disjoint_samples_are_impossible(
    monkeypatch,
    tmp_path,
) -> None:
    """Stage 01 must not silently overlap gold and silver on tiny frames."""
    thresholds = QThresholdsConfig(HIGH=5.5, MEDIUM=4.5)
    frame = _threshold_boundary_frame()
    q_by_text = dict(zip(frame["mission_text"], frame["Q"], strict=True))
    cfg = BinaryClassifierConfig(
        q_thresholds=thresholds,
        data=DataConfig(allow_synthetic=True),
        sample_sizes=SampleSizesConfig(
            silver=10,
            gold=520,
            prompt_dev=0,
            monitor=0,
        ),
    )
    registry = PathRegistry.from_config(cfg, root=tmp_path)

    monkeypatch.setattr(
        "binary_classifier.data.sample.load_missions",
        lambda cfg: frame.drop(columns=["Q"]),
    )
    monkeypatch.setattr(
        "binary_classifier.data.sample.compute_quality_score",
        lambda text: q_by_text[text],
    )

    with pytest.raises(ValueError, match="Cannot build disjoint stage-01 samples"):
        build_sample(cfg, registry)


def test_build_sample_writes_disjoint_silver_and_gold(tmp_path) -> None:
    """Stage 01 constructs persisted silver/gold manifests disjointly."""
    cfg = BinaryClassifierConfig(
        data=DataConfig(allow_synthetic=True),
        sample_sizes=SampleSizesConfig(silver=400, gold=120, prompt_dev=20, monitor=10),
    )
    registry = PathRegistry.from_config(cfg, root=tmp_path)

    build_sample(cfg, registry)

    silver = pd.read_csv(registry.silver_manifest)
    gold = pd.read_csv(registry.gold_manifest)

    assert len(silver) == 400
    assert len(gold) == 120
    assert set(silver["EIN2"]).isdisjoint(gold["EIN2"])


def test_build_sample_emits_template_from_synthetic(tmp_path) -> None:
    """Synthetic stage 01 writes gold coding and monitor manifests."""
    cfg = BinaryClassifierConfig(
        data=DataConfig(allow_synthetic=True),
        sample_sizes=SampleSizesConfig(silver=200, gold=68, prompt_dev=10, monitor=8),
    )
    registry = PathRegistry.from_config(cfg, root=tmp_path)
    build_sample(cfg, registry)
    assert registry.gold_coding_template.exists()
    df = pd.read_csv(registry.gold_coding_template)
    assert list(df.columns) == _TEMPLATE_COLS
    assert df["human_label"].isna().all()
    assert len(df) > 0
    assert "monitor" in set(df["split"])

    assert registry.monitor_manifest.exists()
    monitor = pd.read_csv(registry.monitor_manifest)
    assert "EIN2" in monitor.columns
    assert set(monitor["split"]) == {"monitor"}
    silver = pd.read_csv(registry.silver_manifest)
    gold = pd.read_csv(registry.gold_manifest)
    assert set(silver["EIN2"]).isdisjoint(gold["EIN2"])


def test_label_set_disjointness_rejects_silver_gold_anchor_overlap() -> None:
    silver = pd.DataFrame({"EIN2": ["S1", "OVERLAP", "ANCHOR"]})
    gold = pd.DataFrame({"EIN2": ["G1", " OVERLAP "]})
    anchor = pd.DataFrame({"EIN2": ["ANCHOR"]})

    with pytest.raises(ValueError, match="OVERLAP") as excinfo:
        validate_label_set_disjointness(silver, gold, anchor)

    assert "ANCHOR" in str(excinfo.value)


def test_label_set_disjointness_allows_clean_sets() -> None:
    validate_label_set_disjointness(
        pd.DataFrame({"EIN2": ["S1", "S2"]}),
        pd.DataFrame({"EIN2": ["G1"]}),
        pd.DataFrame({"EIN2": ["A1"]}),
    )
