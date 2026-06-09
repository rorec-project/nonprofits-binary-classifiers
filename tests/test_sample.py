"""Tests for T1.1: gold coding template emit + clobber protection."""

import pandas as pd

from binary_classifier.config import (
    BinaryClassifierConfig,
    DataConfig,
    PathsConfig,
    SampleSizesConfig,
)
from binary_classifier.data.sample import _write_gold_coding_template, build_sample
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


def test_build_sample_emits_template_from_synthetic(tmp_path) -> None:
    cfg = BinaryClassifierConfig(
        paths=PathsConfig(upstream_repo=tmp_path),
        data=DataConfig(allow_synthetic=True),
        sample_sizes=SampleSizesConfig(silver=200, gold=60, prompt_dev=10),
    )
    registry = PathRegistry.from_config(cfg, root=tmp_path)
    build_sample(cfg, registry)
    assert registry.gold_coding_template.exists()
    df = pd.read_csv(registry.gold_coding_template)
    assert list(df.columns) == _TEMPLATE_COLS
    assert df["human_label"].isna().all()
    assert len(df) > 0
