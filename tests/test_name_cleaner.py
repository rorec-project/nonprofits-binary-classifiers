"""Tests for names-arm cleaning and its divergence gate."""

import json

import pandas as pd
import pytest

from binary_classifier.data.quality import RELIGIOUS_LEXICON
from binary_classifier.names.cleaner import clean_names, normalize_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FIRST AME ZION CHURCH, INC.", "First AME Zion Church"),
        ("COGIC COMMUNITY CENTER LLC", "COGIC Community Center"),
        ("ELCA COMMUNITY CENTER LLC", "ELCA Community Center"),
        ("LDS COMMUNITY CENTER LLC", "LDS Community Center"),
        ("CAFÃ‰ MINISTRIES, INC", "Café Ministries"),
        (None, ""),
        (pd.NA, ""),
        (float("nan"), ""),
        ("   ", ""),
    ],
)
def test_normalize_name_repairs_suffixes_encoding_and_case(raw, expected) -> None:
    assert normalize_name(raw) == expected


def test_normalize_name_is_idempotent_and_preserves_religious_lexicon() -> None:
    for token in RELIGIOUS_LEXICON:
        cleaned = normalize_name(token)
        assert normalize_name(cleaned) == cleaned
        assert all(word in cleaned.lower().split() for word in token.split())


def _write_name_frames(registry, *, panel_name: str, bare_name: str) -> None:
    registry.names_panel_frame.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "name_raw": panel_name,
                "name_bare": bare_name,
                "population": "panel_scoped",
            },
            {
                "EIN2": "P002",
                "name_raw": "FOO BAR INC.",
                "name_bare": "Foo Baz",
                "population": "panel_scoped",
            },
        ],
    ).to_parquet(registry.names_panel_frame, index=False)
    pd.DataFrame(
        [{"EIN2": "B001", "name_raw": "BMF FOUNDATION LLC", "population": "bmf_only"}],
    ).to_parquet(registry.names_bmf_only_frame, index=False)


def test_clean_names_writes_both_cleaned_frames_and_passing_audit(tiny_registry) -> None:
    _write_name_frames(
        tiny_registry,
        panel_name="FIRST AME ZION CHURCH, INC.",
        bare_name="First AME Zion Church",
    )
    clean_names(tiny_registry.cfg, tiny_registry)

    panel = pd.read_parquet(tiny_registry.names_panel_cleaned)
    bmf = pd.read_parquet(tiny_registry.names_bmf_only_cleaned)
    audit = json.loads(tiny_registry.names_divergence_audit.read_text())
    assert panel["name_cleaned"].tolist() == ["First AME Zion Church", "Foo Bar"]
    assert bmf["name_cleaned"].tolist() == ["Bmf Foundation"]
    assert audit["blocking_divergences"] == []
    assert audit["nonblocking_divergence_count"] == 0


def test_clean_names_does_not_block_religious_token_only_in_upstream_bare_name(
    tiny_registry,
) -> None:
    _write_name_frames(
        tiny_registry,
        panel_name="MT DESERT ISLAND YMCA",
        bare_name="Mount Desert Island Young Mens Christian Association",
    )

    clean_names(tiny_registry.cfg, tiny_registry)

    audit = json.loads(tiny_registry.names_divergence_audit.read_text())
    assert audit["blocking_divergences"] == []


def test_clean_names_blocks_religious_token_loss(tiny_registry, monkeypatch) -> None:
    _write_name_frames(
        tiny_registry,
        panel_name="FIRST MINISTRIES INC.",
        bare_name="First Ministries",
    )
    monkeypatch.setattr(
        "binary_classifier.names.cleaner.normalize_name",
        lambda raw: "First",
    )

    with pytest.raises(ValueError, match="religious"):
        clean_names(tiny_registry.cfg, tiny_registry)


@pytest.mark.parametrize(
    ("acronym", "bare_name"),
    [
        ("AME", "First AME Church"),
        ("AME", "First Ame Church"),
        ("COGIC", "First Cogic Church"),
        ("ELCA", "First Elca Church"),
        ("LDS", "First Lds Church"),
    ],
)
def test_clean_names_blocks_denominational_acronym_loss(
    tiny_registry,
    monkeypatch,
    acronym,
    bare_name,
) -> None:
    _write_name_frames(
        tiny_registry,
        panel_name=f"FIRST {acronym} CHURCH INC.",
        bare_name=bare_name,
    )
    monkeypatch.setattr(
        "binary_classifier.names.cleaner.normalize_name",
        lambda raw: "First Church",
    )

    with pytest.raises(ValueError, match="acronym"):
        clean_names(tiny_registry.cfg, tiny_registry)


def test_clean_names_does_not_block_non_denominational_uppercase_tokens(
    tiny_registry,
) -> None:
    _write_name_frames(
        tiny_registry,
        panel_name="ABC FOOD BANK INC.",
        bare_name="ABC Food Bank",
    )

    clean_names(tiny_registry.cfg, tiny_registry)
