"""Tests for T1.4: pre-flight gate wiring."""

import json

import pandas as pd

from binary_classifier.qc.preflight import _STAGE_SPLITS, validate_gates


def _write_coding(registry, rows) -> None:
    """rows: list of (EIN2, split, human_label)."""
    df = pd.DataFrame(
        [{"EIN2": e, "split": s, "text": "t", "human_label": h} for e, s, h in rows]
    )
    registry.gold_coding_template.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(registry.gold_coding_template, index=False)
    # Matching gold manifest so the EIN2-drift check stays quiet.
    pd.DataFrame({"EIN2": [r[0] for r in rows]}).to_csv(
        registry.gold_manifest, index=False
    )


def _full_coding(registry) -> None:
    _write_coding(
        registry,
        [
            ("00-1", "prompt_dev", 1),
            ("00-2", "prompt_dev", 0),
            ("00-3", "validation", 1),
            ("00-4", "validation", 0),
        ],
    )


def _write_anchor_coding(registry, rows, manifest_eins=None) -> None:
    """Write an anchor coding template and optional anchor manifest."""
    df = pd.DataFrame(
        [{"EIN2": e, "tier": tier, "text": "t", "human_label": h} for e, tier, h in rows]
    )
    registry.anchor_coding_template.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(registry.anchor_coding_template, index=False)
    if manifest_eins is not None:
        registry.anchor_manifest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"EIN2": manifest_eins}).to_csv(
            registry.anchor_manifest,
            index=False,
        )


# ── G1: labels ───────────────────────────────────────────────────────────────


def test_complete_coding_passes_g1(tiny_config, tiny_registry) -> None:
    _full_coding(tiny_registry)
    assert validate_gates(tiny_config, tiny_registry, {"02", "04"}) == []


def test_blank_validation_row_flagged(tiny_config, tiny_registry) -> None:
    _write_coding(
        tiny_registry,
        [
            ("00-1", "prompt_dev", 1),
            ("00-3", "validation", 1),
            ("00-4", "validation", None),  # blank
        ],
    )
    problems = validate_gates(tiny_config, tiny_registry, {"04"})
    assert len(problems) == 1
    assert "validation" in problems[0]


def test_non_binary_value_flagged(tiny_config, tiny_registry) -> None:
    _write_coding(tiny_registry, [("00-1", "prompt_dev", 2)])
    problems = validate_gates(tiny_config, tiny_registry, {"02"})
    assert len(problems) == 1
    assert "prompt_dev" in problems[0]


def test_abstain_string_flagged(tiny_config, tiny_registry) -> None:
    _write_coding(tiny_registry, [("00-1", "prompt_dev", "abstain")])
    problems = validate_gates(tiny_config, tiny_registry, {"02"})
    assert problems


def test_missing_template_flagged(tiny_config, tiny_registry) -> None:
    problems = validate_gates(tiny_config, tiny_registry, {"02"})
    assert len(problems) == 1
    assert "not found" in problems[0]


# ── G2: slate ────────────────────────────────────────────────────────────────


def test_stage_03_without_slate_flags_g2(tiny_config, tiny_registry) -> None:
    _full_coding(tiny_registry)
    problems = validate_gates(tiny_config, tiny_registry, {"03"})
    assert len(problems) == 1
    assert problems[0].startswith("G2")


def test_confirmed_slate_passes_g2(tiny_config, tiny_registry) -> None:
    tiny_registry.production_slate.parent.mkdir(parents=True, exist_ok=True)
    tiny_registry.production_slate.write_text(
        json.dumps(
            {
                "confirmed": True,
                "models": [{"id": "gpt-4o-mini", "provider": "openai"}],
            }
        )
    )
    assert validate_gates(tiny_config, tiny_registry, {"03"}) == []


def test_unconfigured_model_flagged(tiny_config, tiny_registry) -> None:
    tiny_registry.production_slate.parent.mkdir(parents=True, exist_ok=True)
    tiny_registry.production_slate.write_text(
        json.dumps(
            {"confirmed": True, "models": [{"id": "mystery-model", "provider": "openai"}]}
        )
    )
    problems = validate_gates(tiny_config, tiny_registry, {"03"})
    assert any("not among the configured" in p for p in problems)


# ── G4: anchor labels ────────────────────────────────────────────────────────


def test_missing_anchor_template_flagged_for_g4(tiny_config, tiny_registry) -> None:
    problems = validate_gates(tiny_config, tiny_registry, {"09"})
    assert len(problems) == 1
    assert problems[0].startswith("G4")
    assert "not found" in problems[0]


def test_partial_anchor_template_ein_set_flagged(tiny_config, tiny_registry) -> None:
    _write_anchor_coding(
        tiny_registry,
        [("00-1", "high", 1)],
        manifest_eins=["00-1", "00-2"],
    )
    problems = validate_gates(tiny_config, tiny_registry, {"09"})
    assert len(problems) == 1
    assert "EIN2 set does not match" in problems[0]
    assert "1 missing" in problems[0]


def test_non_binary_anchor_label_flagged(tiny_config, tiny_registry) -> None:
    _write_anchor_coding(tiny_registry, [("00-1", "high", 2)])
    problems = validate_gates(tiny_config, tiny_registry, {"09"})
    assert len(problems) == 1
    assert "non-{0,1}" in problems[0]


def test_complete_anchor_labels_pass_g4(tiny_config, tiny_registry) -> None:
    _write_anchor_coding(
        tiny_registry,
        [("00-1", "high", 1), ("00-2", "medium", 0)],
        manifest_eins=["00-1", "00-2"],
    )
    assert validate_gates(tiny_config, tiny_registry, {"09"}) == []


# ── Stage selection ──────────────────────────────────────────────────────────


def test_only_stage_01_has_no_gates(tiny_config, tiny_registry) -> None:
    assert validate_gates(tiny_config, tiny_registry, {"01"}) == []


def test_stage_splits_include_training_and_test_entries() -> None:
    assert _STAGE_SPLITS["06"] == "validation"
    assert _STAGE_SPLITS["07"] == "test"
