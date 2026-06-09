"""Tests for T1.4: G1 (labels) and G2 (slate) pre-flight gates."""

import json

import pandas as pd

from binary_classifier.qc.preflight import validate_gates


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


# ── Stage selection ──────────────────────────────────────────────────────────


def test_only_stage_01_has_no_gates(tiny_config, tiny_registry) -> None:
    assert validate_gates(tiny_config, tiny_registry, {"01"}) == []
