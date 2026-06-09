"""Tests for T1.5: orchestrator wiring of gates G1 (labels) and G2 (slate)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# scripts/ is not a package; load the orchestrator module by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pipeline as rp  # noqa: E402


def _spy_stages(monkeypatch, calls: list) -> None:
    monkeypatch.setattr(
        "binary_classifier.data.sample.build_sample",
        lambda *a, **k: calls.append("01"),
    )
    monkeypatch.setattr(
        "binary_classifier.annotate.bakeoff_prompts.run_bakeoff",
        lambda *a, **k: calls.append("02"),
    )
    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation.run_annotation",
        lambda *a, **k: calls.append("03"),
    )
    monkeypatch.setattr(
        "binary_classifier.qc.agreement.run_quality_check",
        lambda *a, **k: calls.append("04"),
    )


def _write_coded_template(registry) -> None:
    df = pd.DataFrame(
        [
            {"EIN2": "00-1", "split": "prompt_dev", "text": "t", "human_label": 1},
            {"EIN2": "00-2", "split": "validation", "text": "t", "human_label": 0},
        ]
    )
    registry.gold_coding_template.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(registry.gold_coding_template, index=False)


def _write_confirmed_slate(registry) -> None:
    registry.production_slate.parent.mkdir(parents=True, exist_ok=True)
    registry.production_slate.write_text(
        json.dumps(
            {"confirmed": True, "models": [{"id": "gpt-4o-mini", "provider": "openai"}]}
        )
    )


def test_no_labels_runs_01_then_exits_at_g1(tiny_config, tiny_registry, monkeypatch):
    calls: list = []
    _spy_stages(monkeypatch, calls)
    with pytest.raises(SystemExit) as exc:
        rp.run_pipeline(tiny_config, tiny_registry, {"01", "02", "03", "04"})
    assert exc.value.code == 2
    # Stage 01 ran; no model stage (02/03/04) was touched.
    assert calls == ["01"]


def test_labels_but_no_slate_runs_01_02_then_exits_at_g2(
    tiny_config, tiny_registry, monkeypatch
):
    _write_coded_template(tiny_registry)
    calls: list = []
    _spy_stages(monkeypatch, calls)
    with pytest.raises(SystemExit) as exc:
        rp.run_pipeline(tiny_config, tiny_registry, {"01", "02", "03", "04"})
    assert exc.value.code == 2
    # 02 produced the proposed slate; 03 was gated before any annotation.
    assert calls == ["01", "02"]


def test_full_run_with_confirmed_slate_proceeds(
    tiny_config, tiny_registry, monkeypatch
):
    _write_coded_template(tiny_registry)
    _write_confirmed_slate(tiny_registry)
    calls: list = []
    _spy_stages(monkeypatch, calls)
    rp.run_pipeline(tiny_config, tiny_registry, {"01", "02", "03", "04"})
    assert calls == ["01", "02", "03", "04"]


def test_phase_two_stages_03_04(tiny_config, tiny_registry, monkeypatch):
    _write_coded_template(tiny_registry)
    _write_confirmed_slate(tiny_registry)
    calls: list = []
    _spy_stages(monkeypatch, calls)
    rp.run_pipeline(tiny_config, tiny_registry, {"03", "04"})
    assert calls == ["03", "04"]
