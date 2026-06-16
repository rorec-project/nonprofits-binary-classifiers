"""Tests for T1.6: bake-off scoring, proposed slate, and arm degradation."""

import pandas as pd
import pytest

from binary_classifier.annotate.bakeoff_prompts import run_bakeoff
from binary_classifier.annotate.schema import (
    AnnotationStore,
    BinaryLabel,
    LabelRecord,
    SourceType,
)
from binary_classifier.config import BakeoffCandidate, load_slate


class _StubAnnotator:
    """Predicts RELIGIOUS (label 1.0) for every row — no network."""

    def __init__(self, spec: BakeoffCandidate, prompt_id: str) -> None:
        self.spec = spec
        self.prompt_id = prompt_id

    def annotate(self, text: str, ein2: str = "") -> LabelRecord:
        return LabelRecord(
            EIN2=ein2,
            source_id=f"{self.spec.id}__{self.prompt_id}",
            source_type=SourceType.LLM_PROMPT,
            model_id=self.spec.id,
            prompt_id=self.prompt_id,
            temperature=0.0,
            binary_label=BinaryLabel.RELIGIOUS,
        )


def _factory(fail_provider: str | None = None):
    def factory(spec, prompt_id, prompt_text):
        if fail_provider is not None and spec.provider == fail_provider:
            raise ConnectionError("vLLM server down")
        return _StubAnnotator(spec, prompt_id)

    return factory


def _write_template(registry) -> None:
    registry.gold_coding_template.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"EIN2": "00-1", "split": "prompt_dev", "text": "a church", "human_label": 1},
        {"EIN2": "00-2", "split": "prompt_dev", "text": "a charity", "human_label": 1},
        {"EIN2": "00-3", "split": "prompt_dev", "text": "a bank", "human_label": 0},
        {"EIN2": "00-9", "split": "validation", "text": "other", "human_label": 1},
    ]
    pd.DataFrame(rows).to_csv(registry.gold_coding_template, index=False)


def _prompts(tmp_path) -> list:
    paths = []
    for stem in ("v1", "v2"):
        p = tmp_path / f"{stem}.txt"
        p.write_text(f"prompt {stem}")
        paths.append(p)
    return paths


def test_bakeoff_emits_scores_and_unconfirmed_slate(tiny_config, tiny_registry, tmp_path):
    _write_template(tiny_registry)
    candidates = [BakeoffCandidate(id="m1", provider="openai")]
    out = run_bakeoff(
        tiny_config,
        tiny_registry,
        prompt_paths=_prompts(tmp_path),
        candidates=candidates,
        annotator_factory=_factory(),
    )
    results = out["results"]
    assert len(results) == 2  # 1 candidate × 2 prompts
    # Only prompt_dev rows were scored (3 of them), not the validation row.
    for r in results:
        assert isinstance(r["scores"]["accuracy"], float)
        assert r["scores"]["n_valid"] == 3
        # The imbalanced bundle is now computed and drives selection.
        bundle = r["scores"]["metrics"]
        assert isinstance(bundle, dict)
        assert "cohens_kappa" in bundle
        assert "lower" in bundle["bootstrap_ci"]["minority_f1"]

    # Selection is on κ + minority-F1 CI (freeze-gate criteria), not accuracy;
    # the legacy accuracy benchmark is reported but no longer drives the slate.
    assert out["proposed_slate"]["kappa_threshold"] == tiny_config.qc.kappa_threshold
    assert out["proposed_slate"]["f1_ci_floor"] == tiny_config.qc.f1_ci_floor
    assert "agreement_threshold" in out["proposed_slate"]

    # Proposed slate is written, unconfirmed, and production is NOT written.
    assert tiny_registry.proposed_slate.exists()
    assert tiny_registry.bakeoff_results.exists()
    assert not tiny_registry.production_slate.exists()
    slate = load_slate(tiny_registry.proposed_slate)
    assert slate.confirmed is False
    assert len(slate.models) >= 1


def test_bakeoff_missing_labels_raises(tiny_config, tiny_registry, tmp_path):
    # No template written → must raise, not return a silent note.
    with pytest.raises(FileNotFoundError):
        run_bakeoff(
            tiny_config,
            tiny_registry,
            prompt_paths=_prompts(tmp_path),
            candidates=[BakeoffCandidate(id="m1", provider="openai")],
            annotator_factory=_factory(),
        )


def test_bakeoff_vllm_arm_degrades_without_aborting(tiny_config, tiny_registry, tmp_path):
    _write_template(tiny_registry)
    candidates = [
        BakeoffCandidate(id="m1", provider="openai"),
        BakeoffCandidate(id="gemma", provider="vllm"),
    ]
    out = run_bakeoff(
        tiny_config,
        tiny_registry,
        prompt_paths=_prompts(tmp_path),
        candidates=candidates,
        annotator_factory=_factory(fail_provider="vllm"),
    )
    by_model = {(r["model_id"], r["prompt_id"]): r for r in out["results"]}
    # OpenAI arm scored normally.
    assert isinstance(by_model[("m1", "v1")]["scores"]["accuracy"], float)
    # vLLM arm carries an error, but did not abort the run.
    assert "error" in by_model[("gemma", "v1")]["scores"]
    # Proposed slate is built from the surviving (scored) arm.
    assert any(m["id"] == "m1" for m in out["proposed_slate"]["models"])


def test_bakeoff_deduplicates_predictions_before_scoring(
    tiny_config,
    tiny_registry,
    tmp_path,
) -> None:
    """Stale duplicate source rows do not inflate prompt-dev metrics."""
    _write_template(tiny_registry)
    store_path = tmp_path / "bakeoff_store.csv"
    AnnotationStore(store_path).append_many(
        [
            LabelRecord(
                EIN2="00-1",
                source_id="m1__v1",
                source_type=SourceType.LLM_PROMPT,
                model_id="m1",
                prompt_id="v1",
                temperature=0.0,
                binary_label=BinaryLabel.NONRELIGIOUS,
            ),
            LabelRecord(
                EIN2="00-1",
                source_id="m1__v1",
                source_type=SourceType.LLM_PROMPT,
                model_id="m1",
                prompt_id="v1",
                temperature=0.0,
                binary_label=BinaryLabel.RELIGIOUS,
            ),
        ]
    )

    out = run_bakeoff(
        tiny_config,
        tiny_registry,
        prompt_paths=[_prompts(tmp_path)[0]],
        candidates=[BakeoffCandidate(id="m1", provider="openai")],
        annotator_factory=_factory(),
        store_path=store_path,
    )

    assert out["results"][0]["scores"]["n_total"] == 3
    assert out["results"][0]["scores"]["n_valid"] == 3
