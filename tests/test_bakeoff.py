"""Tests for T1.6: bake-off scoring, proposed slate, and arm degradation."""

import json

import pandas as pd
import pytest

from binary_classifier.annotate.bakeoff_prompts import (
    _build_proposed_slate,
    rebuild_bakeoff_artifacts_from_store,
    rebuild_proposed_slate_from_results,
    run_bakeoff,
)
from binary_classifier.annotate.run_annotation import _selected_prompt_pairs
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


def _scored_result(
    model_id: str,
    prompt_id: str,
    *,
    f1: float,
    kappa: float,
    ci_lower: float,
    provider: str | None = "openai",
    abstain_rate: float = 0.0,
    n_valid: int = 50,
) -> dict:
    return {
        "model_id": model_id,
        "provider": provider,
        "reasoning_effort": None,
        "prompt_id": prompt_id,
        "source_id": f"{model_id}__{prompt_id}",
        "currently_configured": provider is not None,
        "scores": {
            "accuracy": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "f1": f1,
            "abstain_rate": abstain_rate,
            "n_valid": n_valid,
            "n_total": 50,
            "metrics": {
                "f1": f1,
                "cohens_kappa": kappa,
                "bootstrap_ci": {"minority_f1": {"lower": ci_lower}},
            },
        },
    }


class _SpyLimiter:
    """Context-manager probe for provider limiter wiring."""

    def __init__(self) -> None:
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


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
    assert out["proposed_slate"]["recommended"] == out["proposed_slate"]["selected"][0]
    assert isinstance(out["proposed_slate"]["rationale"], str)
    selected_row = out["proposed_slate"]["selected"][0]
    assert set(selected_row) == {
        "model_id",
        "prompt_id",
        "accuracy",
        "f1",
        "cohens_kappa",
        "f1_ci_lower",
        "abstain_rate",
        "n_valid",
        "clears",
    }
    assert (selected_row["model_id"], selected_row["prompt_id"]) in _selected_prompt_pairs(
        slate,
    )


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


def test_build_proposed_slate_emits_ranked_summary_and_recommendation() -> None:
    results = [
        _scored_result("m1", "v1", f1=0.80, kappa=0.75, ci_lower=0.72),
        _scored_result(
            "m2",
            "v2",
            f1=0.92,
            kappa=0.90,
            ci_lower=0.80,
            abstain_rate=0.4,
            n_valid=30,
        ),
        _scored_result("m3", "v3", f1=0.99, kappa=0.10, ci_lower=0.05),
    ]

    slate = _build_proposed_slate(
        results,
        kappa_threshold=0.7,
        f1_ci_floor=0.7,
        agreement_threshold=0.85,
    )

    assert slate["confirmed"] is False
    assert [row["model_id"] for row in slate["selected"]] == ["m2", "m1"]
    assert slate["recommended"] == slate["selected"][0]
    assert "clears both" in slate["rationale"]
    assert slate["selected"][0] == {
        "model_id": "m2",
        "prompt_id": "v2",
        "accuracy": 0.9,
        "f1": 0.92,
        "cohens_kappa": 0.9,
        "f1_ci_lower": 0.8,
        "abstain_rate": 0.4,
        "n_valid": 30,
        "clears": True,
    }
    assert {model["id"] for model in slate["models"]} == {"m1", "m2"}


def test_build_proposed_slate_fallback_keeps_contract() -> None:
    results = [
        _scored_result("m1", "v1", f1=0.70, kappa=0.60, ci_lower=0.30),
        _scored_result("m2", "v2", f1=0.85, kappa=0.65, ci_lower=0.40),
    ]

    slate_dict = _build_proposed_slate(
        results,
        kappa_threshold=0.7,
        f1_ci_floor=0.7,
        agreement_threshold=0.85,
    )

    assert slate_dict["selected"] == [
        {
            "model_id": "m2",
            "prompt_id": "v2",
            "accuracy": 0.9,
            "f1": 0.85,
            "cohens_kappa": 0.65,
            "f1_ci_lower": 0.4,
            "abstain_rate": 0.0,
            "n_valid": 50,
            "clears": False,
        },
    ]
    assert slate_dict["recommended"] == slate_dict["selected"][0]
    assert "fallback" in slate_dict["rationale"]


def test_build_proposed_slate_excludes_no_provider_arms_from_recommendation() -> None:
    results = [
        _scored_result(
            "old-model",
            "v9",
            provider=None,
            f1=0.99,
            kappa=0.95,
            ci_lower=0.90,
        ),
        _scored_result("m1", "v1", f1=0.50, kappa=0.20, ci_lower=0.10),
    ]

    slate_dict = _build_proposed_slate(
        results,
        kappa_threshold=0.7,
        f1_ci_floor=0.7,
        agreement_threshold=0.85,
    )

    assert slate_dict["recommended"]["model_id"] == "m1"
    assert slate_dict["selected"] == [
        {
            "model_id": "m1",
            "prompt_id": "v1",
            "accuracy": 0.9,
            "f1": 0.5,
            "cohens_kappa": 0.2,
            "f1_ci_lower": 0.1,
            "abstain_rate": 0.0,
            "n_valid": 50,
            "clears": False,
        },
    ]
    assert any(
        row["model_id"] == "old-model" and row["provider"] is None
        for row in slate_dict["arms"]
    )


def test_rebuild_slate_from_results_validates_and_yields_selected_pairs(
    tiny_config,
    tiny_registry,
) -> None:
    results = [
        _scored_result("m1", "v1", f1=0.80, kappa=0.75, ci_lower=0.72),
        _scored_result("m2", "v2", f1=0.92, kappa=0.90, ci_lower=0.80),
    ]
    tiny_registry.bakeoff_results.parent.mkdir(parents=True, exist_ok=True)
    tiny_registry.bakeoff_results.write_text(json.dumps(results))

    proposed = rebuild_proposed_slate_from_results(
        tiny_registry,
        kappa_threshold=tiny_config.qc.kappa_threshold,
        f1_ci_floor=tiny_config.qc.f1_ci_floor,
        agreement_threshold=tiny_config.qc.agreement_threshold,
    )

    assert tiny_registry.proposed_slate.exists()
    assert json.loads(tiny_registry.proposed_slate.read_text()) == proposed
    slate = load_slate(tiny_registry.proposed_slate)
    pairs = _selected_prompt_pairs(slate)
    assert pairs == {("m1", "v1"), ("m2", "v2")}


def test_full_store_bakeoff_artifacts_include_store_only_no_provider_arms(
    tiny_config,
    tiny_registry,
    tmp_path,
) -> None:
    _write_template(tiny_registry)
    prompt_path = _prompts(tmp_path)[0]
    store_path = tmp_path / "bakeoff_store.csv"
    AnnotationStore(store_path).append_many(
        [
            LabelRecord(
                EIN2="00-1",
                source_id="old-model__v9",
                source_type=SourceType.LLM_PROMPT,
                model_id="old-model",
                prompt_id="v9",
                temperature=0.0,
                binary_label=BinaryLabel.RELIGIOUS,
            ),
            LabelRecord(
                EIN2="00-2",
                source_id="old-model__v9",
                source_type=SourceType.LLM_PROMPT,
                model_id="old-model",
                prompt_id="v9",
                temperature=0.0,
                binary_label=BinaryLabel.RELIGIOUS,
            ),
            LabelRecord(
                EIN2="00-3",
                source_id="old-model__v9",
                source_type=SourceType.LLM_PROMPT,
                model_id="old-model",
                prompt_id="v9",
                temperature=0.0,
                binary_label=BinaryLabel.NONRELIGIOUS,
            ),
        ],
    )

    out = run_bakeoff(
        tiny_config,
        tiny_registry,
        prompt_paths=[prompt_path],
        candidates=[BakeoffCandidate(id="m1", provider="openai")],
        annotator_factory=_factory(),
        store_path=store_path,
    )

    by_source = {row["source_id"]: row for row in out["results"]}
    assert set(by_source) == {"m1__v1", "old-model__v9"}
    assert by_source["old-model__v9"]["provider"] is None
    assert by_source["old-model__v9"]["currently_configured"] is False

    proposed = out["proposed_slate"]
    assert proposed["selection_scope"] == "full_store"
    assert {row["source_id"] for row in proposed["arms"]} == set(by_source)
    assert proposed["not_currently_configured"] == [
        row for row in proposed["arms"] if row["source_id"] == "old-model__v9"
    ]
    assert proposed["recommended"]["model_id"] == "m1"
    assert all(row["model_id"] != "old-model" for row in proposed["selected"])

    slate = load_slate(tiny_registry.proposed_slate)
    assert _selected_prompt_pairs(slate) == {("m1", "v1")}
    written_results = json.loads(tiny_registry.bakeoff_results.read_text())
    assert {row["source_id"] for row in written_results} == set(by_source)

    rebuilt = rebuild_bakeoff_artifacts_from_store(
        tiny_config,
        tiny_registry,
        prompt_paths=[prompt_path],
        candidates=[BakeoffCandidate(id="m1", provider="openai")],
        store_path=store_path,
    )
    assert {row["source_id"] for row in rebuilt["results"]} == set(by_source)
    assert rebuilt["proposed_slate"]["recommended"]["model_id"] == "m1"


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


def test_bakeoff_wraps_annotation_calls_with_provider_limiter(
    tiny_config,
    tiny_registry,
    tmp_path,
    monkeypatch,
) -> None:
    """Stage 02 applies the OpenAI limiter across model×prompt workers."""
    _write_template(tiny_registry)
    spy = _SpyLimiter()
    monkeypatch.setattr(
        "binary_classifier.annotate.bakeoff_prompts.build_provider_limiters",
        lambda cfg: {"openai": spy},
    )

    run_bakeoff(
        tiny_config,
        tiny_registry,
        prompt_paths=_prompts(tmp_path),
        candidates=[BakeoffCandidate(id="m1", provider="openai")],
        annotator_factory=_factory(),
    )

    assert spy.entries == 6  # 3 prompt-dev rows × 2 prompts.


def test_bakeoff_ignores_stage03_openai_batch_flag(
    tiny_config,
    tiny_registry,
    tmp_path,
    monkeypatch,
) -> None:
    """Stage 02 bake-off stays on live calls when Stage 03 batch mode is on."""
    _write_template(tiny_registry)
    tiny_config.annotation.openai_batch = True
    spy = _SpyLimiter()
    monkeypatch.setattr(
        "binary_classifier.annotate.bakeoff_prompts.build_provider_limiters",
        lambda cfg: {"openai": spy},
    )

    run_bakeoff(
        tiny_config,
        tiny_registry,
        prompt_paths=_prompts(tmp_path),
        candidates=[BakeoffCandidate(id="m1", provider="openai")],
        annotator_factory=_factory(),
    )

    assert spy.entries == 6
