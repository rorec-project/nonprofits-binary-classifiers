"""Tests for the T1.0 foundation: config schema, path registry, and factory."""

from pathlib import Path

import pytest

from binary_classifier.annotate.annotators.factory import make_annotator
from binary_classifier.annotate.annotators.openai_annotator import OpenAIAnnotator
from binary_classifier.annotate.annotators.vllm_annotator import VLLMAnnotator
from binary_classifier.config import (
    AnnotationConfig,
    BakeoffCandidate,
    BinaryClassifierConfig,
    load_config,
)
from binary_classifier.paths import PathRegistry

_REAL_CONFIG = Path("config/religious_missions.yaml")


# ── Config schema ──────────────────────────────────────────────────────────


def test_real_yaml_loads_with_slate_list() -> None:
    """The shipped YAML validates slate choices, monitor sizing, and QC gates."""
    cfg = load_config(_REAL_CONFIG)
    candidates = cfg.model_slate.bakeoff_candidates
    assert isinstance(candidates, list)
    assert all(isinstance(c, BakeoffCandidate) for c in candidates)
    # Three OpenAI tiers + one open-weight comparison arm.
    providers = [c.provider for c in candidates]
    assert providers.count("openai") == 3
    assert providers.count("vllm") == 1
    # GPT-5 tiers carry reasoning_effort; gpt-4o-mini does not.
    by_id = {c.id: c for c in candidates}
    assert by_id["gpt-5-mini-2025-08-07"].reasoning_effort == "minimal"
    assert by_id["gpt-4o-mini-2024-07-18"].reasoning_effort is None
    assert isinstance(cfg.model_slate.production, str)
    assert cfg.sample_sizes.gold == 450
    assert cfg.sample_sizes.monitor == 50
    assert cfg.annotation.openai_max_concurrency == 2
    assert cfg.annotation.vllm_max_concurrency == 8
    assert cfg.annotation.openai_batch is True
    assert cfg.annotation.openai_batch_poll_seconds == 30
    assert cfg.annotation.openai_batch_completion_window == "24h"
    assert cfg.qc.kappa_threshold == 0.70
    assert cfg.qc.f1_ci_floor == 0.70


def test_new_config_sections_have_defaults() -> None:
    """data.allow_synthetic, QC gates, and monitor sizes default sanely."""
    cfg = BinaryClassifierConfig()
    assert cfg.data.allow_synthetic is False
    assert cfg.qc.agreement_threshold == 0.85
    assert cfg.qc.kappa_threshold == 0.70
    assert cfg.qc.f1_ci_floor == 0.70
    assert cfg.sample_sizes.gold == 450
    assert cfg.sample_sizes.monitor == 50
    assert cfg.annotation.openai_max_concurrency == 2
    assert cfg.annotation.vllm_max_concurrency == 8
    assert cfg.annotation.openai_batch is False


def test_annotation_concurrency_must_be_positive() -> None:
    """Provider concurrency caps reject zero/negative values."""
    with pytest.raises(ValueError):
        AnnotationConfig(openai_max_concurrency=0)
    with pytest.raises(ValueError):
        AnnotationConfig(vllm_max_concurrency=0)
    with pytest.raises(ValueError):
        AnnotationConfig(openai_batch_poll_seconds=0)


def test_invalid_provider_rejected() -> None:
    """A provider outside {openai, vllm} fails config validation."""
    with pytest.raises(ValueError):
        BakeoffCandidate(id="x", provider="anthropic")  # type: ignore[arg-type]


# ── Path registry ──────────────────────────────────────────────────────────


def test_registry_exposes_new_paths(tiny_registry: PathRegistry, tmp_path) -> None:
    """Every new property is a Path under the configured root."""
    new_props = [
        "gold_dir",
        "interim_dir",
        "processed_dir",
        "gold_coding_template",
        "monitor_manifest",
        "proposed_slate",
        "production_slate",
        "bakeoff_results",
        "annotation_store",
        "bakeoff_store",
    ]
    for name in new_props:
        value = getattr(tiny_registry, name)
        assert isinstance(value, Path), f"{name} is not a Path"
        assert str(value).startswith(str(tmp_path)), f"{name} escapes root"

    # Artifacts land under the right parent dirs.
    assert tiny_registry.gold_coding_template.parent == tiny_registry.gold_dir
    assert tiny_registry.production_slate.parent == tiny_registry.gold_dir
    assert tiny_registry.annotation_store.parent == tiny_registry.interim_dir
    assert tiny_registry.proposed_slate.parent == tiny_registry.bakeoff_dir
    assert tiny_registry.bakeoff_store.parent == tiny_registry.bakeoff_dir
    assert tiny_registry.monitor_manifest.parent == (
        tiny_registry.interim_dir / "manifests"
    )


def test_ensure_dirs_creates_gold_and_interim(tiny_registry: PathRegistry) -> None:
    """ensure_dirs (already called by the fixture) makes gold/silver dirs."""
    assert tiny_registry.gold_dir.is_dir()
    assert tiny_registry.interim_dir.is_dir()


# ── Annotator factory ──────────────────────────────────────────────────────


def test_factory_routes_openai_and_forwards_reasoning_effort(
    tiny_config: BinaryClassifierConfig,
    monkeypatch,
) -> None:
    """openai spec → OpenAIAnnotator; reasoning_effort forwarded when set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    spec = BakeoffCandidate(id="gpt-5-mini", provider="openai", reasoning_effort="minimal")
    annotator = make_annotator(tiny_config, spec, "v1", "prompt text")
    assert isinstance(annotator, OpenAIAnnotator)
    assert annotator.prompt_id == "v1"
    assert annotator.reasoning_effort == "minimal"


def test_factory_openai_without_reasoning_effort(
    tiny_config: BinaryClassifierConfig,
    monkeypatch,
) -> None:
    """When reasoning_effort is unset, it is not forwarded (stays None)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    spec = BakeoffCandidate(id="gpt-4o-mini", provider="openai")
    annotator = make_annotator(tiny_config, spec, "v2", "prompt text")
    assert isinstance(annotator, OpenAIAnnotator)
    assert annotator.reasoning_effort is None


def test_factory_routes_vllm(tiny_config: BinaryClassifierConfig) -> None:
    """vllm spec → VLLMAnnotator (no API key required)."""
    spec = BakeoffCandidate(id="google/gemma-3-27b-it", provider="vllm")
    annotator = make_annotator(tiny_config, spec, "v3", "prompt text")
    assert isinstance(annotator, VLLMAnnotator)
    assert annotator.prompt_id == "v3"


def test_factory_forwards_guided_json_false_to_provider_annotators(
    tiny_config: BinaryClassifierConfig,
    monkeypatch,
) -> None:
    """Factory forwards guided_json so config can disable provider decoding."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    tiny_config.annotation.guided_json = False

    openai_spec = BakeoffCandidate(id="gpt-4o-mini", provider="openai")
    openai_annotator = make_annotator(tiny_config, openai_spec, "v1", "prompt text")
    assert isinstance(openai_annotator, OpenAIAnnotator)
    assert openai_annotator.guided_json is False

    vllm_spec = BakeoffCandidate(id="google/gemma-3-27b-it", provider="vllm")
    vllm_annotator = make_annotator(tiny_config, vllm_spec, "v1", "prompt text")
    assert isinstance(vllm_annotator, VLLMAnnotator)
    assert vllm_annotator.guided_json is False
