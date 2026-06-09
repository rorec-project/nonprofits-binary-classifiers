"""Tests for the T1.0 foundation: config schema, path registry, and factory."""

from pathlib import Path

import pytest

from binary_classifier.annotate.annotators.factory import make_annotator
from binary_classifier.annotate.annotators.openai_annotator import OpenAIAnnotator
from binary_classifier.annotate.annotators.vllm_annotator import VLLMAnnotator
from binary_classifier.config import BakeoffCandidate, BinaryClassifierConfig, load_config
from binary_classifier.paths import PathRegistry

_REAL_CONFIG = Path("config/religious_missions.yaml")


# ── Config schema ──────────────────────────────────────────────────────────


def test_real_yaml_loads_with_slate_list() -> None:
    """The shipped YAML validates into the new config-driven slate schema."""
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
    assert by_id["gpt-5-mini"].reasoning_effort == "minimal"
    assert by_id["gpt-4o-mini"].reasoning_effort is None
    assert isinstance(cfg.model_slate.production, str)


def test_new_config_sections_have_defaults() -> None:
    """data.allow_synthetic and qc.agreement_threshold default sanely."""
    cfg = BinaryClassifierConfig()
    assert cfg.data.allow_synthetic is False
    assert cfg.qc.agreement_threshold == 0.85


def test_invalid_provider_rejected() -> None:
    """A provider outside {openai, vllm} fails config validation."""
    with pytest.raises(ValueError):
        BakeoffCandidate(id="x", provider="anthropic")  # type: ignore[arg-type]


# ── Path registry ──────────────────────────────────────────────────────────


def test_registry_exposes_new_paths(tiny_registry: PathRegistry, tmp_path) -> None:
    """Every new property is a Path under the configured root."""
    new_props = [
        "gold_dir",
        "silver_dir",
        "gold_coding_template",
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
    assert tiny_registry.annotation_store.parent == tiny_registry.silver_dir
    assert tiny_registry.proposed_slate.parent == tiny_registry.results_dir


def test_ensure_dirs_creates_gold_and_silver(tiny_registry: PathRegistry) -> None:
    """ensure_dirs (already called by the fixture) makes gold/silver dirs."""
    assert tiny_registry.gold_dir.is_dir()
    assert tiny_registry.silver_dir.is_dir()


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
