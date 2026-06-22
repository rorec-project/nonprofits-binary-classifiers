"""Tests for the Stage 02 CLI helper functions."""

import importlib.util
import sys
from pathlib import Path

import pytest

from binary_classifier.config import BakeoffCandidate


def _load_stage02_cli():
    path = Path("scripts/02_bakeoff_prompts.py")
    spec = importlib.util.spec_from_file_location("stage02_cli_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage02_cli_for_test"] = module
    spec.loader.exec_module(module)
    return module


def test_stage02_only_model_selects_configured_candidate(tiny_config) -> None:
    module = _load_stage02_cli()
    tiny_config.model_slate.bakeoff_candidates = [
        BakeoffCandidate(id="openai-a", provider="openai"),
        BakeoffCandidate(id="vllm-a", provider="vllm"),
    ]

    selected = module._select_candidates(tiny_config, "vllm-a")

    assert [candidate.id for candidate in selected] == ["vllm-a"]
    assert module._select_candidates(tiny_config, None) is None
    with pytest.raises(ValueError, match="missing"):
        module._select_candidates(tiny_config, "missing")
