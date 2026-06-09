"""Tests for T1.3: prompt_id provenance, resume, and confirmed-slate consume."""

import json

import pandas as pd
import pytest

from binary_classifier.annotate.run_annotation import (
    resolve_production_specs,
    run_annotation_matrix,
)
from binary_classifier.annotate.schema import BinaryLabel, LabelRecord, SourceType
from binary_classifier.config import BakeoffCandidate


class _StubAnnotator:
    """A no-network annotator that echoes its (spec, prompt_id) provenance."""

    def __init__(self, spec: BakeoffCandidate, prompt_id: str, prompt_text: str) -> None:
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


def _stub_factory(spec, prompt_id, prompt_text):
    return _StubAnnotator(spec, prompt_id, prompt_text)


def _write_prompts(tmp_path) -> list:
    paths = []
    for stem in ("v1", "v2", "v3"):
        p = tmp_path / f"{stem}.txt"
        p.write_text(f"prompt {stem}")
        paths.append(p)
    return paths


def test_distinct_source_id_per_prompt(tmp_path) -> None:
    """One row under v1/v2/v3 yields three distinct source_ids."""
    prompts = _write_prompts(tmp_path)
    df = pd.DataFrame({"EIN2": ["00-1"], "text": ["a mission"]})
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    store = run_annotation_matrix(
        df=df,
        specs=specs,
        prompt_paths=prompts,
        store_path=tmp_path / "store.csv",
        annotator_factory=_stub_factory,
        checkpoint_every=1,
    )
    frame = store.to_frame()
    assert len(frame) == 3
    assert set(frame["source_id"]) == {"m1__v1", "m1__v2", "m1__v3"}
    assert set(frame["prompt_id"]) == {"v1", "v2", "v3"}


def test_resume_skips_done_no_duplicates(tmp_path) -> None:
    """A resume re-run adds no duplicate (EIN2, source_id) rows."""
    prompts = _write_prompts(tmp_path)
    df = pd.DataFrame({"EIN2": ["00-1", "00-2"], "text": ["m a", "m b"]})
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    store_path = tmp_path / "store.csv"

    run_annotation_matrix(
        df=df,
        specs=specs,
        prompt_paths=prompts,
        store_path=store_path,
        annotator_factory=_stub_factory,
        checkpoint_every=10,
    )
    # Second run with a fresh store object over the same file.
    store2 = run_annotation_matrix(
        df=df,
        specs=specs,
        prompt_paths=prompts,
        store_path=store_path,
        annotator_factory=_stub_factory,
        checkpoint_every=10,
        resume=True,
    )
    frame = store2.to_frame()
    # 2 rows × 3 prompts = 6, no duplicates after resume.
    assert len(frame) == 6
    assert not frame.duplicated(subset=["EIN2", "source_id"]).any()


# ── Confirmed production slate (gate G2 backstop) ────────────────────────────


def _write_slate(registry, confirmed: bool, models: list[dict]) -> None:
    registry.production_slate.parent.mkdir(parents=True, exist_ok=True)
    registry.production_slate.write_text(
        json.dumps({"confirmed": confirmed, "models": models})
    )


def test_resolve_production_specs_returns_listed_models(tiny_registry) -> None:
    """The production set is exactly what production_slate.json lists."""
    _write_slate(
        tiny_registry,
        confirmed=True,
        models=[
            {"id": "gpt-5-mini", "provider": "openai", "reasoning_effort": "minimal"},
            {"id": "google/gemma-3-27b-it", "provider": "vllm"},
        ],
    )
    specs = resolve_production_specs(tiny_registry)
    assert [s.id for s in specs] == ["gpt-5-mini", "google/gemma-3-27b-it"]
    assert specs[0].reasoning_effort == "minimal"


def test_resolve_production_specs_missing_raises(tiny_registry) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_production_specs(tiny_registry)


def test_resolve_production_specs_unconfirmed_raises(tiny_registry) -> None:
    _write_slate(tiny_registry, confirmed=False, models=[{"id": "m", "provider": "vllm"}])
    with pytest.raises(ValueError, match="not confirmed"):
        resolve_production_specs(tiny_registry)


def test_resolve_production_specs_empty_raises(tiny_registry) -> None:
    _write_slate(tiny_registry, confirmed=True, models=[])
    with pytest.raises(ValueError, match="no models"):
        resolve_production_specs(tiny_registry)
