"""Tests for T2.6: evidence-span guard."""

import json
from pathlib import Path

import pandas as pd
import pytest

from binary_classifier.annotate.schema import (
    AnnotationStore,
    BinaryLabel,
    LabelRecord,
    SourceType,
)
from binary_classifier.config import BinaryClassifierConfig
from binary_classifier.paths import PathRegistry
from binary_classifier.qc.evidence import (
    abstain_fabricated_positives,
    verify_evidence_spans,
)


def _make_registry(tmp_path: Path, text_map: dict[str, str]) -> PathRegistry:
    """Create a registry anchored in tmp_path with a synthetic missions parquet."""
    cfg = BinaryClassifierConfig()
    registry = PathRegistry.from_config(cfg, root=tmp_path)
    registry.ensure_dirs()

    missions_dir = (
        tmp_path / cfg.paths.raw_dir
    )
    missions_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"EIN2": ein2, "LONGEST_MISSION": text}
            for ein2, text in text_map.items()
        ]
    )
    df.to_parquet(missions_dir / "missions_cross_section.parquet", index=False)
    return registry


def _seed_store(registry: PathRegistry, records: list[LabelRecord]) -> None:
    store = AnnotationStore(registry.annotation_store)
    store.append_many(records)


def test_real_span_passes(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path, {"E1": "We preach the gospel every Sunday."}
    )
    record = LabelRecord(
        EIN2="E1",
        source_id="m__v1",
        source_type=SourceType.LLM_PROMPT,
        model_id="m",
        prompt_id="v1",
        temperature=0.0,
        confidence=0.9,
        binary_label=BinaryLabel.RELIGIOUS,
        evidence_spans=["preach the gospel"],
    )
    _seed_store(registry, [record])

    store = AnnotationStore(registry.annotation_store)
    ev = verify_evidence_spans(registry, store.to_frame())

    assert ev["total_spans"] == 1
    assert ev["verified_spans"] == 1
    assert ev["fabricated_spans"] == 0
    assert ev["fabrication_rate"] == 0.0
    assert ev["fabricated_records"] == []


def test_fabricated_span_flagged(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path, {"E1": "We preach the gospel every Sunday."}
    )
    record = LabelRecord(
        EIN2="E1",
        source_id="m__v1",
        source_type=SourceType.LLM_PROMPT,
        model_id="m",
        prompt_id="v1",
        temperature=0.0,
        confidence=0.9,
        binary_label=BinaryLabel.RELIGIOUS,
        evidence_spans=["fabricated nonsense"],
    )
    _seed_store(registry, [record])

    store = AnnotationStore(registry.annotation_store)
    ev = verify_evidence_spans(registry, store.to_frame())

    assert ev["total_spans"] == 1
    assert ev["verified_spans"] == 0
    assert ev["fabricated_spans"] == 1
    assert ev["fabrication_rate"] == 1.0
    assert ev["fabricated_records"] == [("E1", "m__v1", "fabricated nonsense")]


def test_empty_spans_no_count(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path, {"E1": "We preach the gospel every Sunday."}
    )
    record = LabelRecord(
        EIN2="E1",
        source_id="m__v1",
        source_type=SourceType.LLM_PROMPT,
        model_id="m",
        prompt_id="v1",
        temperature=0.0,
        confidence=0.9,
        binary_label=BinaryLabel.RELIGIOUS,
        evidence_spans=[],
    )
    _seed_store(registry, [record])

    store = AnnotationStore(registry.annotation_store)
    ev = verify_evidence_spans(registry, store.to_frame())

    assert ev["total_spans"] == 0
    assert ev["verified_spans"] == 0
    assert ev["fabricated_spans"] == 0
    assert ev["fabrication_rate"] == 0.0
    assert ev["fabricated_records"] == []


def test_null_spans_no_count(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path, {"E1": "We preach the gospel every Sunday."}
    )
    record = LabelRecord(
        EIN2="E1",
        source_id="m__v1",
        source_type=SourceType.LLM_PROMPT,
        model_id="m",
        prompt_id="v1",
        temperature=0.0,
        confidence=0.9,
        binary_label=BinaryLabel.RELIGIOUS,
        evidence_spans=None,
    )
    _seed_store(registry, [record])

    store = AnnotationStore(registry.annotation_store)
    ev = verify_evidence_spans(registry, store.to_frame())

    assert ev["total_spans"] == 0
    assert ev["fabrication_rate"] == 0.0


def test_rate_calculation(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path,
        {
            "E1": "We preach the gospel every Sunday.",
            "E2": "We feed the homeless daily.",
        },
    )
    records = [
        LabelRecord(
            EIN2="E1",
            source_id="m__v1",
            source_type=SourceType.LLM_PROMPT,
            model_id="m",
            prompt_id="v1",
            temperature=0.0,
            confidence=0.9,
            binary_label=BinaryLabel.RELIGIOUS,
            evidence_spans=["preach the gospel", "fabricated"],
        ),
        LabelRecord(
            EIN2="E2",
            source_id="m__v1",
            source_type=SourceType.LLM_PROMPT,
            model_id="m",
            prompt_id="v1",
            temperature=0.0,
            confidence=0.9,
            binary_label=BinaryLabel.NONRELIGIOUS,
            evidence_spans=["feed the homeless"],
        ),
    ]
    _seed_store(registry, records)

    store = AnnotationStore(registry.annotation_store)
    ev = verify_evidence_spans(registry, store.to_frame())

    assert ev["total_spans"] == 3
    assert ev["verified_spans"] == 2
    assert ev["fabricated_spans"] == 1
    assert ev["fabrication_rate"] == pytest.approx(1 / 3)


def test_missing_missions_raises(tmp_path: Path) -> None:
    cfg = BinaryClassifierConfig()
    registry = PathRegistry.from_config(cfg, root=tmp_path)
    registry.ensure_dirs()

    record = LabelRecord(
        EIN2="E1",
        source_id="m__v1",
        source_type=SourceType.LLM_PROMPT,
        model_id="m",
        prompt_id="v1",
        temperature=0.0,
        confidence=0.9,
        binary_label=BinaryLabel.RELIGIOUS,
        evidence_spans=["any"],
    )
    _seed_store(registry, [record])

    store = AnnotationStore(registry.annotation_store)
    with pytest.raises(FileNotFoundError):
        verify_evidence_spans(registry, store.to_frame())


def test_abstain_fabricated_positive(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path, {"E1": "We preach the gospel every Sunday."}
    )
    record = LabelRecord(
        EIN2="E1",
        source_id="m__v1",
        source_type=SourceType.LLM_PROMPT,
        model_id="m",
        prompt_id="v1",
        temperature=0.0,
        confidence=0.9,
        binary_label=BinaryLabel.RELIGIOUS,
        evidence_spans=["fabricated nonsense"],
    )
    _seed_store(registry, [record])

    store = AnnotationStore(registry.annotation_store)
    df = store.to_frame()
    ev = verify_evidence_spans(registry, df)
    df_abstained = abstain_fabricated_positives(df, ev["fabricated_records"])

    assert pd.isna(df_abstained.loc[0, "label"])
    assert pd.isna(df_abstained.loc[0, "binary_label"])


def test_abstain_only_positives(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path, {"E1": "We preach the gospel every Sunday."}
    )
    record = LabelRecord(
        EIN2="E1",
        source_id="m__v1",
        source_type=SourceType.LLM_PROMPT,
        model_id="m",
        prompt_id="v1",
        temperature=0.0,
        confidence=0.9,
        binary_label=BinaryLabel.NONRELIGIOUS,
        evidence_spans=["fabricated nonsense"],
    )
    _seed_store(registry, [record])

    store = AnnotationStore(registry.annotation_store)
    df = store.to_frame()
    ev = verify_evidence_spans(registry, df)
    df_abstained = abstain_fabricated_positives(df, ev["fabricated_records"])

    assert df_abstained.loc[0, "label"] == 0.0
    assert df_abstained.loc[0, "binary_label"] == "nonreligious"
