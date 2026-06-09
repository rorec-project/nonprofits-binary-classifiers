"""Tests for T1.7: the QC agreement gate blocks the freeze below threshold."""

import pandas as pd
import pytest

from binary_classifier.annotate.schema import (
    AnnotationStore,
    BinaryLabel,
    LabelRecord,
    SourceType,
)
from binary_classifier.qc.agreement import run_quality_check


def _seed_store(registry, rows) -> None:
    """rows: list of (EIN2, BinaryLabel) — two prompt votes written per row."""
    store = AnnotationStore(registry.annotation_store)
    records = []
    for ein2, label in rows:
        for prompt_id in ("v1", "v2"):
            records.append(
                LabelRecord(
                    EIN2=ein2,
                    source_id=f"m__{prompt_id}",
                    source_type=SourceType.LLM_PROMPT,
                    model_id="m",
                    prompt_id=prompt_id,
                    temperature=0.0,
                    confidence=0.9,
                    binary_label=label,
                )
            )
    store.append_many(records)


def _write_validation(registry, rows) -> None:
    """rows: list of (EIN2, human_label)."""
    df = pd.DataFrame(
        [
            {"EIN2": e, "split": "validation", "text": "t", "human_label": h}
            for e, h in rows
        ]
    )
    registry.gold_coding_template.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(registry.gold_coding_template, index=False)


def test_gate_passes_and_freezes(tiny_config, tiny_registry) -> None:
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.NONRELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    result = run_quality_check(tiny_config, tiny_registry)
    assert result["agreement"] == 1.0
    frozen = tiny_registry.train_test_dir / "silver_labels.csv"
    assert frozen.exists()


def test_gate_below_threshold_raises_and_writes_nothing(
    tiny_config, tiny_registry
) -> None:
    # 00-2 silver (religious) disagrees with human (0) → 50% < 85%.
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.RELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    frozen = tiny_registry.train_test_dir / "silver_labels.csv"
    with pytest.raises(ValueError, match="AGREEMENT GATE FAILED"):
        run_quality_check(tiny_config, tiny_registry)
    assert not frozen.exists()


def test_absent_validation_labels_raises(tiny_config, tiny_registry) -> None:
    _seed_store(tiny_registry, [("00-1", BinaryLabel.RELIGIOUS)])
    # No coding template written at all.
    with pytest.raises(FileNotFoundError):
        run_quality_check(tiny_config, tiny_registry)


def test_empty_store_raises(tiny_config, tiny_registry) -> None:
    _write_validation(tiny_registry, [("00-1", 1)])
    with pytest.raises(ValueError, match="empty"):
        run_quality_check(tiny_config, tiny_registry)
