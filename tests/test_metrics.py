"""Tests for T2.1: QC metric bundle + CIs."""

import pandas as pd
import pytest

from binary_classifier.annotate.schema import (
    AnnotationStore,
    BinaryLabel,
    LabelRecord,
    SourceType,
)
from binary_classifier.qc.agreement import run_quality_check


def _seed_store(registry, rows, confidence=0.9) -> None:
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
                    confidence=confidence,
                    binary_label=label,
                )
            )
    store.append_many(records)


def _seed_store_no_confidence(registry, rows) -> None:
    """rows: list of (EIN2, BinaryLabel) — two prompt votes, no confidence."""
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
                    confidence=None,
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


def test_gate_passes_returns_full_metrics(tiny_config, tiny_registry) -> None:
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.NONRELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    result = run_quality_check(tiny_config, tiny_registry)
    assert result["agreement"] == 1.0
    frozen = tiny_registry.train_test_dir / "silver_labels.csv"
    assert frozen.exists()

    assert "confusion_matrix" in result
    cm = result["confusion_matrix"]
    assert cm == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}

    assert result["minority_class"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["mcc"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["cohens_kappa"] == 1.0
    assert result["pr_auc"] is not None

    assert "bootstrap_ci" in result
    ci = result["bootstrap_ci"]
    assert "accuracy" in ci
    assert "minority_f1" in ci
    assert ci["accuracy"]["lower"] is not None
    assert ci["accuracy"]["upper"] is not None
    assert ci["minority_f1"]["lower"] is not None
    assert ci["minority_f1"]["upper"] is not None


def test_gate_below_threshold_metrics_in_exception(tiny_config, tiny_registry) -> None:
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.RELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    frozen = tiny_registry.train_test_dir / "silver_labels.csv"
    with pytest.raises(ValueError, match="AGREEMENT GATE FAILED") as exc_info:
        run_quality_check(tiny_config, tiny_registry)
    assert not frozen.exists()
    msg = str(exc_info.value)
    assert "MCC=" in msg
    assert "cohens_kappa=" in msg


def test_absent_validation_labels_raises(tiny_config, tiny_registry) -> None:
    _seed_store(tiny_registry, [("00-1", BinaryLabel.RELIGIOUS)])
    with pytest.raises(FileNotFoundError):
        run_quality_check(tiny_config, tiny_registry)


def test_bootstrap_ci_runs(tiny_config, tiny_registry) -> None:
    _seed_store(
        tiny_registry,
        [
            ("00-1", BinaryLabel.RELIGIOUS),
            ("00-2", BinaryLabel.NONRELIGIOUS),
            ("00-3", BinaryLabel.RELIGIOUS),
            ("00-4", BinaryLabel.NONRELIGIOUS),
        ],
    )
    _write_validation(
        tiny_registry,
        [("00-1", 1), ("00-2", 0), ("00-3", 1), ("00-4", 0)],
    )
    result = run_quality_check(tiny_config, tiny_registry)
    ci = result["bootstrap_ci"]
    assert ci["accuracy"]["lower"] <= ci["accuracy"]["upper"]
    assert ci["minority_f1"]["lower"] <= ci["minority_f1"]["upper"]


def test_pr_auc_present_when_confidence(tiny_config, tiny_registry) -> None:
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.NONRELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    result = run_quality_check(tiny_config, tiny_registry)
    assert result["pr_auc"] is not None


def test_pr_auc_skipped_when_no_confidence(tiny_config, tiny_registry) -> None:
    _seed_store_no_confidence(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.NONRELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    result = run_quality_check(tiny_config, tiny_registry)
    assert result["pr_auc"] is None


def test_cohens_kappa_present(tiny_config, tiny_registry) -> None:
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.NONRELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    result = run_quality_check(tiny_config, tiny_registry)
    assert "cohens_kappa" in result
    assert result["cohens_kappa"] == 1.0
