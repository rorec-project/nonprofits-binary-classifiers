"""Tests for T2.1: QC metric bundle + CIs."""

import pandas as pd
import pytest

from binary_classifier.annotate.schema import (
    AnnotationStore,
    BinaryLabel,
    LabelRecord,
    SourceType,
)
from binary_classifier.metrics import compute_metric_bundle
from binary_classifier.qc.agreement import _compute_metrics, run_quality_check


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
    """Write coded validation rows and matching gold-manifest entries.

    Metric tests exercise the same stage-04 freeze path as production, where
    the post-gate leak guard requires the gold manifest before writing labels.
    """
    df = pd.DataFrame(
        [
            {"EIN2": e, "split": "validation", "text": "t", "human_label": h}
            for e, h in rows
        ]
    )
    registry.gold_coding_template.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(registry.gold_coding_template, index=False)
    pd.DataFrame(
        [{"EIN2": e, "split": "validation"} for e, _ in rows]
    ).to_csv(registry.gold_manifest, index=False)


def test_metric_bundle_matches_agreement_private_api() -> None:
    valid = pd.DataFrame(
        {
            "EIN2": ["00-1", "00-2", "00-3", "00-4"],
            "human_label": [1, 0, 1, 0],
            "silver_label": [1, 0, 0, 0],
            "silver_confidence": [0.9, 0.2, 0.4, 0.1],
        }
    )

    old_api = _compute_metrics(valid, seed=123)
    new_api = compute_metric_bundle(
        valid["human_label"].to_numpy(),
        valid["silver_label"].to_numpy(),
        y_score=valid["silver_confidence"].to_numpy(),
        minority_class=old_api["minority_class"],
        seed=123,
    )

    assert old_api == new_api


def test_metric_bundle_roc_auc_present_only_with_scores() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    with_scores = compute_metric_bundle(
        y_true,
        y_pred,
        y_score=[0.1, 0.9, 0.4, 0.2],
        minority_class=0,
        seed=123,
        n_resamples=10,
    )
    without_scores = compute_metric_bundle(
        y_true,
        y_pred,
        minority_class=0,
        seed=123,
        n_resamples=10,
    )

    assert with_scores["roc_auc"] == pytest.approx(1.0)
    assert "roc_auc" not in without_scores


def test_gate_passes_returns_full_metrics(tiny_config, tiny_registry) -> None:
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.NONRELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    result = run_quality_check(tiny_config, tiny_registry)
    assert result["agreement"] == 1.0
    frozen = tiny_registry.processed_dir / "silver_labels.csv"
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
    frozen = tiny_registry.processed_dir / "silver_labels.csv"
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
