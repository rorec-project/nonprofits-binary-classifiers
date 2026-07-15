"""Tests for T1.7/T9: the QC agreement gate blocks unsafe freezes."""

import logging

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
    """Write coded validation rows and their gold-manifest entries.

    Stage 04 now needs the gold manifest at freeze time because the leak guard
    excludes held-out gold rows only after the validation agreement gate runs.
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


def _label(value: int) -> BinaryLabel:
    """Translate compact integer labels into the annotation enum for fixtures."""
    return BinaryLabel.RELIGIOUS if value == 1 else BinaryLabel.NONRELIGIOUS


def _balanced_rows(n_per_class: int) -> list[tuple[str, int]]:
    """Build a balanced validation set so κ mirrors the intended gate design."""
    rows = [(f"00-P{i:02d}", 1) for i in range(n_per_class)]
    rows.extend((f"00-N{i:02d}", 0) for i in range(n_per_class))
    return rows


def _seed_numeric_store(registry, rows: list[tuple[str, int]]) -> None:
    """Seed deterministic silver labels from integer fixture rows."""
    _seed_store(registry, [(ein2, _label(label)) for ein2, label in rows])


def _seed_one_error_store(registry, rows: list[tuple[str, int]]) -> None:
    """Seed rows with one wrong silver label to exercise the κ threshold."""
    predictions = rows.copy()
    first_ein2, first_label = predictions[0]
    predictions[0] = (first_ein2, 1 - first_label)
    _seed_numeric_store(registry, predictions)


def test_gate_passes_and_freezes(tiny_config, tiny_registry) -> None:
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.NONRELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    result = run_quality_check(tiny_config, tiny_registry)
    assert result["agreement"] == 1.0
    frozen = tiny_registry.processed_dir / "silver_labels.csv"
    assert frozen.exists()


def test_freeze_excludes_gold_manifest_rows_after_validation_gate(
    tiny_config,
    tiny_registry,
) -> None:
    """Gold rows score the gate but never enter ``silver_labels.csv``.

    The leak guard is an exclusion rule over the gold manifest, not a keep-list
    over the silver manifest. The overlapping ``00-BOTH`` row proves a held-out
    gold EIN2 is dropped even when it also appears in the silver draw.
    """
    _seed_store(
        tiny_registry,
        [
            ("00-SILVER", BinaryLabel.NONRELIGIOUS),
            ("00-PROMPT", BinaryLabel.RELIGIOUS),
            ("00-VALID1", BinaryLabel.RELIGIOUS),
            ("00-VALID0", BinaryLabel.NONRELIGIOUS),
            ("00-TEST", BinaryLabel.RELIGIOUS),
            ("00-MONITOR", BinaryLabel.NONRELIGIOUS),
            ("00-BOTH", BinaryLabel.RELIGIOUS),
        ],
    )
    _write_validation(tiny_registry, [("00-VALID1", 1), ("00-VALID0", 0)])
    pd.DataFrame({"EIN2": ["00-SILVER", "00-BOTH"]}).to_csv(
        tiny_registry.silver_manifest,
        index=False,
    )
    gold_rows = [
        {"EIN2": "00-PROMPT", "split": "prompt_dev"},
        {"EIN2": "00-VALID1", "split": "validation"},
        {"EIN2": "00-VALID0", "split": "validation"},
        {"EIN2": "00-TEST", "split": "test"},
        {"EIN2": "00-MONITOR", "split": "monitor"},
        {"EIN2": "00-BOTH", "split": "test"},
    ]
    pd.DataFrame(gold_rows).to_csv(tiny_registry.gold_manifest, index=False)

    result = run_quality_check(tiny_config, tiny_registry)

    assert result["n_valid"] == 2
    frozen = pd.read_csv(tiny_registry.processed_dir / "silver_labels.csv")
    gold_ein2s = {row["EIN2"] for row in gold_rows}
    assert set(frozen["EIN2"]).isdisjoint(gold_ein2s)
    assert set(frozen["EIN2"]) == {"00-SILVER"}


def test_run_quality_check_abstains_fabricated_positive_before_aggregation(
    tiny_config,
    tiny_registry,
) -> None:
    """Package QC mirrors script evidence abstention before aggregation.

    The fabricated positive vote would tie with the nonreligious vote if it
    reached majority voting. With the hallucination guard enabled, it becomes
    an abstention first and the remaining supported vote determines the silver
    label.
    """
    tiny_config.qc.abstain_on_fabricated_positive = True
    tiny_registry.missions_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "EIN2": "00-SILVER",
                "LONGEST_MISSION": "community services for neighbors",
            },
            {"EIN2": "00-VALID1", "LONGEST_MISSION": "church worship"},
            {"EIN2": "00-VALID0", "LONGEST_MISSION": "food pantry"},
        ]
    ).to_parquet(tiny_registry.missions_parquet, index=False)
    _seed_store(
        tiny_registry,
        [
            ("00-VALID1", BinaryLabel.RELIGIOUS),
            ("00-VALID0", BinaryLabel.NONRELIGIOUS),
        ],
    )
    _write_validation(tiny_registry, [("00-VALID1", 1), ("00-VALID0", 0)])

    store = AnnotationStore(tiny_registry.annotation_store)
    store.append_many(
        [
            LabelRecord(
                EIN2="00-SILVER",
                source_id="m__v1",
                source_type=SourceType.LLM_PROMPT,
                model_id="m",
                prompt_id="v1",
                temperature=0.0,
                confidence=0.9,
                binary_label=BinaryLabel.RELIGIOUS,
                evidence_spans=["invented chapel"],
            ),
            LabelRecord(
                EIN2="00-SILVER",
                source_id="m__v2",
                source_type=SourceType.LLM_PROMPT,
                model_id="m",
                prompt_id="v2",
                temperature=0.0,
                confidence=0.8,
                binary_label=BinaryLabel.NONRELIGIOUS,
                evidence_spans=["community services"],
            ),
        ]
    )

    result = run_quality_check(tiny_config, tiny_registry)

    assert result["n_valid"] == 2
    frozen = pd.read_csv(tiny_registry.processed_dir / "silver_labels.csv")
    row = frozen.loc[frozen["EIN2"] == "00-SILVER"].iloc[0]
    assert row["silver_label"] == 0.0
    assert row["num_votes"] == 1
    assert row["num_abstain"] == 1


def test_gate_blocks_when_minority_f1_ci_floor_fails(
    tiny_config,
    tiny_registry,
) -> None:
    """A perfect point estimate still blocks when the F1 CI lower bound fails."""
    tiny_config.qc.kappa_threshold = 0.70
    tiny_config.qc.f1_ci_floor = 0.50
    rows = [("00-P", 1), ("00-N", 0)]
    _seed_numeric_store(tiny_registry, rows)
    _write_validation(tiny_registry, rows)

    frozen = tiny_registry.processed_dir / "silver_labels.csv"
    with pytest.raises(ValueError, match="minority_f1_ci_lower") as exc_info:
        run_quality_check(tiny_config, tiny_registry)

    assert not frozen.exists()
    msg = str(exc_info.value)
    assert "cohens_kappa=1.000" in msg
    assert "krippendorff_alpha=1.000" in msg
    assert "minority_f1=1.000" in msg


def test_gate_freezes_when_kappa_and_f1_ci_floor_pass(
    tiny_config,
    tiny_registry,
) -> None:
    """The freeze proceeds only when both T9 gate conditions clear."""
    tiny_config.qc.kappa_threshold = 0.70
    tiny_config.qc.f1_ci_floor = 0.70
    rows = _balanced_rows(n_per_class=10)
    _seed_numeric_store(tiny_registry, rows)
    _write_validation(tiny_registry, rows)

    result = run_quality_check(tiny_config, tiny_registry)

    assert result["cohens_kappa"] == 1.0
    assert result["krippendorff_alpha"] == 1.0
    assert result["bootstrap_ci"]["minority_f1"]["lower"] >= 0.70
    assert (tiny_registry.processed_dir / "silver_labels.csv").exists()


def test_gate_uses_configured_kappa_threshold(
    tiny_config,
    tiny_registry,
) -> None:
    """Changing the config threshold changes the blocking decision."""
    rows = _balanced_rows(n_per_class=5)
    _seed_one_error_store(tiny_registry, rows)
    _write_validation(tiny_registry, rows)
    tiny_config.qc.f1_ci_floor = 0.0
    tiny_config.qc.kappa_threshold = 0.85

    with pytest.raises(ValueError, match="threshold=0.850"):
        run_quality_check(tiny_config, tiny_registry)

    tiny_config.qc.kappa_threshold = 0.70
    result = run_quality_check(tiny_config, tiny_registry)
    assert result["kappa_threshold"] == 0.70
    assert result["cohens_kappa"] >= tiny_config.qc.kappa_threshold


def test_raw_agreement_still_logged_and_reported(
    tiny_config,
    tiny_registry,
    caplog,
) -> None:
    """Raw agreement remains visible even though κ/F1-CI drive the gate."""
    tiny_config.qc.kappa_threshold = 0.70
    tiny_config.qc.f1_ci_floor = 0.0
    rows = [("00-P", 1), ("00-N", 0)]
    _seed_numeric_store(tiny_registry, rows)
    _write_validation(tiny_registry, rows)

    caplog.set_level(logging.INFO, logger="binary_classifier.qc.agreement")
    result = run_quality_check(tiny_config, tiny_registry)

    assert result["agreement"] == 1.0
    assert "Validation raw agreement (reported only)" in caplog.text


def test_pr_auc_uses_positive_class_score_not_winner_confidence(
    tiny_config,
    tiny_registry,
) -> None:
    """Confident negative labels should score low for positive-class AUC."""
    tiny_config.qc.f1_ci_floor = 0.0
    rows = [("00-P", 1), ("00-N", 0)]
    _seed_numeric_store(tiny_registry, rows)
    _write_validation(tiny_registry, rows)

    result = run_quality_check(tiny_config, tiny_registry)

    assert result["pr_auc"] == pytest.approx(1.0)


def test_gate_below_threshold_raises_and_writes_nothing(
    tiny_config, tiny_registry
) -> None:
    # 00-2 silver (religious) disagrees with human (0) → 50% < 85%.
    _seed_store(
        tiny_registry,
        [("00-1", BinaryLabel.RELIGIOUS), ("00-2", BinaryLabel.RELIGIOUS)],
    )
    _write_validation(tiny_registry, [("00-1", 1), ("00-2", 0)])
    frozen = tiny_registry.processed_dir / "silver_labels.csv"
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
