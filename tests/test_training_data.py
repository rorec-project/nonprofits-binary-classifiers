"""Tests for training-frame construction and split helpers."""

import pandas as pd
import pytest

from binary_classifier.annotate.schema import (
    AnnotationStore,
    BinaryLabel,
    LabelRecord,
    SourceType,
)
from binary_classifier.train import data as train_data
from binary_classifier.train.data import (
    build_training_frame,
    load_human_split,
    split_dev,
    subset_fraction,
)


def _write_manifests(tiny_registry, *, gold=(), anchor=()) -> None:
    """Write leak-guard manifests for tests."""
    pd.DataFrame({"EIN2": list(gold)}).to_csv(
        tiny_registry.gold_manifest,
        index=False,
    )
    pd.DataFrame({"EIN2": list(anchor)}).to_csv(
        tiny_registry.anchor_manifest,
        index=False,
    )


def _write_silver(tiny_registry, rows: list[tuple[object, float | None]]) -> None:
    """Write a tiny frozen silver-label artifact."""
    pd.DataFrame(
        [{"EIN2": ein2, "silver_label": label} for ein2, label in rows],
    ).to_csv(tiny_registry.silver_labels, index=False)


def _mission_frame(rows: list[tuple[object, str, str]]) -> pd.DataFrame:
    """Build a load_missions-compatible fixture frame."""
    return pd.DataFrame(
        [
            {"EIN2": ein2, "mission_text": text, "ntee_major_group": ntee}
            for ein2, text, ntee in rows
        ],
    )


def _seed_votes(tiny_registry, votes: dict[str, list[int | None]]) -> None:
    """Write numeric and abstain votes to the annotation store."""
    records = []
    for ein2, labels in votes.items():
        for i, label in enumerate(labels):
            if label == 1:
                binary_label = BinaryLabel.RELIGIOUS
            elif label == 0:
                binary_label = BinaryLabel.NONRELIGIOUS
            else:
                binary_label = BinaryLabel.AMBIGUOUS_REVIEW
            records.append(
                LabelRecord(
                    EIN2=ein2,
                    source_id=f"source-{i}",
                    source_type=SourceType.LLM_PROMPT,
                    model_id="model",
                    prompt_id=f"p{i}",
                    temperature=0.0,
                    confidence=0.9,
                    binary_label=binary_label,
                ),
            )
    AnnotationStore(tiny_registry.annotation_store).append_many(records)


def _frame_for_splits(n_per_label: int = 10) -> pd.DataFrame:
    """Build a balanced training frame for split/subset tests."""
    rows = []
    for label in (0, 1):
        for i in range(n_per_label):
            rows.append(
                {
                    "EIN2": f"{label}-{i:02d}",
                    "text": f"text {label}-{i}",
                    "ntee_major_group": "X",
                    "p_pos": float(label),
                    "hard_label": label,
                },
            )
    return pd.DataFrame(rows)


def test_build_training_frame_recomputes_p_pos_and_drops_nan_silver(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """Soft labels come from long-store votes, excluding abstentions."""
    _write_silver(
        tiny_registry,
        [("E1", 1), ("E2", 1), ("E3", 0), ("E4", None)],
    )
    _seed_votes(
        tiny_registry,
        {
            "E1": [1, 0, None],
            "E2": [1, 1],
            "E3": [0, 0],
            "E4": [1],
        },
    )
    _write_manifests(tiny_registry)
    monkeypatch.setattr(
        train_data,
        "load_missions",
        lambda cfg: _mission_frame(
            [
                ("E1", "mission one", "A"),
                ("E2", "mission two", "B"),
                ("E3", "mission three", "C"),
                ("E4", "mission four", "D"),
            ],
        ),
    )

    frame = build_training_frame(tiny_config, tiny_registry)

    assert list(frame.columns) == [
        "EIN2",
        "text",
        "ntee_major_group",
        "p_pos",
        "hard_label",
    ]
    assert set(frame["EIN2"]) == {"E1", "E2", "E3"}
    p_pos = dict(zip(frame["EIN2"], frame["p_pos"], strict=True))
    assert p_pos == {"E1": 0.5, "E2": 1.0, "E3": 0.0}
    assert frame.set_index("EIN2").loc["E1", "text"] == "mission one"
    assert frame["hard_label"].tolist() == [1, 1, 0]


def test_build_training_frame_raises_on_zero_non_abstain_votes(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """A non-abstain silver label with only abstain votes is corruption."""
    _write_silver(tiny_registry, [("E1", 1), ("E2", 0)])
    _seed_votes(tiny_registry, {"E1": [1, 0], "E2": [None, None]})
    _write_manifests(tiny_registry)
    monkeypatch.setattr(
        train_data,
        "load_missions",
        lambda cfg: _mission_frame(
            [("E1", "mission one", "A"), ("E2", "mission two", "B")],
        ),
    )

    with pytest.raises(ValueError, match="1 EIN2"):
        build_training_frame(tiny_config, tiny_registry)


def test_build_training_frame_blocks_gold_overlap_with_dtype_drift(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """Leak guard normalizes EIN2 strings before manifest comparison."""
    _write_silver(tiny_registry, [(123, 1)])
    _seed_votes(tiny_registry, {"123": [1, 1]})
    _write_manifests(tiny_registry, gold=[" 123 "], anchor=[])
    monkeypatch.setattr(
        train_data,
        "load_missions",
        lambda cfg: _mission_frame([("123", "mission", "A")]),
    )

    with pytest.raises(ValueError, match="gold manifest: 1 EIN2"):
        build_training_frame(tiny_config, tiny_registry)


def test_build_training_frame_blocks_anchor_overlap(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """Anchor rows are excluded from training alongside held-out gold rows."""
    _write_silver(tiny_registry, [("A-1", 0)])
    _seed_votes(tiny_registry, {"A-1": [0, 0]})
    _write_manifests(tiny_registry, gold=[], anchor=["A-1"])
    monkeypatch.setattr(
        train_data,
        "load_missions",
        lambda cfg: _mission_frame([("A-1", "mission", "A")]),
    )

    with pytest.raises(ValueError, match="anchor manifest: 1 EIN2"):
        build_training_frame(tiny_config, tiny_registry)


def test_split_dev_is_deterministic_and_stratified() -> None:
    """Dev splits are reproducible and preserve hard-label strata."""
    frame = _frame_for_splits(n_per_label=10)

    train_a, dev_a = split_dev(frame, dev_fraction=0.3, seed=7)
    train_b, dev_b = split_dev(frame, dev_fraction=0.3, seed=7)

    assert dev_a["EIN2"].tolist() == dev_b["EIN2"].tolist()
    assert train_a["EIN2"].tolist() == train_b["EIN2"].tolist()
    assert dev_a["hard_label"].value_counts().to_dict() == {0: 3, 1: 3}
    assert train_a["hard_label"].value_counts().to_dict() == {0: 7, 1: 7}
    assert set(train_a["EIN2"]).isdisjoint(set(dev_a["EIN2"]))


def test_subset_fraction_is_label_stratified_and_nested() -> None:
    """Stable-hash prefixes make smaller fractions subsets of larger ones."""
    frame = _frame_for_splits(n_per_label=8)

    sub25 = subset_fraction(frame, fraction=0.25, seed=11)
    sub50 = subset_fraction(frame, fraction=0.5, seed=11)
    sub100 = subset_fraction(frame, fraction=1.0, seed=11)

    assert sub25["hard_label"].value_counts().to_dict() == {0: 2, 1: 2}
    assert sub50["hard_label"].value_counts().to_dict() == {0: 4, 1: 4}
    assert set(sub25["EIN2"]).issubset(set(sub50["EIN2"]))
    assert set(sub50["EIN2"]).issubset(set(sub100["EIN2"]))
    assert set(sub100["EIN2"]) == set(frame["EIN2"])


def test_load_human_split_raises_for_test_split(tiny_config, tiny_registry) -> None:
    """Training code must never unlock the human test split."""
    pd.DataFrame(
        [
            {
                "EIN2": "T-1",
                "split": "test",
                "text": "held-out",
                "human_label": 1,
            },
        ],
    ).to_csv(tiny_registry.gold_coding_template, index=False)

    with pytest.raises(ValueError, match="test split"):
        load_human_split(tiny_config, tiny_registry, "test")


def test_load_human_split_loads_non_test_split(tiny_config, tiny_registry) -> None:
    """Non-test human splits are returned with normalized EIN2s and int labels."""
    pd.DataFrame(
        [
            {
                "EIN2": " V-1 ",
                "split": "validation",
                "text": "coded validation",
                "human_label": 0,
            },
            {"EIN2": "T-1", "split": "test", "text": "held-out", "human_label": 1},
        ],
    ).to_csv(tiny_registry.gold_coding_template, index=False)

    split = load_human_split(tiny_config, tiny_registry, "validation")

    assert split.to_dict("records") == [
        {"EIN2": "V-1", "text": "coded validation", "human_label": 0},
    ]
