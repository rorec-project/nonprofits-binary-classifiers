"""Tests for stage 11 aggregation comparison reports."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import pytest

from binary_classifier.qc import aggregation_compare as compare_mod
from binary_classifier.qc.aggregation_compare import run_aggregation_compare


def test_aggregation_compare_report_schema_and_adoption_true(
    monkeypatch: pytest.MonkeyPatch,
    tiny_config,
    tiny_registry,
) -> None:
    """A comparison arm can pass only when its CI lower bound clears majority F1."""
    tiny_config.aggregation.comparison_arms = ["crowdlab"]
    tiny_config.evaluation.bootstrap_resamples = 100
    validation = _write_inputs(tiny_registry)
    majority_pred = [1 if idx < 5 else 0 for idx in range(len(validation))]
    perfect_pred = validation["human_label"].astype(int).tolist()

    monkeypatch.setattr(
        compare_mod,
        "aggregate_labels",
        _fake_aggregator(
            validation["EIN2"].tolist(),
            {"majority": majority_pred, "crowdlab": perfect_pred},
            require_pred_probs_for={"crowdlab"},
        ),
    )

    run_aggregation_compare(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.aggregation_compare.read_text())
    assert report["metadata"]["stage"] == "11_aggregation_compare"
    assert report["metadata"]["production_method"] == "majority"
    assert "re-run stages 04→06" in report["metadata"]["rerun_required_if_adopted"]
    assert set(report["arms"]) == {"majority", "crowdlab"}
    assert report["arms"]["crowdlab"]["n_scored"] == len(validation)
    assert report["arms"]["crowdlab"]["scored_ein2"] == validation["EIN2"].tolist()
    verdict = report["adoption"]["verdicts"]["crowdlab"]
    assert verdict["may_replace_majority"] is True
    assert report["adoption"]["eligible_arms"] == ["crowdlab"]
    assert report["adoption"]["recommended_arm"] == "crowdlab"


def test_aggregation_compare_adoption_false_when_ci_does_not_clear_majority(
    monkeypatch: pytest.MonkeyPatch,
    tiny_config,
    tiny_registry,
) -> None:
    """Equal or worse comparison arms do not pass the replacement rule."""
    tiny_config.aggregation.comparison_arms = ["dawid_skene"]
    tiny_config.evaluation.bootstrap_resamples = 100
    validation = _write_inputs(tiny_registry)
    majority_pred = validation["human_label"].astype(int).tolist()
    worse_pred = [0 for _ in range(len(validation))]

    monkeypatch.setattr(
        compare_mod,
        "aggregate_labels",
        _fake_aggregator(
            validation["EIN2"].tolist(),
            {"majority": majority_pred, "dawid_skene": worse_pred},
        ),
    )

    run_aggregation_compare(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.aggregation_compare.read_text())
    verdict = report["adoption"]["verdicts"]["dawid_skene"]
    assert verdict["may_replace_majority"] is False
    assert report["adoption"]["eligible_arms"] == []
    assert report["adoption"]["recommended_arm"] is None
    assert "invalidates the frozen silver_labels.csv" in report["adoption"]["message"]


def test_aggregation_compare_crowdlab_requires_oof_probs(
    tiny_config,
    tiny_registry,
) -> None:
    """Configured CROWDLAB comparisons fail fast without PR-2 OOF probabilities."""
    tiny_config.aggregation.comparison_arms = ["crowdlab"]
    _write_inputs(tiny_registry, write_oof=False)

    with pytest.raises(FileNotFoundError, match="requires OOF probabilities"):
        run_aggregation_compare(tiny_config, tiny_registry)


def _write_inputs(tiny_registry, *, write_oof: bool = True) -> pd.DataFrame:
    ein2s = [f"V{i:03d}" for i in range(30)]
    labels = [1 if i < 10 else 0 for i in range(30)]
    store_rows = []
    for ein2 in ein2s:
        for source_idx in range(2):
            store_rows.append(
                {
                    "EIN2": ein2,
                    "source_id": f"m{source_idx}:p0",
                    "label": 1.0,
                    "confidence": 0.8,
                },
            )
    pd.DataFrame(store_rows).to_csv(tiny_registry.annotation_store, index=False)

    validation = pd.DataFrame(
        {
            "EIN2": ein2s,
            "split": ["validation"] * len(ein2s),
            "text": [f"validation row {i}" for i in range(len(ein2s))],
            "human_label": labels,
        },
    )
    validation.to_csv(tiny_registry.gold_coding_template, index=False)

    if write_oof:
        pd.DataFrame(
            {
                "EIN2": [f" {ein2} " for ein2 in ein2s],
                "fold": [0] * len(ein2s),
                "p0": [1.0 - label for label in labels],
                "p1": labels,
            },
        ).to_parquet(tiny_registry.oof_pred_probs, index=False)
    return validation[["EIN2", "human_label"]]


def _fake_aggregator(
    ein2s: list[str],
    predictions_by_method: dict[str, list[int]],
    *,
    require_pred_probs_for: set[str] | None = None,
) -> Callable[..., pd.DataFrame]:
    required = require_pred_probs_for or set()

    def aggregate_labels(
        df: pd.DataFrame,
        method: str = "majority",
        pred_probs: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        assert not df.empty
        if method in required:
            assert pred_probs is not None
            assert pred_probs["EIN2"].tolist() == ein2s
        predictions = predictions_by_method[method]
        return pd.DataFrame(
            {
                "EIN2": ein2s,
                "silver_label": predictions,
                "silver_confidence": [0.9] * len(ein2s),
                "num_votes": [2] * len(ein2s),
                "num_abstain": [0] * len(ein2s),
                "agreement": [1.0] * len(ein2s),
                "tie": [False] * len(ein2s),
            },
        )

    return aggregate_labels
