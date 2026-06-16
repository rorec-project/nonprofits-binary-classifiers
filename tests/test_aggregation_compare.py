"""Tests for stage 11 aggregation comparison reports."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import pytest

from binary_classifier.qc import aggregation_compare as compare_mod
from binary_classifier.qc.aggregation_compare import run_aggregation_compare


def test_aggregation_compare_report_schema_and_sensitivity_screen_true(
    monkeypatch: pytest.MonkeyPatch,
    tiny_config,
    tiny_registry,
) -> None:
    """A diagnostic arm can pass only when its CI lower bound clears majority F1."""
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
    assert report["metadata"]["report_purpose"] == "sensitivity_diagnostic"
    assert report["metadata"]["production_method"] == "majority"
    assert (
        "stage 04 intentionally freezes majority-vote"
        in report["metadata"]["production_note"]
    )
    assert set(report["arms"]) == {"majority", "crowdlab"}
    assert report["arms"]["crowdlab"]["n_scored"] == len(validation)
    assert report["arms"]["crowdlab"]["scored_ein2"] == validation["EIN2"].tolist()
    verdict = report["sensitivity"]["verdicts"]["crowdlab"]
    assert verdict["clears_majority_sensitivity_screen"] is True
    assert report["sensitivity"]["arms_clearing_screen"] == ["crowdlab"]
    assert report["sensitivity"]["best_diagnostic_arm"] == "crowdlab"


def test_aggregation_compare_sensitivity_false_when_ci_does_not_clear_majority(
    monkeypatch: pytest.MonkeyPatch,
    tiny_config,
    tiny_registry,
) -> None:
    """Equal or worse comparison arms do not pass the diagnostic screen."""
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
    verdict = report["sensitivity"]["verdicts"]["dawid_skene"]
    assert verdict["clears_majority_sensitivity_screen"] is False
    assert report["sensitivity"]["arms_clearing_screen"] == []
    assert report["sensitivity"]["best_diagnostic_arm"] is None
    assert "Diagnostic only" in report["sensitivity"]["message"]


def test_aggregation_compare_crowdlab_skips_when_oof_missing(
    tiny_config,
    tiny_registry,
) -> None:
    """Configured CROWDLAB reports a diagnostic skip without OOF probabilities."""
    tiny_config.aggregation.comparison_arms = ["crowdlab"]
    validation = _write_inputs(tiny_registry, write_oof=False)

    run_aggregation_compare(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.aggregation_compare.read_text())
    assert report["arms"]["majority"]["status"] == "scored"
    assert report["arms"]["majority"]["n_scored"] == len(validation)
    assert report["arms"]["crowdlab"]["status"] == "skipped"
    assert "requires OOF probabilities" in report["arms"]["crowdlab"]["skip_reason"]
    verdict = report["sensitivity"]["verdicts"]["crowdlab"]
    assert verdict["clears_majority_sensitivity_screen"] is False


def test_aggregation_compare_crowdlab_skips_when_oof_misses_validation(
    tiny_config,
    tiny_registry,
) -> None:
    """CROWDLAB skips when OOF probabilities do not cover validation EIN2s."""
    tiny_config.aggregation.comparison_arms = ["crowdlab"]
    validation = _write_inputs(
        tiny_registry,
        oof_ein2s=[f"T{i:03d}" for i in range(30)],
    )

    run_aggregation_compare(tiny_config, tiny_registry)

    report = json.loads(tiny_registry.aggregation_compare.read_text())
    assert report["arms"]["majority"]["status"] == "scored"
    assert report["arms"]["majority"]["n_scored"] == len(validation)
    skipped = report["arms"]["crowdlab"]
    assert skipped["status"] == "skipped"
    assert "do not cover all validation annotation EIN2s" in skipped["skip_reason"]
    assert skipped["n_pred_probs_validation_overlap"] == 0
    assert report["sensitivity"]["arms_clearing_screen"] == []


def _write_inputs(
    tiny_registry,
    *,
    write_oof: bool = True,
    oof_ein2s: list[str] | None = None,
) -> pd.DataFrame:
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
        prob_ein2s = oof_ein2s or ein2s
        pd.DataFrame(
            {
                "EIN2": [f" {ein2} " for ein2 in prob_ein2s],
                "fold": [0] * len(prob_ein2s),
                "p0": [1.0 - labels[idx % len(labels)] for idx in range(len(prob_ein2s))],
                "p1": [labels[idx % len(labels)] for idx in range(len(prob_ein2s))],
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
