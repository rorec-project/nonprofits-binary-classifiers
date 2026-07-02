"""Rendering tests for pure visualization helpers."""

import json

import matplotlib

matplotlib.use("Agg")

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt

import pandas as pd
import pytest

from binary_classifier.viz.bakeoff import (
    bakeoff_summary,
    canary_drift,
    production_annotation_summary,
)

from binary_classifier.viz import (
    bakeoff_summary as exported_bakeoff_summary,
    canary_drift as exported_canary_drift,
    documentation_curve,
    frozen_test_confusion_matrices,
    frozen_test_curves,
    ngram_log_odds,
    pr_curve,
    prevalence_decomposition,
    prevalence_forest,
    quantification_sensitivity,
    production_annotation_summary as exported_production_annotation_summary,
    reliability_diagram,
    rule_validation_intervals,
    score_distribution_by_tier_label,
    subgroup_performance,
)
from binary_classifier.viz.style import (
    OKABE_ITO_ORANGE,
    PAPER_RCPARAMS,
    style_context,
)


def _load_visualize_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "10_visualize.py"
    spec = importlib.util.spec_from_file_location("stage10_visualize_for_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ngram_log_odds_renders_tmp_png(tmp_path):
    rows = []
    for idx in range(8):
        rows.append(
            {
                "EIN2": f"P{idx}",
                "mission_text": "church worship parish community aid",
                "silver_label": 1,
            }
        )
        rows.append(
            {
                "EIN2": f"N{idx}",
                "mission_text": "clinic health science community aid",
                "silver_label": 0,
            }
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        ngram_log_odds(pd.DataFrame(rows), ax, top_k=8)
        out = tmp_path / "ngram_log_odds.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_documentation_curve_renders_tmp_png(tmp_path):
    rows = [
        {
            "encoder": "small",
            "train_fraction": 0.25,
            "seed": 1,
            "validation": {"pr_auc": 0.62},
        },
        {
            "encoder": "small",
            "train_fraction": 0.25,
            "seed": 2,
            "validation": {"pr_auc": 0.66},
        },
        {
            "encoder": "small",
            "train_fraction": 1.0,
            "seed": 1,
            "validation": {"pr_auc": 0.78},
        },
        {
            "encoder": "large",
            "train_fraction": 0.25,
            "seed": 1,
            "validation": {"pr_auc": 0.70},
        },
        {
            "encoder": "large",
            "train_fraction": 1.0,
            "seed": 1,
            "validation": {"pr_auc": 0.84},
        },
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        documentation_curve(rows, ax)
        out = tmp_path / "documentation_curve.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_pr_curve_renders_tmp_png(tmp_path):
    points = {
        "pr_curve_points": [
            {"threshold": 0.2, "precision": 0.50, "recall": 1.00},
            {"threshold": 0.5, "precision": 0.75, "recall": 0.75},
            {"threshold": 0.8, "precision": 1.00, "recall": 0.25},
        ]
    }
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        pr_curve(points, ax)
        out = tmp_path / "pr_curve.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_frozen_test_curves_require_test_scores_and_render(tmp_path):
    payload = _test_evaluation_payload()
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        frozen_test_curves(payload, ax)
        out = tmp_path / "frozen_test_curves.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        with pytest.raises(ValueError, match="test_scores"):
            frozen_test_curves({"pr_curve_points": []}, ax)
    finally:
        plt.close(fig)


def test_score_distribution_by_tier_label_renders_tmp_png(tmp_path):
    predictions = pd.DataFrame(
        {
            "prob_calibrated": [0.02, 0.08, 0.15, 0.62, 0.72, 0.91],
            "tier": ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW"],
            "pred_label": [0, 1, 1, 1, 1, 1],
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        score_distribution_by_tier_label(
            predictions,
            {"operating": 0.05, "max_f1": 0.60, "base_rate": 0.70},
            ax,
        )
        out = tmp_path / "score_distribution.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_frozen_test_confusion_matrices_render_tmp_png(tmp_path):
    payload = _test_evaluation_payload()
    fig, ax = plt.subplots(figsize=(7, 3))
    try:
        frozen_test_confusion_matrices(payload, ax)
        out = tmp_path / "confusion_matrices.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_reliability_diagram_renders_tmp_png(tmp_path):
    payload = {
        "ece": 0.07,
        "reliability_curve": [
            {
                "bin": 0,
                "lower": 0.0,
                "upper": 0.5,
                "count": 4,
                "mean_predicted": 0.20,
                "observed_fraction": 0.25,
                "gap": -0.05,
            },
            {
                "bin": 1,
                "lower": 0.5,
                "upper": 1.0,
                "count": 6,
                "mean_predicted": 0.78,
                "observed_fraction": 0.67,
                "gap": 0.11,
            },
        ],
    }
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        reliability_diagram(payload, ax)
        out = tmp_path / "reliability_diagram.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_bakeoff_summary_renders_synthetic_fixture(tmp_path):
    results = [
        _bakeoff_result("model-a", "v1", f1=0.72, lower=0.60, upper=0.83, kappa=0.68),
        _bakeoff_result(
            "model-b",
            "v2",
            f1=0.90,
            lower=0.75,
            upper=0.97,
            kappa=0.82,
            abstain_rate=0.33,
            n_valid=8,
        ),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        bakeoff_summary(results, ax)
        out = tmp_path / "bakeoff_summary.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_bakeoff_summary_renders_real_results():
    path = Path(__file__).resolve().parents[1] / "data/interim/bakeoff/bakeoff_results.json"
    if not path.exists():
        pytest.skip("local bakeoff_results.json artifact is absent")
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        bakeoff_summary(json.loads(path.read_text()), ax)
        assert ax.get_title()
    finally:
        plt.close(fig)


def test_production_annotation_summary_renders_synthetic_fixture(tmp_path):
    frame = pd.DataFrame(
        [
            {"EIN2": "1", "source_id": "m1__v1", "label": 1.0},
            {"EIN2": "1", "source_id": "m2__v1", "label": 1.0},
            {"EIN2": "2", "source_id": "m1__v1", "label": 0.0},
            {"EIN2": "2", "source_id": "m2__v1", "label": 1.0},
            {"EIN2": "3", "source_id": "m1__v1", "label": float("nan")},
            {"EIN2": "3", "source_id": "m2__v1", "label": 0.0},
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        production_annotation_summary(frame, ax)
        out = tmp_path / "production_annotation_summary.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_canary_drift_renders_synthetic_fixture(tmp_path):
    rows = [
        {
            "run_timestamp": "2026-06-21T10:00:00+00:00",
            "kappa_alpha_change_test": {
                "status": "baseline",
                "changed": False,
                "n_common": 0,
                "n_changed": 0,
                "cohens_kappa": None,
                "krippendorff_alpha": None,
            },
        },
        {
            "run_timestamp": "2026-06-22T10:00:00+00:00",
            "kappa_alpha_change_test": {
                "status": "compared",
                "changed": True,
                "n_common": 10,
                "n_changed": 2,
                "cohens_kappa": 0.62,
                "krippendorff_alpha": 0.60,
            },
        },
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        canary_drift(rows, ax)
        out = tmp_path / "canary_drift.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("plotter", "empty"),
    [
        (bakeoff_summary, []),
        (production_annotation_summary, pd.DataFrame(columns=["EIN2", "source_id", "label"])),
        (canary_drift, []),
    ],
)
def test_bakeoff_viz_helpers_raise_on_empty_input(plotter, empty):
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        with pytest.raises(ValueError):
            plotter(empty, ax)
    finally:
        plt.close(fig)


def test_prevalence_forest_renders_tmp_png(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "ntee_major_group": "A",
                "n_anchor": 40,
                "estimator": "ppi_rg_composite",
                "estimate": 0.18,
                "ci_lower": 0.11,
                "ci_upper": 0.25,
                "suppressed": False,
            },
            {
                "ntee_major_group": "B",
                "n_anchor": 3,
                "estimator": "emq_fallback",
                "estimate": 0.07,
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "suppressed": True,
            },
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        prevalence_forest(frame, ax)
        out = tmp_path / "prevalence_forest.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_prevalence_wave4_helpers_render_tmp_png(tmp_path):
    prevalence = _prevalence_report_payload()
    rule_validation = _rule_validation_payload()
    for name, plotter, payload in (
        ("prevalence_decomposition", prevalence_decomposition, prevalence),
        ("rule_validation_intervals", rule_validation_intervals, rule_validation),
        ("quantification_sensitivity", quantification_sensitivity, prevalence),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            plotter(payload, ax)
            out = tmp_path / f"{name}.png"
            fig.savefig(out)
            assert out.stat().st_size > 0
        finally:
            plt.close(fig)


def test_subgroup_performance_renders_tmp_png(tmp_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        subgroup_performance(_test_evaluation_payload()["subgroups"], ax)
        out = tmp_path / "subgroup_performance.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def _bakeoff_result(
    model_id,
    prompt_id,
    *,
    f1,
    lower,
    upper,
    kappa,
    abstain_rate=0.1,
    n_valid=18,
):
    return {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "source_id": f"{model_id}__{prompt_id}",
        "scores": {
            "f1": f1,
            "abstain_rate": abstain_rate,
            "n_valid": n_valid,
            "metrics": {
                "cohens_kappa": kappa,
                "bootstrap_ci": {
                    "minority_f1": {"lower": lower, "upper": upper},
                },
            },
        },
    }


def test_paper_style_context_applies_okabe_ito_cycle():
    with style_context():
        first_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        assert first_color == OKABE_ITO_ORANGE
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["pdf.fonttype"] == 42
        assert plt.rcParams["svg.fonttype"] == "none"


def test_paper_mplstyle_parses():
    with plt.rc_context(PAPER_RCPARAMS):
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["savefig.dpi"] == 300


def test_save_plot_emits_pdf_svg_and_png(tmp_path):
    visualize = _load_visualize_module()
    registry = SimpleNamespace(figures_dir=tmp_path)

    visualize._save_plot(
        registry,
        "paper_style_smoke",
        lambda ax: ax.plot([0, 1], [0, 1]),
        figsize=(5.5, 3.0),
    )

    for suffix in (".pdf", ".svg", ".png"):
        out = tmp_path / f"paper_style_smoke{suffix}"
        assert out.exists()
        assert out.stat().st_size > 0


def test_production_summary_sizes_by_diagnostic_rows_not_raw_rows():
    visualize = _load_visualize_module()
    frame = pd.DataFrame(
        [
            {"EIN2": f"E{idx:03d}", "source_id": "m1__v1", "label": idx % 2}
            for idx in range(100)
        ]
    )
    sizing = visualize._production_summary_sizing_frame(frame)
    assert len(sizing) == 4


def test_bakeoff_helpers_exported_from_viz_package():
    assert exported_bakeoff_summary is bakeoff_summary
    assert exported_production_annotation_summary is production_annotation_summary
    assert exported_canary_drift is canary_drift


def test_visualize_new_wrappers_render_and_skip(tmp_path, caplog):
    visualize = _load_visualize_module()
    bakeoff_results = tmp_path / "bakeoff_results.json"
    figures_dir = tmp_path / "figures"
    annotation_store = tmp_path / "annotation_store.csv"
    silver_labels = tmp_path / "silver_labels.csv"
    interim_dir = tmp_path / "interim"
    registry = SimpleNamespace(
        bakeoff_results=bakeoff_results,
        figures_dir=figures_dir,
        annotation_store=annotation_store,
        silver_labels=silver_labels,
        interim_dir=interim_dir,
    )
    bakeoff_results.write_text(
        json.dumps([_bakeoff_result("model-a", "v1", f1=0.72, lower=0.60, upper=0.83, kappa=0.68)]),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"EIN2": "1", "source_id": "m1__v1", "label": 1.0},
            {"EIN2": "1", "source_id": "m2__v1", "label": 1.0},
            {"EIN2": "2", "source_id": "m1__v1", "label": 0.0},
            {"EIN2": "2", "source_id": "m2__v1", "label": float("nan")},
        ]
    ).to_csv(annotation_store, index=False)

    with caplog.at_level("WARNING"):
        assert visualize._maybe_render_bakeoff_summary(None, registry)
        assert visualize._maybe_render_production_summary(None, registry)
        assert not visualize._maybe_render_canary_drift(None, registry)

    for name in ("bakeoff_summary", "production_annotation_summary"):
        for suffix in (".pdf", ".svg", ".png"):
            assert (figures_dir / f"{name}{suffix}").exists()
    assert "Skipping canary drift; missing input" in caplog.text


def test_visualize_wave4_wrappers_render_and_skip(tmp_path, caplog):
    visualize = _load_visualize_module()
    figures_dir = tmp_path / "figures"
    test_evaluation = tmp_path / "test_evaluation.json"
    predictions = tmp_path / "predictions.parquet"
    calibrator = tmp_path / "calibrator.json"
    base_rate = tmp_path / "base_rate_precision.json"
    prevalence = tmp_path / "prevalence_report.json"
    rule_validation = tmp_path / "rule_validation.json"
    registry = SimpleNamespace(
        figures_dir=figures_dir,
        test_evaluation=test_evaluation,
        predictions_parquet=predictions,
        calibrator_path=calibrator,
        base_rate_precision=base_rate,
        prevalence_report=prevalence,
        rule_validation=rule_validation,
    )
    test_evaluation.write_text(json.dumps(_test_evaluation_payload()), encoding="utf-8")
    pd.DataFrame(
        {
            "prob_calibrated": [0.01, 0.07, 0.65, 0.75],
            "tier": ["HIGH", "HIGH", "LOW", "LOW"],
            "pred_label": [0, 1, 1, 1],
        }
    ).to_parquet(predictions)
    calibrator.write_text(
        json.dumps({"threshold": 0.05, "max_f1_threshold": 0.60}), encoding="utf-8"
    )
    base_rate.write_text(json.dumps({"threshold": 0.70}), encoding="utf-8")
    prevalence.write_text(json.dumps(_prevalence_report_payload()), encoding="utf-8")
    rule_validation.write_text(json.dumps(_rule_validation_payload()), encoding="utf-8")

    assert visualize._maybe_render_pr_curve(None, registry)
    assert visualize._maybe_render_frozen_test_confusion_matrices(None, registry)
    assert visualize._maybe_render_score_distribution(None, registry)
    assert visualize._maybe_render_prevalence_decomposition(None, registry)
    assert visualize._maybe_render_rule_validation_intervals(None, registry)
    assert visualize._maybe_render_quantification_sensitivity(None, registry)
    assert visualize._maybe_render_subgroup_performance(None, registry)

    for name in (
        "precision_recall_curve",
        "frozen_test_confusion_matrices",
        "score_distribution_by_tier_label",
        "prevalence_decomposition",
        "rule_validation_intervals",
        "quantification_sensitivity",
        "subgroup_performance",
    ):
        assert (figures_dir / f"{name}.png").exists()

    test_evaluation.write_text(json.dumps({"metric_bundle": {}}), encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert not visualize._maybe_render_pr_curve(None, registry)
    assert "lacks frozen-test test_scores" in caplog.text


def _test_evaluation_payload():
    return {
        "test_scores": {
            "1": {"y_true": 0, "prob_calibrated": 0.02},
            "2": {"y_true": 1, "prob_calibrated": 0.80},
        },
        "pr_curve_points": [
            {"threshold": 0.01, "precision": 0.50, "recall": 1.00},
            {"threshold": 0.05, "precision": 0.67, "recall": 1.00},
            {"threshold": 0.60, "precision": 1.00, "recall": 0.50},
        ],
        "roc_curve_points": [
            {"threshold": 1.0, "fpr": 0.0, "tpr": 0.0},
            {"threshold": 0.60, "fpr": 0.0, "tpr": 0.5},
            {"threshold": 0.05, "fpr": 0.5, "tpr": 1.0},
        ],
        "confusion_matrices": {
            "operating": {"threshold": 0.05, "confusion_matrix": {"tn": 2, "fp": 1, "fn": 0, "tp": 3}},
            "max_f1": {"threshold": 0.60, "confusion_matrix": {"tn": 3, "fp": 0, "fn": 1, "tp": 2}},
            "base_rate": {"threshold": 0.70, "confusion_matrix": {"tn": 3, "fp": 0, "fn": 2, "tp": 1}},
        },
        "subgroups": [
            {"grouping": "ntee_major_group", "value": "A", "n": 3, "suppressed": False, "minority_f1": 0.8, "fpr": 0.1, "fnr": 0.2},
            {"grouping": "data_source", "value": "gold", "n": 3, "suppressed": False, "minority_f1": 0.7, "fpr": 0.2, "fnr": 0.1},
            {"grouping": "word_count_bin", "value": "0-10", "n": 3, "suppressed": False, "minority_f1": 0.9, "fpr": 0.0, "fnr": 0.1},
        ],
    }


def _prevalence_report_payload():
    return {
        "tier_shares": {"HIGH_MEDIUM": 0.8, "LOW": 0.2},
        "hm": {
            "primary": "weighted_ppi",
            "weighted_ppi": {"estimate": 0.15, "ci_lower": 0.10, "ci_upper": 0.20},
            "unweighted_ppi": {"estimate": 0.13, "ci_lower": 0.09, "ci_upper": 0.18},
            "sensitivity_anchor_residual_multiplicity_weighted": {"estimate": 0.16, "ci_lower": 0.11, "ci_upper": 0.21},
        },
        "low": {
            "sub_strata": {
                "low_via_classifier": {"share": 0.4, "estimate": {"estimate": 0.20, "ci_lower": 0.12, "ci_upper": 0.30}},
                "rule": {"share": 0.6, "estimate": {"estimate": 0.05, "ci_lower": 0.02, "ci_upper": 0.10}},
            }
        },
        "cross_checks": {"emq": {"status": "ok", "estimate": 0.14}},
    }


def _rule_validation_payload():
    return {
        "metrics": {
            "sensitivity": {"value": 0.90, "ci_lower": 0.75, "ci_upper": 0.98, "n": 30},
            "specificity": {"value": 0.85, "ci_lower": 0.70, "ci_upper": 0.95, "n": 40},
        }
    }
