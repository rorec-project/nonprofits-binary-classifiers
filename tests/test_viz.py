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
    ngram_log_odds,
    pr_curve,
    prevalence_forest,
    production_annotation_summary as exported_production_annotation_summary,
    reliability_diagram,
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
