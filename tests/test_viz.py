"""Rendering tests for pure visualization helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from binary_classifier.viz import (
    documentation_curve,
    ngram_log_odds,
    pr_curve,
    prevalence_forest,
    reliability_diagram,
)


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
