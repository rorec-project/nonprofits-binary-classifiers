"""Curve plots for training, evaluation, and calibration artifacts."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from binary_classifier.viz.style import (
    LIGHT_GREY,
    MUTED_GREY,
    OKABE_ITO_BLACK,
    OKABE_ITO_BLUE,
    OKABE_ITO_BLUISH_GREEN,
    OKABE_ITO_ORANGE,
    OKABE_ITO_VERMILLION,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def documentation_curve(
    results_jsonl_rows: Iterable[Mapping[str, Any]],
    ax: Axes,
) -> None:
    """Plot validation PR-AUC against training fraction by encoder.

    Args:
        results_jsonl_rows: Iterable of JSONL-like training-result rows. Rows may
            store PR-AUC at ``row["validation"]["pr_auc"]`` or common flattened
            metric keys. Encoder/model, train fraction, and seed are detected
            from top-level or nested dictionaries.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If no complete rows can be extracted.

    """
    extracted = [_documentation_row(row) for row in results_jsonl_rows]
    rows = [row for row in extracted if row is not None]
    if not rows:
        raise ValueError("No documentation-curve rows with encoder/fraction/PR-AUC.")

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["encoder", "train_fraction"], as_index=False)["pr_auc"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .sort_values(["encoder", "train_fraction"])
    )

    for encoder, group in summary.groupby("encoder", sort=True):
        xs = group["train_fraction"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        ax.plot(xs, mean, marker="o", label=str(encoder))
        if (group["count"].to_numpy(dtype=int) > 1).any():
            lower = group["min"].to_numpy(dtype=float)
            upper = group["max"].to_numpy(dtype=float)
            ax.fill_between(xs, lower, upper, alpha=0.18)

    ax.set_xlabel("Training fraction")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Documentation curve")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Encoder")
    logger.info("Rendered documentation curve for %d rows", len(rows))


def pr_curve(points: object, ax: Axes) -> None:
    """Plot a precision-recall curve from serialized evaluation points.

    Args:
        points: Point list or containing dictionary with ``precision`` and
            ``recall`` values, such as ``threshold_report["pr_curve_points"]``.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If no valid precision-recall points are present.

    """
    frame = _points_frame(points, required=("recall", "precision"))
    frame = frame.sort_values("recall")
    ax.plot(
        frame["recall"].to_numpy(dtype=float),
        frame["precision"].to_numpy(dtype=float),
        marker="o",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    logger.info("Rendered PR curve with %d points", len(frame))


def frozen_test_curves(payload: Mapping[str, Any], ax: Axes) -> None:
    """Plot frozen-test PR and ROC curves from persisted test-score artifacts."""
    if not isinstance(payload.get("test_scores"), Mapping):
        raise ValueError("test_evaluation payload missing test_scores block.")
    pr_frame = _points_frame(
        payload.get("pr_curve_points"),
        required=("recall", "precision"),
    ).sort_values("recall")
    roc_frame = _points_frame(
        payload.get("roc_curve_points"),
        required=("fpr", "tpr"),
        containers=("roc_curve_points", "points"),
    ).sort_values("fpr")

    ax.plot(pr_frame["recall"], pr_frame["precision"], color=OKABE_ITO_BLUE, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Frozen-test precision-recall curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    _annotate_threshold_points(ax, pr_frame, payload)
    ax.legend(loc="lower left")

    inset = ax.inset_axes((0.56, 0.12, 0.40, 0.40))
    inset.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=MUTED_GREY,
        linestyle="--",
        linewidth=0.8,
        label="Random",
    )
    inset.plot(roc_frame["fpr"], roc_frame["tpr"], color=OKABE_ITO_ORANGE, label="ROC")
    inset.set_xlabel("FPR", fontsize=7)
    inset.set_ylabel("TPR", fontsize=7)
    inset.set_title("ROC", fontsize=8)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.grid(alpha=0.20)
    inset.legend(loc="lower right", fontsize=6)
    logger.info("Rendered frozen-test curves with %d PR points", len(pr_frame))


def score_distribution_by_tier_label(
    predictions: pd.DataFrame,
    thresholds: Mapping[str, float],
    ax: Axes,
) -> None:
    """Plot calibrated-score distributions by quality tier and predicted label."""
    required = {"prob_calibrated", "tier", "pred_label"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["prob_calibrated"] = pd.to_numeric(frame["prob_calibrated"], errors="coerce")
    frame = frame.dropna(subset=["prob_calibrated", "tier", "pred_label"])
    if frame.empty:
        raise ValueError("No finite prediction scores to plot.")

    tiers = sorted(frame["tier"].astype(str).str.upper().unique())
    colors = {0: OKABE_ITO_BLUE, 1: OKABE_ITO_ORANGE}
    bins = np.linspace(0.0, 1.0, 31)
    low_band, high_band = _threshold_band(thresholds)
    fig = ax.figure
    ax.remove()
    axes = fig.subplots(len(tiers), 1, sharex=True, squeeze=False).ravel()
    for tier, tier_ax in zip(tiers, axes, strict=True):
        group = frame.loc[frame["tier"].astype(str).str.upper() == tier]
        if low_band < high_band:
            tier_ax.axvspan(
                low_band,
                high_band,
                color=LIGHT_GREY,
                alpha=0.25,
                linewidth=0,
            )
        for label_value, label_group in group.groupby("pred_label", sort=True):
            numeric_label = int(float(label_value))
            tier_ax.hist(
                label_group["prob_calibrated"],
                bins=bins,
                histtype="stepfilled",
                alpha=0.32,
                density=True,
                color=colors.get(numeric_label, MUTED_GREY),
                label=f"pred_label={numeric_label}",
            )
        for name, threshold in thresholds.items():
            tier_ax.axvline(
                threshold,
                color=_threshold_color(name),
                linestyle=_threshold_linestyle(name),
                linewidth=0.9,
            )
        tier_ax.set_ylabel(tier)
        tier_ax.grid(axis="x", alpha=0.20)
    axes[0].set_title("Calibrated-score distribution by tier and label")
    axes[-1].set_xlabel("Calibrated probability")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right")
    logger.info("Rendered score distributions for %d tiers", len(tiers))


def draw_single_confusion_matrix(
    item: Mapping[str, Any],
    name: str,
    ax: Axes,
) -> None:
    """Render one confusion matrix on the given axes."""
    counts = item.get("confusion_matrix", item)
    if not isinstance(counts, Mapping):
        raise ValueError(f"confusion matrix {name!r} missing counts.")
    values = np.array(
        [
            [int(counts.get("tn", 0)), int(counts.get("fp", 0))],
            [int(counts.get("fn", 0)), int(counts.get("tp", 0))],
        ],
    )
    ax.imshow(values, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(values[row, col]),
                ha="center",
                va="center",
            )
    threshold = item.get("threshold")
    suffix = "" if threshold is None else f"\nthr={float(threshold):.3f}"
    ax.set_title(f"{name.replace('_', ' ').title()}{suffix}")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])


def frozen_test_confusion_matrices(payload: Mapping[str, Any], ax: Axes) -> None:
    """Render frozen-test confusion matrices at persisted operating thresholds."""
    matrices = payload.get("confusion_matrices")
    if not isinstance(matrices, Mapping):
        raise ValueError("test_evaluation payload missing confusion_matrices block.")
    names = [name for name in ("operating", "max_f1", "base_rate") if name in matrices]
    if not names:
        raise ValueError("No operating/max_f1/base_rate confusion matrices found.")

    fig = ax.figure
    ax.remove()
    axes = fig.subplots(1, len(names), squeeze=False).ravel()
    for name, matrix_ax in zip(names, axes, strict=True):
        draw_single_confusion_matrix(matrices[name], name, matrix_ax)
    logger.info("Rendered %d frozen-test confusion matrices", len(names))


def subgroup_performance(subgroups: object, ax: Axes) -> None:
    """Plot subgroup F1/FPR/FNR dot diagnostics from ``test_evaluation``."""
    frame = pd.DataFrame(_extract_points(subgroups, ("subgroups", "points")))
    if frame.empty:
        raise ValueError("No subgroup rows to plot.")
    required = {"grouping", "value", "n", "minority_f1", "fpr", "fnr", "suppressed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"subgroup rows missing columns: {sorted(missing)}")
    frame = frame.loc[~frame["suppressed"].astype(bool)].copy()
    if frame.empty:
        raise ValueError("All subgroup rows are suppressed.")
    for metric in ("minority_f1", "fpr", "fnr"):
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["minority_f1", "fpr", "fnr"])
    if frame.empty:
        raise ValueError("No finite subgroup metrics to plot.")
    frame["label"] = frame.apply(
        lambda row: f"{row['grouping']}={row['value']} (n={int(row['n'])})",
        axis=1,
    )
    frame = frame.sort_values(["grouping", "value"]).reset_index(drop=True)
    y = np.arange(len(frame), dtype=float)
    for metric, color, marker, label in (
        ("minority_f1", OKABE_ITO_BLUE, "o", "Minority F1"),
        ("fpr", OKABE_ITO_ORANGE, "s", "FPR"),
        ("fnr", OKABE_ITO_BLUISH_GREEN, "D", "FNR"),
    ):
        ax.scatter(
            frame[metric],
            y,
            label=label,
            color=color,
            marker=marker,
        )
    ax.set_yticks(y, labels=frame["label"].to_list())
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Metric value")
    ax.set_ylabel("Subgroup")
    ax.set_title("Frozen-test subgroup performance")
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=0.25)
    logger.info("Rendered subgroup performance for %d rows", len(frame))


def reliability_diagram(points: object, ax: Axes, ece: float | None = None) -> None:
    """Plot a reliability diagram from serialized calibration-bin points.

    Args:
        points: Point list or containing dictionary with calibration bins. A
            containing dictionary may provide ``reliability_curve`` and ``ece``.
        ax: Matplotlib axes to draw into.
        ece: Optional expected calibration error for plot annotation.

    Raises:
        ValueError: If no non-empty reliability points are present.

    """
    if ece is None and isinstance(points, Mapping):
        points_mapping = cast(Mapping[str, Any], points)
        raw_ece = points_mapping.get("ece")
        if raw_ece is not None:
            ece = float(raw_ece)
    frame = _points_frame(
        points,
        required=("mean_predicted", "observed_fraction"),
        containers=("reliability_curve", "calibration_curve", "points"),
    )
    frame = frame.dropna(subset=["mean_predicted", "observed_fraction"])
    if frame.empty:
        raise ValueError("No non-empty reliability bins to plot.")

    counts = frame["count"].to_numpy(dtype=float) if "count" in frame else None
    marker_sizes = None
    if counts is not None and counts.max(initial=0.0) > 0.0:
        marker_sizes = 35.0 + 165.0 * counts / counts.max()

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color=OKABE_ITO_BLACK,
        linewidth=0.9,
        label="Perfect calibration",
    )
    ax.scatter(
        frame["mean_predicted"].to_numpy(dtype=float),
        frame["observed_fraction"].to_numpy(dtype=float),
        s=marker_sizes,
        color=OKABE_ITO_BLUE,
        alpha=0.8,
        label="Observed bins",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    title = "Reliability diagram"
    if ece is not None:
        title = f"{title} (ECE={ece:.3f})"
    ax.set_title(title)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    logger.info("Rendered reliability diagram with %d bins", len(frame))


def _annotate_threshold_points(
    ax: Axes,
    pr_frame: pd.DataFrame,
    payload: Mapping[str, Any],
) -> None:
    thresholds = _thresholds_from_payload(payload)
    for name, threshold in thresholds.items():
        point = _nearest_threshold_point(pr_frame, threshold)
        ax.scatter(
            [point["recall"]],
            [point["precision"]],
            s=28,
            color=_threshold_color(name),
            marker="o",
            zorder=3,
            label=name.replace("_", " "),
        )
        ax.annotate(
            name.replace("_", " "),
            (point["recall"], point["precision"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    if thresholds:
        ax.legend(loc="lower left")


def _thresholds_from_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    matrices = payload.get("confusion_matrices")
    if isinstance(matrices, Mapping):
        for name, item in matrices.items():
            if name in {"operating", "max_f1", "base_rate"} and isinstance(
                item,
                Mapping,
            ):
                threshold = item.get("threshold")
                if threshold is not None:
                    thresholds[str(name)] = float(threshold)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        calibrator = metadata.get("calibrator")
        if isinstance(calibrator, Mapping):
            if "threshold" in calibrator:
                thresholds.setdefault("operating", float(calibrator["threshold"]))
            if "max_f1_threshold" in calibrator:
                thresholds.setdefault("max_f1", float(calibrator["max_f1_threshold"]))
        base_rate = metadata.get("base_rate_precision")
        if isinstance(base_rate, Mapping) and "threshold" in base_rate:
            thresholds.setdefault("base_rate", float(base_rate["threshold"]))
    return thresholds


def _nearest_threshold_point(frame: pd.DataFrame, threshold: float) -> pd.Series:
    if "threshold" not in frame.columns:
        return frame.iloc[(frame["recall"] - 1.0).abs().argmin()]
    thresholds = pd.to_numeric(frame["threshold"], errors="coerce")
    if thresholds.notna().any():
        return frame.loc[(thresholds - threshold).abs().idxmin()]
    return frame.iloc[(frame["recall"] - 1.0).abs().argmin()]


def _threshold_band(thresholds: Mapping[str, float]) -> tuple[float, float]:
    values = [
        float(value) for value in thresholds.values() if np.isfinite(float(value))
    ]
    if len(values) < 2:
        return (0.0, 0.0)
    return (min(values), max(values))


def _threshold_color(name: str) -> str:
    if name == "operating":
        return OKABE_ITO_VERMILLION
    if name == "max_f1":
        return OKABE_ITO_BLACK
    if name == "base_rate":
        return OKABE_ITO_BLUISH_GREEN
    return MUTED_GREY


def _threshold_linestyle(name: str) -> str:
    if name == "operating":
        return "--"
    if name == "max_f1":
        return ":"
    return "-."


def _documentation_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    encoder = _first_present(
        row,
        ("encoder", "model", "model_name", "encoder_model", "base_model"),
    )
    if encoder is None:
        for container in ("config", "params", "run", "arm"):
            nested = row.get(container)
            if isinstance(nested, Mapping):
                nested_mapping = cast(Mapping[str, Any], nested)
                encoder = _first_present(
                    nested_mapping,
                    ("encoder", "model", "model_name", "encoder_model", "base_model"),
                )
                if encoder is not None:
                    break
    fraction = _first_present(row, ("train_fraction", "fraction", "train_frac"))
    if fraction is None:
        train = row.get("train")
        if isinstance(train, Mapping):
            fraction = _first_present(
                cast(Mapping[str, Any], train),
                ("fraction", "train_fraction"),
            )
    pr_auc = _validation_pr_auc(row)
    if encoder is None or fraction is None or pr_auc is None:
        return None
    return {
        "encoder": str(encoder),
        "train_fraction": float(fraction),
        "pr_auc": float(pr_auc),
        "seed": _first_present(row, ("seed", "random_seed")),
    }


def _validation_pr_auc(row: Mapping[str, Any]) -> float | None:
    validation = row.get("validation")
    if isinstance(validation, Mapping):
        validation_mapping = cast(Mapping[str, Any], validation)
        value = _first_present(
            validation_mapping,
            ("pr_auc", "average_precision", "auprc"),
        )
        if value is not None:
            return float(value)
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        metrics_mapping = cast(Mapping[str, Any], metrics)
        value = _first_present(
            metrics_mapping,
            ("validation_pr_auc", "val_pr_auc", "pr_auc", "average_precision"),
        )
        if value is not None:
            return float(value)
    value = _first_present(row, ("validation_pr_auc", "val_pr_auc", "pr_auc"))
    return None if value is None else float(value)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _points_frame(
    points: object,
    *,
    required: tuple[str, ...],
    containers: tuple[str, ...] = ("pr_curve_points", "points"),
) -> pd.DataFrame:
    point_rows = _extract_points(points, containers)
    frame = pd.DataFrame(point_rows)
    if frame.empty:
        raise ValueError("No serialized points to plot.")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Serialized points missing columns: {missing}.")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        raise ValueError("No finite serialized points to plot.")
    return frame


def _extract_points(
    points: object,
    containers: tuple[str, ...],
) -> list[Mapping[str, Any]]:
    if isinstance(points, Mapping):
        points_mapping = cast(Mapping[str, Any], points)
        for key in containers:
            nested = points_mapping.get(key)
            if nested is not None:
                return _extract_points(nested, containers)
        return [points_mapping]
    if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        rows: list[Mapping[str, Any]] = []
        for point in points:
            if not isinstance(point, Mapping):
                raise ValueError("Serialized points must be mappings.")
            rows.append(cast(Mapping[str, Any], point))
        return rows
    raise ValueError("Serialized points must be a mapping or sequence of mappings.")


def threshold_sweep_plot(
    predictions: pd.DataFrame,
    thresholds: Mapping[str, float],
    anchor_oof: pd.DataFrame,
    ax: Axes,
) -> None:
    """Plot threshold-sweep dual-axis curves faceted by quality tier."""
    required = {
        "prob_calibrated",
        "tier",
        "pred_label",
        "pred_label_baserate",
        "pred_label_maxf1",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")

    frame = predictions.copy()
    frame["prob_calibrated"] = pd.to_numeric(frame["prob_calibrated"], errors="coerce")
    frame = frame.dropna(subset=["prob_calibrated", "tier"])
    if frame.empty:
        raise ValueError("No finite prediction scores to plot.")

    frame["tier"] = frame["tier"].astype(str).str.upper()
    tiers = sorted(frame["tier"].unique())

    oof = anchor_oof.copy()
    oof["prob_calibrated_oof"] = pd.to_numeric(
        oof["prob_calibrated_oof"], errors="coerce"
    )
    oof["human_label"] = pd.to_numeric(oof["human_label"], errors="coerce")
    oof = oof.dropna(subset=["prob_calibrated_oof", "human_label"])

    sweep = np.linspace(0.0, 1.0, 200)
    pct_positive: dict[str, np.ndarray] = {}
    precision: np.ndarray = np.full_like(sweep, np.nan)

    for thresh_idx, thresh in enumerate(sweep):
        tp = ((oof["prob_calibrated_oof"] >= thresh) & (oof["human_label"] == 1)).sum()
        fp = ((oof["prob_calibrated_oof"] >= thresh) & (oof["human_label"] == 0)).sum()
        if tp + fp > 0:
            precision[thresh_idx] = tp / (tp + fp)

        for tier in tiers:
            tier_mask = frame["tier"] == tier
            if tier not in pct_positive:
                pct_positive[tier] = np.full_like(sweep, np.nan)
            pct_positive[tier][thresh_idx] = (
                frame.loc[tier_mask, "prob_calibrated"] >= thresh
            ).mean()

    low_band, high_band = _threshold_band(thresholds)

    fig = ax.figure
    ax.remove()
    axes = fig.subplots(len(tiers), 1, sharex=True, squeeze=False).ravel()

    for tier, tier_ax in zip(tiers, axes, strict=True):
        if low_band < high_band:
            tier_ax.axvspan(
                low_band,
                high_band,
                color=LIGHT_GREY,
                alpha=0.25,
                linewidth=0,
            )

        tier_ax.plot(
            sweep,
            pct_positive[tier] * 100,
            color=OKABE_ITO_BLUE,
            linewidth=1.2,
            label="Predicted positive (%)",
        )

        tier_ax_twin = tier_ax.twinx()
        tier_ax_twin.plot(
            sweep,
            precision * 100,
            color=OKABE_ITO_ORANGE,
            linewidth=1.2,
            linestyle="--",
            label="Precision (%)",
        )

        for name, threshold in thresholds.items():
            tier_ax.axvline(
                threshold,
                color=_threshold_color(name),
                linestyle=_threshold_linestyle(name),
                linewidth=0.9,
                label=name.replace("_", " ").title(),
            )

        tier_ax.set_ylabel(f"{tier}\nPredicted positive (%)", color=OKABE_ITO_BLUE)
        tier_ax_twin.set_ylabel("Precision (%)", color=OKABE_ITO_ORANGE)
        tier_ax.set_ylim(0, 100)
        tier_ax_twin.set_ylim(0, 100)
        tier_ax.grid(alpha=0.20)

        handles1, labels1 = tier_ax.get_legend_handles_labels()
        handles2, labels2 = tier_ax_twin.get_legend_handles_labels()
        tier_ax.legend(
            handles=handles1 + handles2,
            labels=labels1 + labels2,
            loc="upper right",
            fontsize=6.5,
        )

    axes[0].set_title("Threshold-sweep: predicted positive rate and precision by tier")
    axes[-1].set_xlabel("Threshold")

    logger.info(
        "Rendered threshold-sweep plot for %d tiers, %d sweep points",
        len(tiers),
        len(sweep),
    )
