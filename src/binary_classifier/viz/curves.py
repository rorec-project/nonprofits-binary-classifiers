"""Curve plots for training, evaluation, and calibration artifacts."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def documentation_curve(
    results_jsonl_rows: Iterable[Mapping[str, Any]], ax: "Axes"
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


def pr_curve(points: object, ax: "Axes") -> None:
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


def reliability_diagram(points: object, ax: "Axes", ece: float | None = None) -> None:
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

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=0.9)
    ax.scatter(
        frame["mean_predicted"].to_numpy(dtype=float),
        frame["observed_fraction"].to_numpy(dtype=float),
        s=marker_sizes,
        color="tab:blue",
        alpha=0.8,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    title = "Reliability diagram"
    if ece is not None:
        title = f"{title} (ECE={ece:.3f})"
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    logger.info("Rendered reliability diagram with %d bins", len(frame))


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
    points: object, containers: tuple[str, ...]
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
