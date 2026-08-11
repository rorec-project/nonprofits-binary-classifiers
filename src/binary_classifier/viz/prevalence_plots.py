"""Prevalence-estimate plots for stage-09 artifacts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from binary_classifier.data.ntee_labels import load_ntee_labels
from binary_classifier.viz.style import (
    LIGHT_GREY,
    MUTED_GREY,
    OKABE_ITO_BLACK,
    OKABE_ITO_BLUE,
    OKABE_ITO_BLUISH_GREEN,
    OKABE_ITO_ORANGE,
    OKABE_ITO_VERMILLION,
    pad_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_UNIVERSE = "501C3-charity mission frame"

_REQUIRED_COLUMNS = {
    "ntee_major_group",
    "n_anchor",
    "estimator",
    "estimate",
    "ci_lower",
    "ci_upper",
    "suppressed",
}


def prevalence_forest(prevalence_by_ntee_df: pd.DataFrame, ax: Axes) -> None:
    """Plot per-NTEE prevalence estimates with confidence intervals.

    Args:
        prevalence_by_ntee_df: DataFrame with the stage-09
            ``prevalence_by_ntee.csv`` schema: ``ntee_major_group``,
            ``n_anchor``, ``estimator``, ``estimate``, ``ci_lower``, ``ci_upper``,
            and ``suppressed``.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If the artifact schema is missing required columns or contains
            no rows.

    """
    missing = sorted(_REQUIRED_COLUMNS - set(prevalence_by_ntee_df.columns))
    if missing:
        raise ValueError(f"prevalence_by_ntee_df missing columns: {missing}.")
    if prevalence_by_ntee_df.empty:
        raise ValueError("prevalence_by_ntee_df must contain at least one row.")

    frame = prevalence_by_ntee_df.copy().sort_values("ntee_major_group")
    suppressed = frame["suppressed"].astype(bool).to_numpy()
    estimates = pd.to_numeric(frame["estimate"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(frame["ci_lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(frame["ci_upper"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(estimates[~suppressed]).all():
        n_bad = int(np.sum(~np.isfinite(estimates[~suppressed])))
        raise ValueError(
            f"{n_bad} non-suppressed prevalence estimate(s) are non-finite."
        )

    keep = ~suppressed
    frame = frame[keep].reset_index(drop=True)
    suppressed = frame["suppressed"].astype(bool).to_numpy()
    estimates = estimates[keep]
    lower = lower[keep]
    upper = upper[keep]
    y_positions = np.arange(len(frame), dtype=float)
    labels = [
        f"{group} (n={int(n_anchor)})"
        for group, n_anchor in zip(
            frame["ntee_major_group"].astype(str),
            frame["n_anchor"],
            strict=True,
        )
    ]

    for is_suppressed in (False, True):
        mask = suppressed == is_suppressed
        if not mask.any():
            continue
        color = MUTED_GREY if is_suppressed else OKABE_ITO_BLUE
        label = "Suppressed" if is_suppressed else "Estimate"
        finite_ci = mask & np.isfinite(lower) & np.isfinite(upper)
        if finite_ci.any():
            xerr = np.vstack(
                [
                    np.maximum(estimates[finite_ci] - lower[finite_ci], 0.0),
                    np.maximum(upper[finite_ci] - estimates[finite_ci], 0.0),
                ],
            )
            ax.errorbar(
                estimates[finite_ci],
                y_positions[finite_ci],
                xerr=xerr,
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                label=label,
                clip_on=False,
            )
        no_ci = mask & ~finite_ci
        if no_ci.any():
            ax.scatter(
                estimates[no_ci],
                y_positions[no_ci],
                color=color,
                marker="o",
                label=label if not finite_ci.any() else None,
                clip_on=False,
            )

    ax.set_yticks(y_positions, labels=labels)
    ax.axvline(0.0, color=OKABE_ITO_BLACK, linewidth=0.8, alpha=0.6)
    ax.set_xlim(0.0, min(1.05, max(1.05, float(upper.max() * 1.05))))
    pad_axes(ax, x=0.02, y=0.0)
    ax.set_xlabel("Estimated religious prevalence")
    ax.set_ylabel("NTEE major group")
    ax.set_title("Prevalence by NTEE group")
    ax.grid(axis="x", alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    deduped = dict(zip(legend_labels, handles, strict=True))
    ax.legend(deduped.values(), deduped.keys())
    logger.info("Rendered prevalence forest plot with %d groups", len(frame))


def prevalence_decomposition(report: Mapping[str, Any], ax: Axes) -> None:
    """Render component contributions to the composite prevalence estimate."""
    rows = _decomposition_rows(report)
    if not rows:
        raise ValueError("prevalence report has no decomposition components.")
    frame = pd.DataFrame(rows)
    frame["contribution"] = frame["share"] * frame["estimate"]
    frame["ci_lower_contribution"] = frame["share"] * frame["ci_lower"]
    frame["ci_upper_contribution"] = frame["share"] * frame["ci_upper"]
    frame = frame.sort_values("label").reset_index(drop=True)

    starts = frame["contribution"].cumsum().shift(fill_value=0.0)
    colors = [OKABE_ITO_BLUE, OKABE_ITO_BLUISH_GREEN, OKABE_ITO_ORANGE]
    for idx, row in enumerate(frame.to_dict("records")):
        contribution = float(row["contribution"])
        start = float(starts.iloc[idx])
        ax.bar(
            idx,
            contribution,
            bottom=start,
            color=colors[idx % len(colors)],
            alpha=0.88,
        )
        lower = float(row["ci_lower_contribution"])
        upper = float(row["ci_upper_contribution"])
        if np.isfinite(lower) and np.isfinite(upper):
            center = start + contribution
            ax.errorbar(
                idx,
                center,
                yerr=np.array(
                    [
                        [max(contribution - lower, 0.0)],
                        [max(upper - contribution, 0.0)],
                    ],
                ),
                fmt="none",
                ecolor=OKABE_ITO_BLACK,
                capsize=3,
            )
    total = float(frame["contribution"].sum())
    ax.axhline(total, color=OKABE_ITO_BLACK, linestyle="--", linewidth=0.9)
    ax.text(
        len(frame) - 0.5,
        total,
        f" composite {total:.1%}",
        va="bottom",
        fontsize=7,
    )
    ax.set_xticks(
        np.arange(len(frame)),
        labels=frame["label"].to_list(),
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Contribution to population prevalence")
    ax.set_title("Prevalence decomposition")
    ax.set_ylim(0.0, min(1.0, max(0.05, total * 1.35)))
    pad_axes(ax, x=0.0, y=0.02)
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=colors[0], label="HM-PPI"),
        Patch(facecolor=colors[1], label="LOW-PPI"),
        Patch(facecolor=colors[2], label="LOW-RG"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    logger.info("Rendered prevalence decomposition with %d components", len(frame))


def rule_validation_intervals(report: Mapping[str, Any], ax: Axes) -> None:
    """Plot rule-validation sensitivity/specificity Wilson intervals."""
    rows = []
    for name in ("sensitivity", "specificity"):
        metric = _find_rule_metric(report, name)
        if metric is None:
            continue
        rows.append({"metric": name.title(), **metric})
    if not rows:
        ax.text(
            0.5,
            0.5,
            "Rule-validation intervals not available\n(no rule-positive/negative anchor cells)",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.set_axis_off()
        ax.set_title("Rule-validation Wilson intervals")
        logger.info("Rendered rule-validation interval placeholder")
        return
    frame = pd.DataFrame(rows)
    y = np.arange(len(frame), dtype=float)
    estimates = frame["value"].to_numpy(float)
    lower = frame["ci_lower"].to_numpy(float)
    upper = frame["ci_upper"].to_numpy(float)
    xerr = np.vstack(
        [np.maximum(estimates - lower, 0.0), np.maximum(upper - estimates, 0.0)],
    )
    ax.errorbar(
        estimates,
        y,
        xerr=xerr,
        fmt="o",
        color=OKABE_ITO_BLUE,
        capsize=3,
        clip_on=False,
    )
    for idx, row in enumerate(frame.to_dict("records")):
        n_text = "" if pd.isna(row.get("n")) else f" (n={int(row['n'])})"
        value = float(row["value"])
        ax.text(
            min(1.01, value + 0.02),
            idx,
            f"{value:.1%}{n_text}",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(y, labels=frame["metric"].to_list())
    ax.set_xlim(0.0, 1.12)
    pad_axes(ax, x=0.02, y=0.0)
    ax.set_xlabel("Validation estimate")
    ax.set_title("Rule-validation Wilson intervals")
    ax.grid(axis="x", alpha=0.25)
    logger.info("Rendered rule-validation intervals")


def quantification_sensitivity(report: Mapping[str, Any], ax: Axes) -> None:
    """Plot PPI primary estimates against EMQ and weighting sensitivities."""
    rows = _quantification_rows(report)
    if not rows:
        raise ValueError("prevalence report has no quantification sensitivity rows.")
    frame = pd.DataFrame(rows).dropna(subset=["estimate"])
    if frame.empty:
        raise ValueError("No finite quantification sensitivity estimates to plot.")
    y = np.arange(len(frame), dtype=float)
    estimates = frame["estimate"].to_numpy(float)
    lower = pd.to_numeric(frame["ci_lower"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(frame["ci_upper"], errors="coerce").to_numpy(float)
    finite_ci = np.isfinite(lower) & np.isfinite(upper)
    if finite_ci.any():
        xerr = np.vstack(
            [
                np.maximum(estimates[finite_ci] - lower[finite_ci], 0.0),
                np.maximum(upper[finite_ci] - estimates[finite_ci], 0.0),
            ],
        )
        ax.errorbar(
            estimates[finite_ci],
            y[finite_ci],
            xerr=xerr,
            fmt="o",
            color=OKABE_ITO_BLUE,
            capsize=3,
            label="CI/range",
            clip_on=False,
        )
    if (~finite_ci).any():
        ax.scatter(
            estimates[~finite_ci],
            y[~finite_ci],
            color=OKABE_ITO_ORANGE,
            label="Point only",
            clip_on=False,
        )
    ax.set_yticks(y, labels=frame["label"].to_list())
    ax.set_xlim(0.0, min(1.0, max(0.1, float(np.nanmax(estimates)) * 1.4)))
    pad_axes(ax, x=0.02, y=0.0)
    ax.set_xlabel("Estimated prevalence")
    ax.set_title("Quantification sensitivity")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    logger.info("Rendered quantification sensitivity with %d rows", len(frame))


_NTEE_MEAN_SCORE_REQUIRED = {
    "ntee_major_group",
    "n",
    "n_scored",
    "mean_prob_raw",
    "mean_prob_calibrated",
    "share_pred_label",
}
_NTEE_CLASSIFIED_SHARE_REQUIRED = {
    "ntee_major_group",
    "share_pred_label",
    "share_pred_label_maxf1",
    "share_pred_label_baserate",
}
_NTEE_CLASSIFIED_COUNT_REQUIRED = {
    "ntee_major_group",
    "n",
    "n_pred_label",
    "n_pred_label_maxf1",
    "n_pred_label_baserate",
}
_NTEE_DESCRIPTIVES_SHARE_REQUIRED = {"ntee_major_group", "share_pred_label"}
_NTEE_PREVALENCE_REQUIRED = {
    "ntee_major_group",
    "estimate",
    "ci_lower",
    "ci_upper",
    "suppressed",
}
# (column suffix, series label, color, marker) for the three operating
# thresholds, shared by the share-based and count-based dot plots below.
_THRESHOLDS = (
    ("pred_label", "Recall-first", OKABE_ITO_BLUE, "o"),
    ("pred_label_maxf1", "Max-F1", OKABE_ITO_ORANGE, "s"),
    ("pred_label_baserate", "Base-rate precision", OKABE_ITO_BLUISH_GREEN, "^"),
)


def _threshold_columns(prefix: str) -> tuple[tuple[str, str, str, str], ...]:
    """Return (column, label, color, marker) for each threshold at ``prefix``."""
    return tuple(
        (f"{prefix}_{suffix}", label, color, marker)
        for suffix, label, color, marker in _THRESHOLDS
    )


def _ntee_order(
    frame: pd.DataFrame,
    *,
    sort_column: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """Attach human-readable NTEE labels and sort rows by a shared column.

    Args:
        frame: A frame with an ``ntee_major_group`` column.
        sort_column: Column to sort rows by.
        ascending: Sort direction; figures use descending order throughout.

    Returns:
        A copy of ``frame`` with an added ``label`` column, sorted and with a
        reset index.

    Raises:
        ValueError: If any ``ntee_major_group`` value is not one of the known
            NTEE major-group letters ``A``-``Z``.

    """
    labels = load_ntee_labels().set_index("ntee_major_group")["label"]
    out = frame.copy()
    out["ntee_major_group"] = out["ntee_major_group"].astype(str)
    out["label"] = out["ntee_major_group"].map(labels)
    if out["label"].isna().any():
        unknown = sorted(out.loc[out["label"].isna(), "ntee_major_group"].unique())
        raise ValueError(f"Unknown NTEE major group letter(s): {unknown}.")
    return out.sort_values(
        sort_column,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)


def _guide_lines(ax: Axes, y: np.ndarray, x_max: np.ndarray) -> None:
    """Draw a light row-guide line from the axis origin to each row's dot(s)."""
    for yi, xi in zip(y, x_max, strict=True):
        ax.plot([0.0, xi], [yi, yi], color=LIGHT_GREY, linewidth=0.8, zorder=1)


def ntee_mean_score_by_group(ntee_descriptives_df: pd.DataFrame, ax: Axes) -> None:
    """Plot calibrated and raw mean classifier score by NTEE major group.

    Two panels, one dot per NTEE letter each, x from 0 to 1. Covers only the
    classifier-scored subset of the **501C3-charity mission frame** — rows
    decided by the rule router before inference have no score.

    Args:
        ntee_descriptives_df: DataFrame with the stage-09/10
            ``ntee_descriptives.csv`` schema.
        ax: Matplotlib axes to draw into; replaced with a two-panel subplot
            layout sharing the y-axis.

    Raises:
        ValueError: If the artifact schema is missing required columns or
            contains no rows.

    """
    missing = sorted(_NTEE_MEAN_SCORE_REQUIRED - set(ntee_descriptives_df.columns))
    if missing:
        raise ValueError(f"ntee_descriptives_df missing columns: {missing}.")
    if ntee_descriptives_df.empty:
        raise ValueError("ntee_descriptives_df must contain at least one row.")

    frame = _ntee_order(ntee_descriptives_df, sort_column="share_pred_label")
    n_total = float(frame["n"].sum())
    n_scored_total = float(frame["n_scored"].sum())
    scored_share = n_scored_total / n_total if n_total else float("nan")
    scored_share_by_row = frame["n_scored"] / frame["n"].replace(0, np.nan)

    y = np.arange(len(frame), dtype=float)[::-1]

    fig = ax.figure
    ax.remove()
    axes = fig.subplots(1, 2, sharey=True)
    panels = (
        (axes[0], "mean_prob_calibrated", "Calibrated mean score"),
        (axes[1], "mean_prob_raw", "Raw mean score"),
    )
    for panel_ax, column, panel_title in panels:
        values = frame[column].to_numpy(dtype=float)
        _guide_lines(panel_ax, y, values)
        panel_ax.scatter(values, y, color=OKABE_ITO_BLUE, zorder=3, clip_on=False)
        panel_ax.set_xlim(0.0, 1.0)
        panel_ax.set_xlabel(panel_title)
        panel_ax.grid(axis="x", alpha=0.25)
        pad_axes(panel_ax, x=0.0, y=0.02)

    axes[0].set_yticks(y, labels=frame["label"].to_list())
    for yi, share in zip(y, scored_share_by_row, strict=True):
        axes[1].text(
            1.03,
            yi,
            f"{share:.0%} scored" if np.isfinite(share) else "n/a",
            va="center",
            fontsize=6,
            color=MUTED_GREY,
            clip_on=False,
        )
    fig.suptitle(
        f"Mean classifier score by NTEE major group, {_UNIVERSE}\n"
        f"Classifier-scored rows only ({scored_share:.1%})",
        fontsize=9,
    )
    logger.info("Rendered NTEE mean-score plot with %d groups", len(frame))


def ntee_classified_share_by_group(ntee_descriptives_df: pd.DataFrame, ax: Axes) -> None:
    """Plot classified share by NTEE major group at three operating thresholds.

    One row per NTEE letter, three dots per row (recall-first, max-F1,
    base-rate precision), denominator is all rows in the group (``n``). This
    is raw classifier output, not a prevalence estimate.

    Args:
        ntee_descriptives_df: DataFrame with the stage-09/10
            ``ntee_descriptives.csv`` schema.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If the artifact schema is missing required columns or
            contains no rows.

    """
    missing = sorted(_NTEE_CLASSIFIED_SHARE_REQUIRED - set(ntee_descriptives_df.columns))
    if missing:
        raise ValueError(f"ntee_descriptives_df missing columns: {missing}.")
    if ntee_descriptives_df.empty:
        raise ValueError("ntee_descriptives_df must contain at least one row.")

    frame = _ntee_order(ntee_descriptives_df, sort_column="share_pred_label")
    y = np.arange(len(frame), dtype=float)[::-1]

    thresholds = _threshold_columns("share")
    row_max = frame[[column for column, _, _, _ in thresholds]].max(axis=1).to_numpy(
        dtype=float
    )
    _guide_lines(ax, y, row_max)
    for column, label, color, marker in thresholds:
        ax.scatter(
            frame[column].to_numpy(dtype=float),
            y,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
            clip_on=False,
        )

    ax.set_yticks(y, labels=frame["label"].to_list())
    ax.set_xlim(0.0, min(1.0, max(0.1, float(row_max.max()) * 1.15)))
    pad_axes(ax, x=0.0, y=0.02)
    ax.set_xlabel("Classified share (raw classifier output)")
    ax.set_title(f"Classified share by NTEE major group, {_UNIVERSE}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Operating threshold", loc="lower right")
    logger.info("Rendered NTEE classified-share plot with %d groups", len(frame))


def ntee_classified_count_by_group(ntee_descriptives_df: pd.DataFrame, ax: Axes) -> None:
    """Plot the count of organizations classified religious, by NTEE group.

    One row per NTEE letter, three dots per row at the same three operating
    thresholds as :func:`ntee_classified_share_by_group`. Sorted by group
    size (``n``) descending, and rows are annotated with group size so counts
    can be read against their base.

    Args:
        ntee_descriptives_df: DataFrame with the stage-09/10
            ``ntee_descriptives.csv`` schema.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If the artifact schema is missing required columns or
            contains no rows.

    """
    missing = sorted(_NTEE_CLASSIFIED_COUNT_REQUIRED - set(ntee_descriptives_df.columns))
    if missing:
        raise ValueError(f"ntee_descriptives_df missing columns: {missing}.")
    if ntee_descriptives_df.empty:
        raise ValueError("ntee_descriptives_df must contain at least one row.")

    frame = _ntee_order(ntee_descriptives_df, sort_column="n")
    y = np.arange(len(frame), dtype=float)[::-1]
    labels = [
        f"{label} (n={int(n):,})"
        for label, n in zip(frame["label"], frame["n"], strict=True)
    ]

    thresholds = _threshold_columns("n")
    row_max = frame[[column for column, _, _, _ in thresholds]].max(axis=1).to_numpy(
        dtype=float
    )
    _guide_lines(ax, y, row_max)
    for column, label, color, marker in thresholds:
        ax.scatter(
            frame[column].to_numpy(dtype=float),
            y,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
            clip_on=False,
        )

    ax.set_yticks(y, labels=labels)
    ax.set_xlim(0.0, max(1.0, float(row_max.max()) * 1.1))
    pad_axes(ax, x=0.0, y=0.02)
    ax.set_xlabel("Organizations classified religious (count)")
    ax.set_title(f"Count classified religious by NTEE major group, {_UNIVERSE}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Operating threshold", loc="lower right")
    logger.info("Rendered NTEE classified-count plot with %d groups", len(frame))


def ntee_classified_share_vs_corrected_estimate(
    ntee_descriptives_df: pd.DataFrame,
    prevalence_by_ntee_df: pd.DataFrame,
    ax: Axes,
) -> None:
    """Plot classified share against the corrected prevalence estimate.

    A dumbbell chart: for each NTEE major group, the recall-first classified
    share is joined by a thin line to the corrected prevalence estimate (with
    its confidence interval), so the line length is the correction bias.
    Suppressed groups (fewer anchor rows than ``ntee_min_n``) are marked as
    having no estimate rather than dropped or drawn at zero.

    Args:
        ntee_descriptives_df: DataFrame with the stage-09/10
            ``ntee_descriptives.csv`` schema.
        prevalence_by_ntee_df: DataFrame with the stage-09
            ``prevalence_by_ntee.csv`` schema.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If either artifact schema is missing required columns, or
            either frame contains no rows.

    """
    missing_descriptives = sorted(
        _NTEE_DESCRIPTIVES_SHARE_REQUIRED - set(ntee_descriptives_df.columns)
    )
    if missing_descriptives:
        raise ValueError(
            f"ntee_descriptives_df missing columns: {missing_descriptives}."
        )
    missing_prevalence = sorted(
        _NTEE_PREVALENCE_REQUIRED - set(prevalence_by_ntee_df.columns)
    )
    if missing_prevalence:
        raise ValueError(
            f"prevalence_by_ntee_df missing columns: {missing_prevalence}."
        )
    if ntee_descriptives_df.empty or prevalence_by_ntee_df.empty:
        raise ValueError("Both input frames must contain at least one row.")

    descriptives = ntee_descriptives_df.copy()
    descriptives["ntee_major_group"] = descriptives["ntee_major_group"].astype(str)
    prevalence = prevalence_by_ntee_df.copy()
    prevalence["ntee_major_group"] = prevalence["ntee_major_group"].astype(str)

    merged = descriptives.merge(
        prevalence[["ntee_major_group", "estimate", "ci_lower", "ci_upper", "suppressed"]],
        on="ntee_major_group",
        how="left",
    )
    merged["suppressed"] = merged["suppressed"].fillna(True).astype(bool)
    frame = _ntee_order(merged, sort_column="share_pred_label")

    y = np.arange(len(frame), dtype=float)[::-1]
    classified = frame["share_pred_label"].to_numpy(dtype=float)
    estimate = pd.to_numeric(frame["estimate"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(frame["ci_lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(frame["ci_upper"], errors="coerce").to_numpy(dtype=float)
    suppressed = frame["suppressed"].to_numpy(dtype=bool)
    has_estimate = np.isfinite(estimate) & ~suppressed

    for yi, x0, x1, keep in zip(y, classified, estimate, has_estimate, strict=True):
        if keep:
            ax.plot([x0, x1], [yi, yi], color=LIGHT_GREY, linewidth=1.0, zorder=1)

    ax.scatter(
        classified,
        y,
        color=OKABE_ITO_BLUE,
        marker="o",
        label="Classified share",
        zorder=3,
        clip_on=False,
    )

    finite_ci = has_estimate & np.isfinite(lower) & np.isfinite(upper)
    if finite_ci.any():
        xerr = np.vstack(
            [
                np.maximum(estimate[finite_ci] - lower[finite_ci], 0.0),
                np.maximum(upper[finite_ci] - estimate[finite_ci], 0.0),
            ],
        )
        ax.errorbar(
            estimate[finite_ci],
            y[finite_ci],
            xerr=xerr,
            fmt="s",
            color=OKABE_ITO_VERMILLION,
            ecolor=OKABE_ITO_VERMILLION,
            capsize=3,
            label="Corrected prevalence estimate",
            zorder=3,
            clip_on=False,
        )
    point_only = has_estimate & ~finite_ci
    if point_only.any():
        ax.scatter(
            estimate[point_only],
            y[point_only],
            color=OKABE_ITO_VERMILLION,
            marker="s",
            label=None if finite_ci.any() else "Corrected prevalence estimate",
            zorder=3,
            clip_on=False,
        )

    all_x = np.concatenate([classified, estimate[has_estimate], upper[finite_ci]])
    x_max = min(1.0, max(0.1, float(np.nanmax(all_x)) * 1.15)) if all_x.size else 1.0
    ax.set_xlim(0.0, x_max)
    pad_axes(ax, x=0.0, y=0.02)

    if suppressed.any():
        for yi in y[suppressed]:
            ax.text(
                x_max * 0.99,
                yi,
                "suppressed — no estimate",
                va="center",
                ha="right",
                fontsize=6,
                color=MUTED_GREY,
                style="italic",
                clip_on=False,
            )

    ax.set_yticks(y, labels=frame["label"].to_list())
    ax.set_xlabel("Share")
    ax.set_title(
        f"Classified share against corrected prevalence estimate, "
        f"by NTEE major group, {_UNIVERSE}",
    )
    ax.grid(axis="x", alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    deduped = dict(zip(legend_labels, handles, strict=True))
    ax.legend(deduped.values(), deduped.keys(), loc="lower left")
    logger.info(
        "Rendered NTEE classified-share-vs-corrected-estimate plot with %d groups",
        len(frame),
    )


def _decomposition_rows(report: Mapping[str, Any]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    shares = report.get("tier_shares")
    if not isinstance(shares, Mapping):
        return rows
    hm = report.get("hm")
    if isinstance(hm, Mapping):
        primary = str(hm.get("primary", "weighted_ppi"))
        estimate = hm.get(primary)
        if isinstance(estimate, Mapping):
            rows.append(
                _component_row(
                    "HM-PPI", _as_float(shares.get("HIGH_MEDIUM")), estimate
                ),
            )
    low = report.get("low")
    if isinstance(low, Mapping):
        low_share = _as_float(shares.get("LOW"))
        sub_strata = low.get("sub_strata")
        if isinstance(sub_strata, Mapping):
            classifier = sub_strata.get("low_via_classifier")
            if isinstance(classifier, Mapping):
                rows.append(
                    _component_row(
                        "LOW-PPI",
                        low_share * _as_float(classifier.get("share")),
                        classifier.get("estimate"),
                    ),
                )
            rule = sub_strata.get("rule")
            if isinstance(rule, Mapping):
                rows.append(
                    _component_row(
                        "LOW-RG",
                        low_share * _as_float(rule.get("share")),
                        rule.get("estimate"),
                    ),
                )
        elif isinstance(low.get("estimate"), Mapping):
            rows.append(_component_row("LOW-RG", low_share, low.get("estimate")))
    return [row for row in rows if np.isfinite(float(row["estimate"]))]


def _component_row(
    label: str,
    share: float,
    estimate: object,
) -> dict[str, float | str]:
    if not isinstance(estimate, Mapping):
        raise ValueError(f"Component {label} missing estimate mapping.")
    point = _as_float(estimate.get("estimate"), np.nan)
    return {
        "label": label,
        "share": float(share),
        "estimate": point,
        "ci_lower": _as_float(estimate.get("ci_lower"), point),
        "ci_upper": _as_float(estimate.get("ci_upper"), point),
    }


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _find_rule_metric(report: Mapping[str, Any], name: str) -> dict[str, float] | None:
    metric = None
    metrics = report.get("metrics")
    if isinstance(metrics, Mapping):
        metric = metrics.get(name) or metrics.get(f"rule_{name}")
    metric = metric or report.get(name) or report.get(f"rule_{name}")
    if metric is None:
        return None
    if isinstance(metric, int | float):
        value = float(metric)
        return {"value": value, "ci_lower": value, "ci_upper": value, "n": np.nan}
    if not isinstance(metric, Mapping):
        return None
    value_obj = metric.get("value", metric.get("estimate", metric.get("point")))
    if value_obj is None:
        return None
    value = float(value_obj)
    ci = (
        metric.get("ci") or metric.get("wilson_ci") or metric.get("confidence_interval")
    )
    if isinstance(ci, Mapping):
        lower = ci.get("lower", ci.get("lo", ci.get("lcl", value)))
        upper = ci.get("upper", ci.get("hi", ci.get("ucl", value)))
    else:
        lower = metric.get("ci_lower", metric.get("lower", value))
        upper = metric.get("ci_upper", metric.get("upper", value))
    n = metric.get("n", metric.get("denominator", np.nan))
    return {
        "value": value,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n": float(n),
    }


def _quantification_rows(report: Mapping[str, Any]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    hm = report.get("hm")
    if isinstance(hm, Mapping):
        for key, label in (
            ("weighted_ppi", "PPI weighted"),
            ("unweighted_ppi", "PPI unweighted"),
        ):
            estimate = hm.get(key)
            if isinstance(estimate, Mapping):
                rows.append(_estimate_row(label, estimate))
    cross_checks = report.get("cross_checks")
    if isinstance(cross_checks, Mapping):
        for key, payload in cross_checks.items():
            if (
                isinstance(payload, Mapping)
                and payload.get("status") == "ok"
                and "estimate" in payload
            ):
                rows.append(
                    {
                        "label": str(key).upper(),
                        "estimate": float(payload["estimate"]),
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                    },
                )
    sensitivity = None
    if isinstance(hm, Mapping):
        sensitivity = hm.get("sensitivity_anchor_residual_multiplicity_weighted")
    if isinstance(sensitivity, Mapping) and "estimate" in sensitivity:
        rows.append(_estimate_row("PPI anchor multiplicity", sensitivity))
    return rows


def _estimate_row(label: str, estimate: Mapping[str, Any]) -> dict[str, float | str]:
    point = float(estimate.get("estimate", np.nan))
    return {
        "label": label,
        "estimate": point,
        "ci_lower": float(estimate.get("ci_lower", np.nan)),
        "ci_upper": float(estimate.get("ci_upper", np.nan)),
    }
