"""Prevalence-estimate plots for stage-09 artifacts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from binary_classifier.viz.style import (
    MUTED_GREY,
    OKABE_ITO_BLACK,
    OKABE_ITO_BLUE,
    OKABE_ITO_BLUISH_GREEN,
    OKABE_ITO_ORANGE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

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
    estimates = pd.to_numeric(frame["estimate"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(frame["ci_lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(frame["ci_upper"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(estimates).all():
        raise ValueError("Prevalence estimates must be finite.")

    suppressed = frame["suppressed"].astype(bool).to_numpy()
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
            )
        no_ci = mask & ~finite_ci
        if no_ci.any():
            ax.scatter(
                estimates[no_ci],
                y_positions[no_ci],
                color=color,
                marker="o",
                label=label if not finite_ci.any() else None,
            )

    ax.set_yticks(y_positions, labels=labels)
    ax.axvline(0.0, color=OKABE_ITO_BLACK, linewidth=0.8, alpha=0.6)
    ax.set_xlim(0.0, min(1.0, max(1.0, float(np.nanmax(upper)))))
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
    ax.grid(axis="y", alpha=0.25)
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
    ax.errorbar(estimates, y, xerr=xerr, fmt="o", color=OKABE_ITO_BLUE, capsize=3)
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
    ax.set_xlim(0.0, 1.05)
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
        )
    if (~finite_ci).any():
        ax.scatter(
            estimates[~finite_ci],
            y[~finite_ci],
            color=OKABE_ITO_ORANGE,
            label="Point only",
        )
    ax.set_yticks(y, labels=frame["label"].to_list())
    ax.set_xlim(0.0, min(1.0, max(0.1, float(np.nanmax(estimates)) * 1.4)))
    ax.set_xlabel("Estimated prevalence")
    ax.set_title("Quantification sensitivity")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    logger.info("Rendered quantification sensitivity with %d rows", len(frame))


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
