"""Prevalence-estimate plots for stage-09 artifacts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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


def prevalence_forest(prevalence_by_ntee_df: pd.DataFrame, ax: "Axes") -> None:
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
        color = "0.65" if is_suppressed else "tab:blue"
        label = "Suppressed" if is_suppressed else "Estimate"
        finite_ci = mask & np.isfinite(lower) & np.isfinite(upper)
        if finite_ci.any():
            xerr = np.vstack(
                [
                    np.maximum(estimates[finite_ci] - lower[finite_ci], 0.0),
                    np.maximum(upper[finite_ci] - estimates[finite_ci], 0.0),
                ]
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
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0.0, min(1.0, max(1.0, float(np.nanmax(upper)))))
    ax.set_xlabel("Estimated religious prevalence")
    ax.set_ylabel("NTEE major group")
    ax.set_title("Prevalence by NTEE group")
    ax.grid(axis="x", alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    deduped = dict(zip(legend_labels, handles, strict=True))
    ax.legend(deduped.values(), deduped.keys())
    logger.info("Rendered prevalence forest plot with %d groups", len(frame))
