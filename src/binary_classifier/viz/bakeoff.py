"""Bake-off and annotation-stage diagnostic plots."""

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
    pad_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_THRESHOLD = 0.70
_HIGH_ABSTAIN_RATE = 0.25


def bakeoff_summary(results: object, ax: Axes) -> None:
    """Plot bake-off minority-F1 intervals and Cohen's kappa by model arm.

    Args:
        results: Sequence from ``bakeoff_results.json``. Each row must contain
            ``model_id``, ``prompt_id``, ``source_id``, and ``scores`` with
            ``f1``, ``abstain_rate``, ``n_valid``, ``metrics.cohens_kappa``,
            and ``metrics.bootstrap_ci.minority_f1.{lower,upper}``.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If ``results`` is empty or lacks required fields.

    """
    rows = [_bakeoff_row(row) for row in _as_mapping_rows(results, "bakeoff_summary")]
    frame = pd.DataFrame(rows).sort_values("f1", ascending=True).reset_index(drop=True)
    if frame.empty:
        raise ValueError("bakeoff_summary requires at least one result row.")

    y_positions = np.arange(len(frame), dtype=float)
    f1 = frame["f1"].to_numpy(dtype=float)
    lower = frame["f1_ci_lower"].to_numpy(dtype=float)
    upper = frame["f1_ci_upper"].to_numpy(dtype=float)
    kappa = frame["cohens_kappa"].to_numpy(dtype=float)
    xerr = np.vstack([np.maximum(f1 - lower, 0.0), np.maximum(upper - f1, 0.0)])
    high_abstain = frame["abstain_rate"].to_numpy(dtype=float) >= _HIGH_ABSTAIN_RATE

    ax.axvspan(0.0, _THRESHOLD, color=LIGHT_GREY, alpha=0.18, linewidth=0)
    ax.axvline(
        _THRESHOLD,
        color=OKABE_ITO_BLACK,
        linestyle="--",
        linewidth=0.9,
        label="Decision floor (0.70)",
    )
    ax.errorbar(
        f1,
        y_positions,
        xerr=xerr,
        fmt="o",
        color=OKABE_ITO_BLUE,
        ecolor=OKABE_ITO_BLUE,
        capsize=3,
        label="Minority F1 (bootstrap CI)",
        clip_on=False,
    )
    ax.scatter(
        kappa,
        y_positions,
        marker="D",
        s=24,
        color=OKABE_ITO_BLUISH_GREEN,
        label="Cohen's κ (point only)",
        zorder=3,
        clip_on=False,
    )
    if high_abstain.any():
        ax.scatter(
            f1[high_abstain],
            y_positions[high_abstain],
            marker="o",
            s=90,
            facecolors="none",
            edgecolors=OKABE_ITO_VERMILLION,
            linewidths=1.2,
            label=f"High abstain (≥{_HIGH_ABSTAIN_RATE:.0%})",
            zorder=4,
            clip_on=False,
        )

    for idx, row in enumerate(frame.to_dict("records")):
        color = (
            OKABE_ITO_VERMILLION
            if float(row["abstain_rate"]) >= _HIGH_ABSTAIN_RATE
            else MUTED_GREY
        )
        ax.text(
            1.01,
            idx,
            f"abstain {float(row['abstain_rate']):.0%}; n={int(row['n_valid'])}",
            va="center",
            ha="left",
            fontsize=7,
            color=color,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

    labels = [
        f"{row['model_id']}\n{row['prompt_id']}" for row in frame.to_dict("records")
    ]
    ax.set_yticks(y_positions, labels=labels)
    ax.set_xlim(0.0, 1.05)
    pad_axes(ax, x=0.02, y=0.0)
    ax.set_xlabel("Score")
    ax.set_ylabel("Model × prompt arm")
    ax.set_title("Bake-off: minority-F1 and chance-corrected agreement")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper left")
    logger.info("Rendered bake-off summary for %d arms", len(frame))


def production_annotation_summary(df: pd.DataFrame, ax: Axes) -> None:
    """Plot production annotation abstain and agreement diagnostics.

    Args:
        df: Long/tidy annotation dataframe with at least ``EIN2``,
            ``source_id``, and numeric ``label`` where ``NaN`` means abstain.
            If a non-duplicated ``model_id`` column is present, diagnostics are
            per model; otherwise they are per ``source_id`` model-prompt arm.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If the dataframe is empty or lacks required columns.

    """
    _require_frame_columns(
        df,
        {"EIN2", "source_id", "label"},
        "production_annotation_summary",
    )

    source_column = _production_source_column(df)
    labels = df[["EIN2", source_column, "label"]].copy()
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    matrix = labels.drop_duplicates(["EIN2", source_column], keep="last").pivot(
        index="EIN2",
        columns=source_column,
        values="label",
    )
    if matrix.empty:
        raise ValueError(
            "production_annotation_summary has no model-task labels to plot.",
        )

    abstain_rate = matrix.isna().mean(axis=0).sort_values(ascending=True)
    pairwise_agreement = _mean_pairwise_agreement(matrix)
    vote_counts = matrix.notna().sum(axis=1).astype(int)
    tie_rate = _tie_rate(matrix)
    all_abstain_rate = float((vote_counts == 0).mean()) if len(vote_counts) else np.nan

    rows: list[dict[str, Any]] = [
        {
            "label": str(source_id),
            "value": float(value),
            "color": OKABE_ITO_ORANGE
            if value >= _HIGH_ABSTAIN_RATE
            else OKABE_ITO_BLUE,
            "kind": "Abstain rate",
        }
        for source_id, value in abstain_rate.items()
    ]
    rows.extend(
        [
            {
                "label": "Mean pairwise agreement",
                "value": pairwise_agreement,
                "color": OKABE_ITO_BLUISH_GREEN,
                "kind": "Agreement",
            },
            {
                "label": "Tie rate",
                "value": tie_rate,
                "color": OKABE_ITO_VERMILLION,
                "kind": "Vote distribution",
            },
            {
                "label": "All-abstain rate",
                "value": all_abstain_rate,
                "color": MUTED_GREY,
                "kind": "Vote distribution",
            },
        ],
    )
    plot_frame = pd.DataFrame(rows)
    y_positions = np.arange(len(plot_frame), dtype=float)
    values = plot_frame["value"].to_numpy(dtype=float)

    ax.barh(y_positions, values, color=plot_frame["color"].to_list(), alpha=0.9)
    ax.axvline(_HIGH_ABSTAIN_RATE, color=MUTED_GREY, linestyle=":", linewidth=0.9)
    for idx, row in enumerate(plot_frame.to_dict("records")):
        value = row["value"]
        label = "n/a" if pd.isna(value) else f"{float(value):.0%}"
        ax.text(
            min(1.01, 0.02 + (0.0 if pd.isna(value) else float(value))),
            idx,
            label,
            va="center",
            ha="left",
            fontsize=7,
            color=OKABE_ITO_BLACK,
        )

    vote_distribution = vote_counts.value_counts().sort_index()
    dist_label = ", ".join(
        f"{int(votes)} votes: {int(count)}"
        for votes, count in vote_distribution.items()
    )
    ax.text(
        0.02,
        0.02,
        f"Vote-count distribution: {dist_label}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7,
        color=OKABE_ITO_BLACK,
        bbox={"facecolor": "white", "edgecolor": LIGHT_GREY, "alpha": 0.85, "pad": 3},
    )

    ax.set_yticks(y_positions, labels=plot_frame["label"].to_list())
    ax.set_xlim(0.0, 1.08)
    pad_axes(ax, x=0.02, y=0.0)
    ax.set_xlabel("Rate")
    ax.set_ylabel("Model/source diagnostic")
    ax.set_title("Production annotation diagnostics")
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=OKABE_ITO_BLUE, label="Abstain rate"),
        Patch(facecolor=OKABE_ITO_ORANGE, label="High abstain (≥25%)"),
        Patch(facecolor=OKABE_ITO_BLUISH_GREEN, label="Mean pairwise agreement"),
        Patch(facecolor=OKABE_ITO_VERMILLION, label="Tie rate"),
        Patch(facecolor=MUTED_GREY, label="All-abstain rate"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    ax.grid(axis="x", alpha=0.25)
    logger.info(
        "Rendered production annotation summary for %d sources and %d rows",
        matrix.shape[1],
        matrix.shape[0],
    )


def canary_drift(rows: object, ax: Axes) -> None:
    """Plot canary κ/α comparisons against the monitor baseline over runs.

    Args:
        rows: Iterable of ``canary_drift_audit.jsonl`` records. Each row must
            include ``run_timestamp`` and ``kappa_alpha_change_test`` with
            ``cohens_kappa``, ``krippendorff_alpha``, ``n_changed``, and
            ``n_common`` keys. Baseline rows may store null κ/α.
        ax: Matplotlib axes to draw into.

    Raises:
        ValueError: If ``rows`` is empty or lacks required fields.

    """
    parsed = [
        _canary_row(row, idx)
        for idx, row in enumerate(_as_mapping_rows(rows, "canary_drift"))
    ]
    frame = (
        pd.DataFrame(parsed)
        .sort_values(["timestamp", "run_index"])
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("canary_drift requires at least one audit row.")

    x_positions = np.arange(len(frame), dtype=float)
    ax.axhspan(0.0, _THRESHOLD, color=LIGHT_GREY, alpha=0.18, linewidth=0)
    ax.axhline(
        _THRESHOLD,
        color=OKABE_ITO_BLACK,
        linestyle="--",
        linewidth=0.9,
        label="Review floor (0.70)",
    )
    for metric, color, label in (
        ("cohens_kappa", OKABE_ITO_BLUE, "Cohen's κ"),
        ("krippendorff_alpha", OKABE_ITO_BLUISH_GREEN, "Krippendorff's α"),
    ):
        values = frame[metric].to_numpy(dtype=float)
        if np.isfinite(values).any():
            ax.plot(x_positions, values, marker="o", color=color, label=label)

    for idx, row in enumerate(frame.to_dict("records")):
        text = str(row["status"])
        if pd.notna(row["n_common"]):
            text = f"{text}\nchanged {int(row['n_changed'])}/{int(row['n_common'])}"
        color = OKABE_ITO_VERMILLION if bool(row["changed"]) else MUTED_GREY
        y_value = row["cohens_kappa"]
        y_text = 0.96 if pd.isna(y_value) else min(0.98, float(y_value) + 0.05)
        ax.text(idx, y_text, text, ha="center", va="bottom", fontsize=7, color=color)

    if not frame[["cohens_kappa", "krippendorff_alpha"]].notna().any().any():
        ax.text(
            0.5,
            0.5,
            "Baseline only: no comparison run yet",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED_GREY,
        )

    tick_labels = [_short_timestamp(value) for value in frame["timestamp"].astype(str)]
    ax.set_xticks(x_positions, labels=tick_labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    pad_axes(ax, x=0.0, y=0.02)
    ax.set_ylabel("Agreement with baseline")
    ax.set_xlabel("Canary monitor run")
    ax.set_title("Canary drift audit")
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower left")
    logger.info("Rendered canary drift plot for %d audit rows", len(frame))


def _as_mapping_rows(data: object, plot_name: str) -> list[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        maybe_rows = data.get("results") or data.get("rows")
        if maybe_rows is None:
            raise ValueError(
                f"{plot_name} expected a sequence of mappings, got a mapping.",
            )
        data = maybe_rows
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            data = list(data)
        else:
            raise ValueError(f"{plot_name} expected a sequence of mappings.")
    rows = list(cast(Sequence[object], data))
    if not rows:
        raise ValueError(f"{plot_name} requires at least one row.")
    output: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{plot_name} rows must be mappings.")
        output.append(cast(Mapping[str, Any], row))
    return output


def _require_frame_columns(
    df: pd.DataFrame,
    required: set[str],
    plot_name: str,
) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{plot_name} missing columns: {missing}.")
    if df.empty:
        raise ValueError(f"{plot_name} requires at least one row.")


def _production_source_column(df: pd.DataFrame) -> str:
    if "model_id" not in df.columns:
        return "source_id"
    model_rows = df[["EIN2", "model_id"]]
    if model_rows["model_id"].isna().any():
        return "source_id"
    if model_rows.duplicated(["EIN2", "model_id"]).any():
        return "source_id"
    return "model_id"


def _bakeoff_row(row: Mapping[str, Any]) -> dict[str, Any]:
    scores = _nested_mapping(row, "scores", "bakeoff_summary")
    metrics = _nested_mapping(scores, "metrics", "bakeoff_summary")
    bootstrap_ci = _nested_mapping(metrics, "bootstrap_ci", "bakeoff_summary")
    minority_f1 = _nested_mapping(bootstrap_ci, "minority_f1", "bakeoff_summary")
    return {
        "model_id": str(_required_value(row, "model_id", "bakeoff_summary")),
        "prompt_id": str(_required_value(row, "prompt_id", "bakeoff_summary")),
        "source_id": str(_required_value(row, "source_id", "bakeoff_summary")),
        "f1": _finite_float(
            _required_value(scores, "f1", "bakeoff_summary"),
            "scores.f1",
        ),
        "f1_ci_lower": _finite_float(
            _required_value(minority_f1, "lower", "bakeoff_summary"),
            "metrics.bootstrap_ci.minority_f1.lower",
        ),
        "f1_ci_upper": _finite_float(
            _required_value(minority_f1, "upper", "bakeoff_summary"),
            "metrics.bootstrap_ci.minority_f1.upper",
        ),
        "cohens_kappa": _finite_float(
            _required_value(metrics, "cohens_kappa", "bakeoff_summary"),
            "metrics.cohens_kappa",
        ),
        "abstain_rate": _finite_float(
            _required_value(scores, "abstain_rate", "bakeoff_summary"),
            "scores.abstain_rate",
        ),
        "n_valid": int(_required_value(scores, "n_valid", "bakeoff_summary")),
    }


def _canary_row(row: Mapping[str, Any], run_index: int) -> dict[str, Any]:
    test = _nested_mapping(row, "kappa_alpha_change_test", "canary_drift")
    timestamp = str(_required_value(row, "run_timestamp", "canary_drift"))
    return {
        "run_index": run_index,
        "timestamp": timestamp,
        "status": str(_required_value(test, "status", "canary_drift")),
        "changed": bool(test.get("changed", False)),
        "n_common": _optional_int(test.get("n_common")),
        "n_changed": _optional_int(test.get("n_changed")),
        "cohens_kappa": _optional_float(test.get("cohens_kappa")),
        "krippendorff_alpha": _optional_float(test.get("krippendorff_alpha")),
    }


def _nested_mapping(
    row: Mapping[str, Any],
    key: str,
    plot_name: str,
) -> Mapping[str, Any]:
    value = _required_value(row, key, plot_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{plot_name} field '{key}' must be a mapping.")
    return cast(Mapping[str, Any], value)


def _required_value(row: Mapping[str, Any], key: str, plot_name: str) -> Any:
    if key not in row:
        raise ValueError(f"{plot_name} missing required field: {key}.")
    value = row[key]
    if value is None:
        raise ValueError(f"{plot_name} field '{key}' cannot be null.")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _optional_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _optional_int(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(int(value))


def _mean_pairwise_agreement(matrix: pd.DataFrame) -> float:
    agreements: list[float] = []
    columns = list(matrix.columns)
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            paired = matrix[[left, right]].dropna()
            if not paired.empty:
                agreements.append(float((paired[left] == paired[right]).mean()))
    return float(np.mean(agreements)) if agreements else float("nan")


def _tie_rate(matrix: pd.DataFrame) -> float:
    ties = []
    for _, row in matrix.iterrows():
        labels = row.dropna()
        if labels.empty:
            ties.append(False)
            continue
        counts = labels.value_counts()
        ties.append(bool((counts == counts.iloc[0]).sum() > 1))
    return float(np.mean(ties)) if ties else float("nan")


def _short_timestamp(value: str) -> str:
    timestamp = value.replace("T", " ")
    if "+" in timestamp:
        timestamp = timestamp.split("+", maxsplit=1)[0]
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1]
    return timestamp[:16]
