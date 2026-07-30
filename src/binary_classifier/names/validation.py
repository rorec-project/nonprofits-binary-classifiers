"""Paired validation of mission-to-name cross-field transfer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from binary_classifier import metrics
from binary_classifier.evaluation.calibration import calibration_metrics

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


_NAME_SCORE_COLUMNS = {
    "EIN2",
    "input_variant",
    "prob_raw",
    "lexicon_rule_label",
}
_MISSION_SCORE_COLUMNS = {"EIN2", "pred_label", "prob_calibrated"}
_PANEL_COLUMNS = {"EIN2", "has_mission", "is_manifest_contaminated"}
_INPUT_VARIANTS = {"suffix_stripped", "suffix_retaining"}
_LIMITATIONS = [
    "This paired test can falsify transfer cheaply but cannot validate it: "
    "mission-derived labels can mark a correct name prediction wrong, and the "
    "filing population excludes organizations motivating the names arm."
]


def run_name_validation(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> None:
    """Compare name transfer and lexicon baselines on paired, clean organizations.

    Mission deployment labels are a deliberately limited reference: the paired
    design holds the organization constant but does not turn mission-text labels
    into name-text gold labels. Transfer decisions use the top ``k`` raw scores,
    where ``k`` is the lexicon's positive count, to match operating points.
    """
    paired, counts = _load_paired_frame(registry)
    variants = {
        variant: _variant_report(cfg, variant_frame)
        for variant, variant_frame in paired.groupby("input_variant", sort=True)
    }
    if not variants:
        raise ValueError("No paired name-score variants are available for validation.")

    report = {
        "comparison_population": counts,
        "reference": {
            "mission_label": "pred_label",
            "mission_score": "prob_calibrated",
            "lexicon_baseline": (
                "Strong-tradition rule positives; rule negatives and abstentions are "
                "treated as negative for a binary, equal-cohort comparison."
            ),
        },
        "variants": variants,
        "limitations": _LIMITATIONS,
    }
    registry.ensure_dirs()
    registry.names_validation.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def _load_paired_frame(registry: PathRegistry) -> tuple[pd.DataFrame, dict[str, int]]:
    panel = _read_parquet(registry.names_panel_cleaned, _PANEL_COLUMNS)
    scores = _read_parquet(registry.names_scores, _NAME_SCORE_COLUMNS)
    missions = _read_parquet(registry.predictions_full_parquet, _MISSION_SCORE_COLUMNS)
    _assert_unique(panel, ["EIN2"], "panel name frame")
    _assert_unique(scores, ["EIN2", "input_variant"], "name scores")
    _assert_unique(missions, ["EIN2"], "mission predictions")

    panel["EIN2"] = _normalize_ein2(panel["EIN2"])
    scores["EIN2"] = _normalize_ein2(scores["EIN2"])
    missions["EIN2"] = _normalize_ein2(missions["EIN2"])
    paired = scores.merge(
        panel.loc[panel["has_mission"].astype(bool)],
        on="EIN2",
        how="inner",
        validate="many_to_one",
    ).merge(missions, on="EIN2", how="inner", validate="many_to_one")
    _validate_paired_values(paired, require_mission_score=False)

    mission_scores = pd.to_numeric(paired["prob_calibrated"], errors="coerce")
    missing_mission_score = mission_scores.isna()
    missing_mission_score_count = int(
        paired.loc[missing_mission_score, "EIN2"].nunique()
    )
    paired = paired.loc[~missing_mission_score].copy()
    _validate_paired_values(paired)
    _assert_complete_variant_pair(paired)

    before_exclusion = int(paired["EIN2"].nunique())
    contaminated = paired["is_manifest_contaminated"].astype(bool)
    excluded_count = int(paired.loc[contaminated, "EIN2"].nunique())
    paired = paired.loc[~contaminated].copy()
    return paired, {
        "n_paired_before_contamination_exclusion": before_exclusion,
        "n_contaminated_excluded": excluded_count,
        "n_evaluated": int(paired["EIN2"].nunique()),
        "n_missing_mission_score_excluded": missing_mission_score_count,
    }


def _assert_complete_variant_pair(paired: pd.DataFrame) -> None:
    variants = set(paired["input_variant"].astype(str))
    if variants != _INPUT_VARIANTS:
        raise ValueError(
            "Name scores must contain both required input variants: "
            "suffix_stripped and suffix_retaining."
        )
    ein2_by_variant = {
        variant: set(group["EIN2"].astype(str))
        for variant, group in paired.groupby("input_variant")
    }
    if len(set(map(frozenset, ein2_by_variant.values()))) != 1:
        raise ValueError(
            "Name-score variants must cover the same paired organizations."
        )


def _variant_report(cfg: BinaryClassifierConfig, frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.reset_index(drop=True)
    y_true = frame["pred_label"].astype(int).to_numpy()
    name_scores = frame["prob_raw"].astype(float).to_numpy()
    mission_scores = frame["prob_calibrated"].astype(float).to_numpy()
    score_difference = name_scores - mission_scores
    lexicon = frame["lexicon_rule_label"].fillna(0).astype(int).to_numpy()
    transfer = _top_k_predictions(frame, int(lexicon.sum()))
    transfer_metrics = metrics.compute_metric_bundle(
        y_true,
        transfer,
        y_score=name_scores,
        seed=int(cfg.SEED),
        n_resamples=int(cfg.evaluation.bootstrap_resamples),
    )
    lexicon_metrics = metrics.compute_metric_bundle(
        y_true,
        lexicon,
        seed=int(cfg.SEED),
        n_resamples=int(cfg.evaluation.bootstrap_resamples),
    )
    return {
        "matched_operating_point": {
            "method": "top_k_by_raw_score",
            "positive_count": int(lexicon.sum()),
            "positive_rate": float(lexicon.mean()),
            "tie_breaker": "EIN2 ascending",
        },
        "transfer_metrics": transfer_metrics,
        "lexicon_metrics": lexicon_metrics,
        "hypothesis": {
            "transfer_beats_lexicon_on_precision": (
                transfer_metrics["precision"] > lexicon_metrics["precision"]
            ),
            "decision_rule": "transfer_precision > lexicon_precision",
        },
        "agreement": {
            "score_pearson_correlation": _pearson(
                frame["prob_raw"], frame["prob_calibrated"]
            ),
            "score_mean_absolute_difference": float(np.mean(np.abs(score_difference))),
            "matched_label_agreement": float(np.mean(transfer == y_true)),
        },
        "calibration_against_mission_labels": calibration_metrics(
            y_true,
            name_scores,
            bins=int(cfg.evaluation.ece_bins),
        ),
        "calibration_against_mission_scores": {
            "mean_absolute_error": float(np.mean(np.abs(score_difference))),
            "mean_squared_error": float(np.mean(np.square(score_difference))),
        },
    }


def _top_k_predictions(frame: pd.DataFrame, positive_count: int) -> np.ndarray:
    predictions = np.zeros(len(frame), dtype=int)
    ranked = frame.sort_values(
        ["prob_raw", "EIN2"], ascending=[False, True], kind="stable"
    )
    predictions[ranked.index.to_numpy()[:positive_count]] = 1
    return predictions


def _pearson(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    return float(left.astype(float).corr(right.astype(float)))


def _read_parquet(path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame.copy()


def _assert_unique(frame: pd.DataFrame, columns: list[str], description: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(
            f"{description} contains duplicate {', '.join(columns)} values."
        )


def _normalize_ein2(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError("Paired validation inputs contain missing EIN2 values.")
    return normalized


def _validate_paired_values(
    frame: pd.DataFrame,
    *,
    require_mission_score: bool = True,
) -> None:
    if frame.empty:
        raise ValueError("No organizations have both usable name and mission scores.")
    labels = pd.to_numeric(frame["pred_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Mission pred_label values must be binary 0/1.")
    columns = ["prob_raw"]
    if require_mission_score:
        columns.append("prob_calibrated")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise ValueError(f"{column} values must be finite probabilities in [0, 1].")
