"""Paired validation of mission-to-name cross-field transfer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from binary_classifier import metrics
from binary_classifier.evaluation.calibration import calibration_metrics
from binary_classifier.names.gold import require_name_gold_coding_complete
from binary_classifier.names.identifiers import normalize_ein2

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


_NAME_SCORE_COLUMNS = {
    "EIN2",
    "population",
    "input_variant",
    "prob_raw",
    "lexicon_rule_label",
}
_MISSION_SCORE_COLUMNS = {"EIN2", "pred_label", "prob_calibrated"}
_PANEL_COLUMNS = {"EIN2", "has_mission", "is_manifest_contaminated"}
_FLAG_COLUMN = "is_external_religious_flag"
_INPUT_VARIANTS = {"suffix_stripped", "suffix_retaining"}
_GOLD_SPLITS = {"prompt_dev", "validation"}
_SHIFT_FAILURE_INTERPRETATION = (
    "If positive rates are approximately flat despite the known external-flag "
    "base-rate shift, the model is responding to input shape rather than "
    "measuring religion; the names arm is falsified."
)
_CONSTRUCT_OFFSET_INTERPRETATION = (
    "External-flag results are auspice-aligned and must not be read as purpose "
    "prevalence."
)
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
    require_name_gold_coding_complete(registry)
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
    report["bmf_only_gold"] = _bmf_only_gold_report(cfg, registry)
    external = _external_flag_report(cfg, registry, paired)
    if external is not None:
        report["external_flag_validation"] = external
    registry.ensure_dirs()
    registry.names_validation.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def _bmf_only_gold_report(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> dict[str, Any]:
    """Report completed BMF-only gold accuracy with sampling-cell weights."""
    template = pd.read_csv(
        registry.names_gold_coding_template, dtype={"EIN2": "string"}
    )
    manifest = pd.read_csv(registry.names_gold_manifest, dtype={"EIN2": "string"})
    required_manifest = {
        "EIN2",
        "population",
        "inclusion_probability",
        "sampling_cell",
    }
    missing = sorted(required_manifest.difference(manifest.columns))
    if missing:
        raise ValueError(
            "Names gold manifest is missing required columns: " + ", ".join(missing)
        )
    gold = template[["EIN2", "human_label"]].merge(
        manifest[["EIN2", "population", "inclusion_probability", "sampling_cell"]],
        on="EIN2",
        how="inner",
        validate="one_to_one",
    )
    if not gold["population"].eq("bmf_only").all():
        raise ValueError(
            "Names gold manifest must contain only BMF-only organizations."
        )
    gold["EIN2"] = normalize_ein2(gold["EIN2"])
    inclusion_probability = pd.to_numeric(
        gold["inclusion_probability"], errors="coerce"
    )
    if (
        inclusion_probability.isna().any()
        or ~np.isfinite(inclusion_probability).all()
        or ~(inclusion_probability.gt(0) & inclusion_probability.le(1)).all()
    ):
        raise ValueError(
            "Names gold inclusion probabilities must be finite values in (0, 1]."
        )
    gold["inclusion_probability"] = inclusion_probability
    scores = _read_parquet(registry.names_scores, _NAME_SCORE_COLUMNS)
    scores["EIN2"] = normalize_ein2(scores["EIN2"])
    if scores.duplicated(["EIN2", "input_variant"]).any():
        raise ValueError("Name scores must contain one row per EIN2 and input variant.")
    gold_scores = gold.merge(scores, on="EIN2", how="inner", validate="one_to_many")
    expected_pairs = {
        (ein2, variant) for ein2 in gold["EIN2"] for variant in _INPUT_VARIANTS
    }
    observed_pairs = set(
        zip(gold_scores["EIN2"], gold_scores["input_variant"], strict=True)
    )
    if observed_pairs != expected_pairs:
        raise ValueError(
            "Name scores must cover both variants for every BMF-only gold row."
        )
    variants: dict[str, Any] = {}
    for variant, frame in gold_scores.groupby("input_variant", sort=True):
        frame = frame.sort_values("EIN2").reset_index(drop=True)
        y_true = frame["human_label"].astype(int).to_numpy()
        prediction = (
            frame["prob_raw"].astype(float).to_numpy()
            >= float(cfg.names.diagnostic_threshold)
        ).astype(int)
        weights = 1 / frame["inclusion_probability"].astype(float).to_numpy()
        variants[str(variant)] = {
            "diagnostic_threshold": float(cfg.names.diagnostic_threshold),
            "unweighted_metrics": metrics.compute_metric_bundle(
                y_true,
                prediction,
                y_score=frame["prob_raw"].astype(float).to_numpy(),
                seed=int(cfg.SEED),
                n_resamples=int(cfg.evaluation.bootstrap_resamples),
            ),
            "design_weighted_accuracy": float(
                np.average(prediction == y_true, weights=weights)
            ),
        }
    return {
        "target_population": "bmf_only",
        "reference": "completed BMF-only names gold labels",
        "design": "sampling-cell weighted stratified conflict-enriched sample",
        "n_gold": len(gold),
        "variants": variants,
    }


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


def _external_flag_report(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    paired: pd.DataFrame,
) -> dict[str, Any] | None:
    """Report the names-arm external-label diagnostic without changing paired use."""
    if not registry.names_bmf_only_cleaned.exists():
        panel = pd.read_parquet(registry.names_panel_cleaned)
        if _FLAG_COLUMN not in panel:
            return None
        raise ValueError("External-flag validation requires the BMF-only name frame.")
    panel = _read_parquet(
        registry.names_panel_cleaned,
        {"EIN2", _FLAG_COLUMN},
    )
    bmf_only = _read_parquet(
        registry.names_bmf_only_cleaned,
        {"EIN2", _FLAG_COLUMN},
    )
    flags = pd.concat(
        [
            panel.assign(population="panel_501c3"),
            bmf_only.assign(population="bmf_only"),
        ],
        ignore_index=True,
    )[["EIN2", "population", _FLAG_COLUMN]]
    flags["EIN2"] = _normalize_ein2(flags["EIN2"])
    _assert_unique(flags, ["EIN2"], "external flag frame")
    scores = _read_parquet(
        registry.names_scores,
        {"EIN2", "input_variant", "prob_raw", "population"},
    )[["EIN2", "input_variant", "prob_raw", "population"]]
    scores["EIN2"] = _normalize_ein2(scores["EIN2"])
    _assert_unique(scores, ["EIN2", "input_variant"], "name scores")
    scores = scores.merge(
        flags,
        on="EIN2",
        how="inner",
        validate="many_to_one",
        suffixes=("_score", "_frame"),
    )
    if not scores["population_score"].eq(scores["population_frame"]).all():
        raise ValueError("Name-score populations do not match the cleaned name frames.")
    scores = scores.rename(columns={"population_score": "population"})
    _validate_external_scores(scores)
    threshold = float(cfg.names.diagnostic_threshold)
    populations: dict[str, dict[str, Any]] = {}
    for (population, variant), frame in scores.groupby(
        ["population", "input_variant"], sort=True
    ):
        flag = frame[_FLAG_COLUMN].astype(int).to_numpy()
        score = frame["prob_raw"].astype(float).to_numpy()
        prediction = (score >= threshold).astype(int)
        populations.setdefault(str(population), {})[str(variant)] = {
            "n": len(frame),
            "diagnostic_threshold": threshold,
            "flag_base_rate": float(flag.mean()),
            "model_positive_rate": float(prediction.mean()),
            "metrics": metrics.compute_metric_bundle(
                flag,
                prediction,
                y_score=score,
                seed=int(cfg.SEED),
                n_resamples=int(cfg.evaluation.bootstrap_resamples),
            ),
        }

    gold_offset = _gold_construct_offset(cfg, registry, flags, paired)
    shift = _base_rate_shift(
        cfg,
        populations["panel_501c3"]["suffix_stripped"],
        populations["bmf_only"]["suffix_stripped"],
    )
    if gold_offset is not None:
        populations["bmf_only"]["gold_construct_offset"] = {
            "offset": gold_offset["offset"],
            "correction_applied": False,
        }
    return {
        "populations": populations,
        "construct_offset": gold_offset,
        "base_rate_shift": shift,
        "limitations": [
            "The external flag measures religious auspice, not observable religious "
            "purpose. Its measured construct offset is reported without numerical "
            "correction."
        ],
    }


def _gold_construct_offset(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    flags: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, Any] | None:
    """Quantify auspice-versus-purpose disagreement on the clean gold overlap."""
    if not registry.gold_coding_template.exists():
        return None
    gold = pd.read_csv(registry.gold_coding_template)
    required = {"EIN2", "split", "human_label"}
    missing = sorted(required.difference(gold.columns))
    if missing:
        raise ValueError(
            f"{registry.gold_coding_template} is missing required columns: "
            f"{', '.join(missing)}"
        )
    gold = gold.loc[gold["split"].astype(str).isin(_GOLD_SPLITS)].copy()
    gold["EIN2"] = _normalize_ein2(gold["EIN2"])
    gold["human_label"] = pd.to_numeric(gold["human_label"], errors="coerce")
    overlap = gold.merge(
        flags.loc[flags["population"].eq("panel_501c3")],
        on="EIN2",
        how="inner",
        validate="one_to_one",
    )
    overlap = overlap.merge(
        paired[["EIN2", "pred_label"]].drop_duplicates("EIN2"),
        on="EIN2",
        how="inner",
        validate="one_to_one",
    )
    if overlap.empty or overlap["human_label"].isna().any():
        return None
    flag_rate = float(overlap[_FLAG_COLUMN].astype(int).mean())
    human_rate = float(overlap["human_label"].astype(int).mean())
    flag_metrics = metrics.compute_metric_bundle(
        overlap["human_label"].astype(int).to_numpy(),
        overlap[_FLAG_COLUMN].astype(int).to_numpy(),
        seed=int(cfg.SEED),
        n_resamples=int(cfg.evaluation.bootstrap_resamples),
    )
    return {
        "source_population": "uncontaminated panel prompt_dev + validation overlap",
        "flag_construct": "IRS religious auspice",
        "target_construct": "observable religious purpose",
        "n_overlap": len(overlap),
        "flag_rate": flag_rate,
        "human_purpose_rate": human_rate,
        "offset": flag_rate - human_rate,
        "mission_label_rate": float(overlap["pred_label"].astype(int).mean()),
        "flag_vs_human_purpose_metrics": flag_metrics,
        "correction_applied": False,
        "interpretation": _CONSTRUCT_OFFSET_INTERPRETATION,
    }


def _base_rate_shift(
    cfg: BinaryClassifierConfig,
    panel: dict[str, Any],
    bmf_only: dict[str, Any],
) -> dict[str, Any]:
    """Falsify transfer when BMF-only scores fail to follow the known flag shift."""
    panel_flag_rate = float(panel["flag_base_rate"])
    bmf_flag_rate = float(bmf_only["flag_base_rate"])
    panel_model_rate = float(panel["model_positive_rate"])
    bmf_model_rate = float(bmf_only["model_positive_rate"])
    flag_ratio = _rate_ratio(bmf_flag_rate, panel_flag_rate)
    model_ratio = _rate_ratio(bmf_model_rate, panel_model_rate)
    relative_error = (
        abs(model_ratio / flag_ratio - 1.0)
        if flag_ratio is not None and model_ratio is not None
        else None
    )
    tolerance = float(cfg.names.base_rate_shift_ratio_tolerance)
    target_rate_higher = bmf_model_rate > panel_model_rate
    return {
        "flag_rate_ratio": flag_ratio,
        "model_positive_rate_ratio": model_ratio,
        "relative_ratio_error": relative_error,
        "ratio_tolerance": tolerance,
        "absolute_flag_rate_difference": abs(bmf_flag_rate - panel_flag_rate),
        "absolute_model_positive_rate_difference": abs(
            bmf_model_rate - panel_model_rate
        ),
        "target_model_positive_rate_higher": target_rate_higher,
        "verdict": (
            "PASS"
            if target_rate_higher
            and relative_error is not None
            and relative_error <= tolerance
            else "FAIL"
        ),
        "interpretation": _SHIFT_FAILURE_INTERPRETATION,
        "failure_interpretation": _SHIFT_FAILURE_INTERPRETATION,
    }


def _rate_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _validate_external_scores(frame: pd.DataFrame) -> None:
    """Validate external-label inputs before calculating diagnostic metrics."""
    populations = set(frame["population"].astype(str))
    required_populations = {"panel_501c3", "bmf_only"}
    if populations != required_populations:
        raise ValueError(
            "External-flag validation requires scores for panel_501c3 and bmf_only."
        )
    variants_by_population = {
        population: set(group["input_variant"].astype(str))
        for population, group in frame.groupby("population")
    }
    if any(variants != _INPUT_VARIANTS for variants in variants_by_population.values()):
        raise ValueError(
            "External-flag validation requires both name-score variants in each population."
        )
    flags = pd.to_numeric(frame[_FLAG_COLUMN], errors="coerce")
    if flags.isna().any() or not flags.isin([0, 1]).all():
        raise ValueError("External religious flags must be binary 0/1.")
    scores = pd.to_numeric(frame["prob_raw"], errors="coerce")
    if scores.isna().any() or not scores.between(0.0, 1.0).all():
        raise ValueError("External-validation prob_raw values must be in [0, 1].")


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
    normalized = normalize_ein2(series)
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
