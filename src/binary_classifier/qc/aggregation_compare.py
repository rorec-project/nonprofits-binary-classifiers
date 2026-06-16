"""Stage 11 aggregation-method comparison report.

The comparison is deliberately diagnostic: majority vote remains the production
default unless a human changes configuration and reruns the freeze/training
stages. This module scores majority vote plus configured comparison arms on the
human-coded validation split and writes a report with the standing adoption rule.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from binary_classifier.annotate.aggregate import aggregate_labels
from binary_classifier.annotate.schema import AnnotationStore
from binary_classifier.metrics import compute_metric_bundle

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

_ADOPTION_RULE = (
    "A non-majority arm may replace majority only if its bootstrap minority-F1 "
    "confidence-interval lower bound is strictly greater than majority's "
    "minority-F1 point estimate."
)
_RERUN_MESSAGE = (
    "Adopting any non-majority aggregation arm invalidates the frozen "
    "silver_labels.csv artifact; re-run stages 04→06 before using the new labels."
)


def run_aggregation_compare(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
) -> None:
    """Compare aggregation arms on the human validation split.

    Args:
        cfg: Validated binary-classifier configuration.
        registry: Path registry containing the annotation store, OOF
            probabilities, human coding template, and output report path.

    Returns:
        None. The comparison report is written to
        ``registry.aggregation_compare``.

    Raises:
        FileNotFoundError: If required input artifacts are missing.
        ValueError: If artifacts have invalid schema or no validation overlap.

    """
    store_df = _load_annotation_store(registry.annotation_store)
    validation_df = _load_validation_labels(registry.gold_coding_template)
    methods = _comparison_methods(cfg.aggregation.comparison_arms)
    pred_probs = (
        _load_oof_pred_probs(registry.oof_pred_probs) if "crowdlab" in methods else None
    )

    arms: dict[str, dict[str, Any]] = {}
    for method in methods:
        logger.info("Scoring aggregation arm: %s", method)
        aggregated = aggregate_labels(
            store_df,
            method=method,
            pred_probs=pred_probs if method == "crowdlab" else None,
        )
        arms[method] = _score_arm(
            method,
            aggregated,
            validation_df,
            seed=int(cfg.SEED),
            n_resamples=int(cfg.evaluation.bootstrap_resamples),
        )

    adoption = _adoption_verdicts(arms)
    report = {
        "metadata": {
            "stage": "11_aggregation_compare",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "seed": int(cfg.SEED),
            "bootstrap_resamples": int(cfg.evaluation.bootstrap_resamples),
            "annotation_store": str(registry.annotation_store),
            "oof_pred_probs": str(registry.oof_pred_probs)
            if pred_probs is not None
            else None,
            "gold_coding_template": str(registry.gold_coding_template),
            "validation_split": "validation",
            "production_method": cfg.aggregation.method,
            "comparison_arms": list(cfg.aggregation.comparison_arms),
            "adoption_rule": _ADOPTION_RULE,
            "rerun_required_if_adopted": _RERUN_MESSAGE,
        },
        "arms": arms,
        "adoption": adoption,
    }

    registry.aggregation_compare.parent.mkdir(parents=True, exist_ok=True)
    registry.aggregation_compare.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Aggregation comparison report written to %s", registry.aggregation_compare
    )


def _load_annotation_store(path: Path) -> pd.DataFrame:
    """Load and validate the long/tidy annotation store.

    Args:
        path: CSV or Parquet annotation store path.

    Returns:
        Normalized store DataFrame.

    Raises:
        ValueError: If the store is empty or lacks required columns.

    """
    df = AnnotationStore(path).to_frame()
    if df.empty:
        raise ValueError(f"Annotation store at {path} is empty. Run stage 03 first.")
    required = {"EIN2", "source_id", "label", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}.")

    work = df.copy()
    work["EIN2"] = _normalize_ein2(work["EIN2"])
    work["source_id"] = work["source_id"].astype(str).str.strip()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce")
    return work


def _load_oof_pred_probs(path: Path) -> pd.DataFrame:
    """Load out-of-fold class probabilities for CROWDLAB.

    Args:
        path: Parquet file with ``EIN2``, ``p0``, and ``p1`` columns.

    Returns:
        Normalized probability DataFrame.

    Raises:
        FileNotFoundError: If the OOF artifact is missing.
        ValueError: If required columns are missing or probabilities are invalid.

    """
    if not path.exists():
        raise FileNotFoundError(
            f"CROWDLAB comparison requires OOF probabilities at {path}. Run stage 06 first.",
        )
    probs = pd.read_parquet(path)
    required = {"EIN2", "p0", "p1"}
    missing = required - set(probs.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}.")
    work = probs.copy()
    work["EIN2"] = _normalize_ein2(work["EIN2"])
    work["p0"] = pd.to_numeric(work["p0"], errors="coerce")
    work["p1"] = pd.to_numeric(work["p1"], errors="coerce")
    if work[["p0", "p1"]].isna().any().any():
        raise ValueError(f"{path} contains non-numeric or missing p0/p1 values.")
    return work[["EIN2", "p0", "p1"]]


def _load_validation_labels(path: Path) -> pd.DataFrame:
    """Load strict 0/1 human labels for the validation split.

    Args:
        path: Human coding template, normally ``gold_to_code.csv``.

    Returns:
        DataFrame with normalized ``EIN2`` and integer ``human_label``.

    Raises:
        FileNotFoundError: If the template is missing.
        ValueError: If required columns are missing or validation labels are not
            complete strict 0/1 labels.

    """
    if not path.exists():
        raise FileNotFoundError(
            f"No human coding template at {path}. Code validation labels before stage 11.",
        )
    df = pd.read_csv(path)
    required = {"EIN2", "split", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}.")

    split = df["split"].astype(str).str.strip()
    sub = df.loc[split == "validation", ["EIN2", "human_label"]].copy()
    if sub.empty:
        raise ValueError(f"No validation rows found in {path}.")
    labels = pd.to_numeric(sub["human_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(
            f"Validation human_label values in {path} must be complete strict 0/1 labels.",
        )
    sub["EIN2"] = _normalize_ein2(sub["EIN2"])
    sub["human_label"] = labels.astype(int)
    return sub.drop_duplicates(subset=["EIN2"], keep="last")


def _comparison_methods(comparison_arms: list[str]) -> list[str]:
    """Return majority plus de-duplicated configured comparison methods."""
    methods: list[str] = []
    for method in ["majority", *comparison_arms]:
        if method not in methods:
            methods.append(method)
    return methods


def _score_arm(
    method: str,
    aggregated: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    """Score one aggregated arm against validation labels.

    Args:
        method: Aggregation method name.
        aggregated: Wide aggregate output with ``EIN2`` and ``silver_label``.
        validation_df: Strict validation human labels.
        seed: Bootstrap seed.
        n_resamples: Number of bootstrap resamples.

    Returns:
        Report entry for the arm.

    Raises:
        ValueError: If required aggregate columns are absent or no rows score.

    """
    required = {"EIN2", "silver_label", "silver_confidence"}
    missing = required - set(aggregated.columns)
    if missing:
        raise ValueError(
            f"Aggregation arm {method} missing columns: {sorted(missing)}."
        )

    aggregate_norm = aggregated.copy()
    aggregate_norm["EIN2"] = _normalize_ein2(aggregate_norm["EIN2"])
    merged = aggregate_norm.merge(validation_df, on="EIN2", how="inner")
    valid = merged.dropna(subset=["silver_label", "human_label"]).copy()
    if valid.empty:
        raise ValueError(
            f"Aggregation arm {method} has no scored validation overlap. Check EIN2s and abstains.",
        )

    y_true = valid["human_label"].astype(int).to_numpy()
    y_pred = valid["silver_label"].astype(int).to_numpy()
    y_score = None
    if valid["silver_confidence"].notna().any():
        y_score = valid["silver_confidence"].astype(float).to_numpy()
    metrics = compute_metric_bundle(
        y_true,
        y_pred,
        y_score=y_score,
        minority_class=_minority_class(y_true),
        seed=seed,
        n_resamples=n_resamples,
    )
    return {
        "method": method,
        "n_aggregated": int(len(aggregate_norm)),
        "n_validation_labels": int(len(validation_df)),
        "n_validation_overlap": int(len(merged)),
        "n_scored": int(len(valid)),
        "n_abstain_validation": int(merged["silver_label"].isna().sum()),
        "scored_ein2": valid["EIN2"].astype(str).tolist(),
        "metrics": metrics,
    }


def _adoption_verdicts(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the standing adoption rule to non-majority arms.

    Args:
        arms: Per-arm score entries including majority.

    Returns:
        Adoption verdict bundle.

    """
    majority_f1 = float(arms["majority"]["metrics"]["f1"])
    verdicts: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []

    for method, entry in arms.items():
        if method == "majority":
            continue
        ci_lower = float(entry["metrics"]["bootstrap_ci"]["minority_f1"]["lower"])
        may_replace = bool(np.isfinite(ci_lower) and ci_lower > majority_f1)
        if may_replace:
            eligible.append(method)
        verdicts[method] = {
            "may_replace_majority": may_replace,
            "minority_f1_ci_lower": ci_lower,
            "majority_minority_f1_point": majority_f1,
            "rule": _ADOPTION_RULE,
            "requires_rerun": _RERUN_MESSAGE,
        }

    recommended = None
    if eligible:
        recommended = max(
            eligible,
            key=lambda arm: arms[arm]["metrics"]["bootstrap_ci"]["minority_f1"][
                "lower"
            ],
        )
    return {
        "majority_minority_f1_point": majority_f1,
        "eligible_arms": eligible,
        "recommended_arm": recommended,
        "verdicts": verdicts,
        "rule": _ADOPTION_RULE,
        "message": _RERUN_MESSAGE,
    }


def _minority_class(y_true: np.ndarray) -> int:
    """Return the validation minority class, choosing positive class on ties."""
    counts = np.bincount(y_true.astype(int), minlength=2)
    if counts[1] <= counts[0]:
        return 1
    return 0


def _normalize_ein2(values: pd.Series) -> pd.Series:
    """Normalize EIN2 values for cross-artifact joins."""
    return values.astype(str).str.strip()


def _json_ready(value: Any) -> Any:
    """Convert NumPy/pandas scalars and non-finite floats for JSON output."""
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value
