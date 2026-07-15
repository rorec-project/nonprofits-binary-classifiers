"""Stage 11 aggregation-method sensitivity/diagnostic report.

The comparison is deliberately diagnostic: stage 04 production labels are always
frozen by majority vote. This module scores majority vote plus configured
Dawid-Skene/CROWDLAB comparison arms on the human-coded validation split, but it
does not continue production or recommend an automatic replacement.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
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

_SENSITIVITY_RULE = (
    "A non-majority diagnostic arm clears the sensitivity screen only if its "
    "bootstrap minority-F1 confidence-interval lower bound is strictly greater "
    "than majority's minority-F1 point estimate."
)
_PRODUCTION_NOTE = (
    "Diagnostic only: stage 04 intentionally freezes majority-vote labels. "
    "Dawid-Skene and CROWDLAB are stage-11 sensitivity arms, not production "
    "continuation methods. CROWDLAB is scored only when leakage-safe, "
    "validation-aligned classifier probabilities are available; otherwise it is "
    "reported as skipped."
)


def run_aggregation_compare(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> None:
    """Compare diagnostic aggregation arms on the human validation split.

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
    pred_probs: pd.DataFrame | None = None
    crowdlab_skip_reason: str | None = None
    if "crowdlab" in methods:
        try:
            pred_probs = _load_oof_pred_probs(registry.oof_pred_probs)
        except (FileNotFoundError, ValueError) as exc:
            crowdlab_skip_reason = str(exc)

    arms: dict[str, dict[str, Any]] = {}
    for method in methods:
        logger.info("Scoring aggregation arm: %s", method)
        if method == "crowdlab":
            skip_reason = crowdlab_skip_reason or _crowdlab_validation_skip_reason(
                store_df,
                validation_df,
                pred_probs,
                registry.oof_pred_probs,
            )
            if skip_reason is not None:
                logger.warning("Skipping CROWDLAB diagnostic arm: %s", skip_reason)
                arms[method] = _skipped_arm_entry(
                    method,
                    skip_reason,
                    validation_df,
                    pred_probs,
                )
                continue
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

    sensitivity = _sensitivity_verdicts(arms)
    report = {
        "metadata": {
            "stage": "11_aggregation_compare",
            "report_purpose": "sensitivity_diagnostic",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "seed": int(cfg.SEED),
            "bootstrap_resamples": int(cfg.evaluation.bootstrap_resamples),
            "annotation_store": str(registry.annotation_store),
            "oof_pred_probs": str(registry.oof_pred_probs)
            if pred_probs is not None
            else None,
            "gold_coding_template": str(registry.gold_coding_template),
            "validation_split": "validation",
            "production_method": "majority",
            "comparison_arms": list(cfg.aggregation.comparison_arms),
            "sensitivity_rule": _SENSITIVITY_RULE,
            "production_note": _PRODUCTION_NOTE,
        },
        "arms": arms,
        "sensitivity": sensitivity,
    }

    registry.aggregation_compare.parent.mkdir(parents=True, exist_ok=True)
    registry.aggregation_compare.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Aggregation comparison report written to %s",
        registry.aggregation_compare,
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


def _comparison_methods(comparison_arms: Sequence[str]) -> list[str]:
    """Return majority plus de-duplicated configured comparison methods."""
    methods: list[str] = []
    for method in ["majority", *comparison_arms]:
        if method not in methods:
            methods.append(method)
    return methods


def _crowdlab_validation_skip_reason(
    store_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    pred_probs: pd.DataFrame | None,
    pred_probs_path: Path,
) -> str | None:
    """Return why CROWDLAB cannot be safely scored on validation, if any."""
    if pred_probs is None:
        return f"CROWDLAB diagnostic requires OOF probabilities at {pred_probs_path}."

    validation_ein2 = set(validation_df["EIN2"].astype(str))
    annotated_validation = sorted(set(store_df["EIN2"].astype(str)) & validation_ein2)
    if not annotated_validation:
        return "CROWDLAB diagnostic has no validation EIN2s in the annotation store."

    prob_ein2 = set(pred_probs["EIN2"].astype(str))
    missing = [ein2 for ein2 in annotated_validation if ein2 not in prob_ein2]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        return (
            "CROWDLAB diagnostic skipped because OOF probabilities do not cover "
            f"all validation annotation EIN2s; missing {preview}{suffix}."
        )
    return None


def _skipped_arm_entry(
    method: str,
    reason: str,
    validation_df: pd.DataFrame,
    pred_probs: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build a report entry for a diagnostic arm skipped by preflight checks."""
    entry: dict[str, Any] = {
        "method": method,
        "status": "skipped",
        "skip_reason": reason,
        "n_validation_labels": int(len(validation_df)),
    }
    if pred_probs is not None:
        validation_ein2 = set(validation_df["EIN2"].astype(str))
        pred_ein2 = set(pred_probs["EIN2"].astype(str))
        entry["n_pred_probs"] = int(len(pred_ein2))
        entry["n_pred_probs_validation_overlap"] = int(len(validation_ein2 & pred_ein2))
    return entry


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
            f"Aggregation arm {method} missing columns: {sorted(missing)}.",
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
    y_score = _positive_class_score(valid)
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
        "status": "scored",
        "n_aggregated": int(len(aggregate_norm)),
        "n_validation_labels": int(len(validation_df)),
        "n_validation_overlap": int(len(merged)),
        "n_scored": int(len(valid)),
        "n_abstain_validation": int(merged["silver_label"].isna().sum()),
        "scored_ein2": valid["EIN2"].astype(str).tolist(),
        "metrics": metrics,
    }


def _positive_class_score(valid: pd.DataFrame) -> np.ndarray | None:
    """Convert winner confidence into a positive-class score for AUC metrics."""
    if not valid["silver_confidence"].notna().any():
        return None
    confidence = valid["silver_confidence"].astype(float)
    label = valid["silver_label"].astype(int)
    return np.where(label == 1, confidence, 1.0 - confidence)


def _sensitivity_verdicts(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the diagnostic sensitivity screen to non-majority arms.

    Args:
        arms: Per-arm score entries including majority.

    Returns:
        Sensitivity verdict bundle.

    """
    majority_f1 = float(arms["majority"]["metrics"]["f1"])
    verdicts: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []

    for method, entry in arms.items():
        if method == "majority":
            continue
        if entry.get("status") == "skipped":
            verdicts[method] = {
                "clears_majority_sensitivity_screen": False,
                "skip_reason": entry.get("skip_reason"),
                "rule": _SENSITIVITY_RULE,
                "production_note": _PRODUCTION_NOTE,
            }
            continue
        ci_lower = float(entry["metrics"]["bootstrap_ci"]["minority_f1"]["lower"])
        clears_screen = bool(np.isfinite(ci_lower) and ci_lower > majority_f1)
        if clears_screen:
            eligible.append(method)
        verdicts[method] = {
            "clears_majority_sensitivity_screen": clears_screen,
            "minority_f1_ci_lower": ci_lower,
            "majority_minority_f1_point": majority_f1,
            "rule": _SENSITIVITY_RULE,
            "production_note": _PRODUCTION_NOTE,
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
        "arms_clearing_screen": eligible,
        "best_diagnostic_arm": recommended,
        "verdicts": verdicts,
        "rule": _SENSITIVITY_RULE,
        "message": _PRODUCTION_NOTE,
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
