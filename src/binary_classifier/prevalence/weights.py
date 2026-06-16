"""Design-weight and EIN2 alignment utilities for prevalence estimation."""

import logging
from collections.abc import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def design_weights(
    anchor_manifest_df: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.Series:
    """Compute inverse-probability design weights for labeled anchor rows.

    Args:
        anchor_manifest_df: Anchor manifest containing a numeric,
            strictly-positive ``sample_prob`` column.
        normalize: When ``True``, divide inverse-probability weights by their
            mean so the labeled-set mean weight is 1.

    Returns:
        A float series named ``design_weight`` with the same index and row order
        as ``anchor_manifest_df``.

    Raises:
        ValueError: If ``sample_prob`` is missing, null, non-numeric, or not
            strictly positive.

    """
    if "sample_prob" not in anchor_manifest_df.columns:
        raise ValueError("anchor manifest must contain a sample_prob column")

    sample_prob = pd.to_numeric(anchor_manifest_df["sample_prob"], errors="coerce")
    if sample_prob.isna().any():
        raise ValueError("sample_prob must be non-null and numeric")
    if (sample_prob <= 0).any():
        raise ValueError("sample_prob must be strictly positive")

    weights = 1.0 / sample_prob
    if normalize and not weights.empty:
        weights = weights / weights.mean()

    weights.name = "design_weight"
    return weights


def align_labels_predictions(
    labeled_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    label_col: str = "human_label",
    prediction_cols: str | Iterable[str] = "prediction",
    ein_col: str = "EIN2",
    weight_col: str = "design_weight",
    normalize_weights: bool = True,
) -> pd.DataFrame:
    """Align labels, predictions, and design weights by normalized EIN2.

    EIN2 values are converted to strings and stripped on both inputs before
    joining. Rows with missing/blank EIN2 are dropped with a warning that
    includes the dropped-row count. Labeled rows without predictions are also
    dropped with a warning because prevalence estimators need complete
    label/prediction pairs.

    Args:
        labeled_df: Labeled anchor rows containing ``EIN2``, ``sample_prob``,
            and ``label_col``.
        predictions_df: Prediction rows containing ``EIN2`` and the requested
            prediction columns.
        label_col: Name of the human-label column in ``labeled_df``.
        prediction_cols: One prediction column name, or an iterable of column
            names, to carry through from ``predictions_df``.
        ein_col: Name of the EIN2 join-key column in both frames.
        weight_col: Name of the output design-weight column.
        normalize_weights: Whether to normalize inverse-probability weights to
            mean 1 over retained labeled rows.

    Returns:
        A dataframe ordered like ``labeled_df`` with normalized ``EIN2``, the
        label column, requested prediction columns, and the design-weight column.

    Raises:
        ValueError: If required columns are missing or normalized EIN2 values
            are duplicated in either input.

    """
    pred_cols = _prediction_columns(prediction_cols)
    _require_columns(labeled_df, [ein_col, "sample_prob", label_col], "labeled_df")
    _require_columns(predictions_df, [ein_col, *pred_cols], "predictions_df")

    labeled = _drop_missing_ein2(labeled_df, ein_col, "labeled_df").copy()
    predictions = _drop_missing_ein2(predictions_df, ein_col, "predictions_df").copy()
    labeled["_ein2_norm"] = _normalize_ein2(labeled[ein_col])
    predictions["_ein2_norm"] = _normalize_ein2(predictions[ein_col])

    _raise_on_duplicate_ein2(labeled, "labeled_df")
    _raise_on_duplicate_ein2(predictions, "predictions_df")

    labeled["_row_order"] = range(len(labeled))

    aligned = labeled.merge(
        predictions[["_ein2_norm", *pred_cols]],
        on="_ein2_norm",
        how="inner",
        sort=False,
    )
    dropped_without_prediction = len(labeled) - len(aligned)
    if dropped_without_prediction:
        logger.warning(
            "Dropped %d labeled rows without matching predictions by EIN2",
            dropped_without_prediction,
        )

    aligned = aligned.sort_values("_row_order", kind="stable")
    weights = design_weights(aligned, normalize=normalize_weights)
    aligned[weight_col] = weights.to_numpy()
    aligned[ein_col] = aligned["_ein2_norm"]
    return aligned[[ein_col, label_col, *pred_cols, weight_col]].reset_index(drop=True)


def _prediction_columns(prediction_cols: str | Iterable[str]) -> list[str]:
    """Normalize prediction column input to a non-empty list."""
    if isinstance(prediction_cols, str):
        return [prediction_cols]
    cols = list(prediction_cols)
    if not cols:
        raise ValueError("prediction_cols must contain at least one column")
    return cols


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    frame_name: str,
) -> None:
    """Validate that a dataframe contains required columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{frame_name} missing required columns: {joined}")


def _normalize_ein2(ein2: pd.Series) -> pd.Series:
    """String-normalize EIN2 values for artifact comparisons and joins."""
    return ein2.astype(str).str.strip()


def _drop_missing_ein2(
    frame: pd.DataFrame,
    ein_col: str,
    frame_name: str,
) -> pd.DataFrame:
    """Drop null or blank EIN2 values and warn with the dropped-row count."""
    normalized = _normalize_ein2(frame[ein_col])
    missing = frame[ein_col].isna() | normalized.eq("")
    dropped = int(missing.sum())
    if dropped:
        logger.warning("Dropped %d rows with missing EIN2 from %s", dropped, frame_name)
    return frame.loc[~missing].copy()


def _raise_on_duplicate_ein2(frame: pd.DataFrame, frame_name: str) -> None:
    """Reject duplicate normalized EIN2 keys to avoid accidental many-to-many joins."""
    duplicates = int(frame["_ein2_norm"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{frame_name} contains {duplicates} duplicate EIN2 values")
