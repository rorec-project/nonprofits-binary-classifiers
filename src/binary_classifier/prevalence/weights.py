"""Design-weight and EIN2 alignment utilities for prevalence estimation.

This module implements the **sampling-design correction** for the anchor
(labeled audit) sample.  Because the anchor is drawn with unequal
probabilities (e.g. stratified by quality tier and NTEE major group), the
standard PPI estimator would be biased if applied without weighting.  The
design-weights function computes inverse-probability weights
(Horvitz & Thompson, 1952, https://doi.org/10.1080/01621459.1952.10483446)
so that the weighted PPI estimator is unbiased for the population mean.

The PPI mean estimator is algebraically equivalent to the classical survey-
sampling **difference estimator** (Cassel et al., 1976), and PPI++ corresponds
to the generalized regression (GREG) estimator (Mozer, 2026,
arXiv:2603.19160, https://doi.org/10.48550/arXiv.2603.19160).  Passing
inverse-probability weights to ``ppi_py`` preserves the Horvitz--Thompson
property that the weighted mean of the labeled residuals is an unbiased
estimate of the population residual.

The ``align_labels_predictions`` helper joins labeled anchor rows to scored
predictions by the normalized ``EIN2`` join key, dropping mismatches and
duplicates with explicit warnings so that prevalence estimation never proceeds
on silently incomplete data.

References
----------
* Horvitz, D. G., & Thompson, D. J. (1952). A Generalization of Sampling
  Without Replacement from a Finite Universe. *Journal of the American
  Statistical Association*, 47(260), 663--685.
  https://doi.org/10.1080/01621459.1952.10483446
* Mozer, R. (2026). PPI is the Difference Estimator: Recognizing the Survey
  Sampling Roots of Prediction-Powered Inference. arXiv:2603.19160.
  https://doi.org/10.48550/arXiv.2603.19160

"""

import logging
from collections.abc import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inverse-probability design weights
# ---------------------------------------------------------------------------
# The anchor sample is drawn with unequal probabilities (e.g. stratified by
# tier and NTEE major group).  Standard PPI would be biased without weighting.
# We compute Horvitz--Thompson inverse-probability weights and normalize them
# to mean one so that the weighted PPI estimator is unbiased for the population
# mean (Horvitz & Thompson, 1952; Mozer, 2026).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# EIN2 alignment
# ---------------------------------------------------------------------------
# Aligns labels, predictions, and design weights by normalized EIN2.  Missing
# or blank EIN2 values are dropped with a warning.  Duplicate normalized EIN2
# keys are rejected to prevent accidental many-to-many joins.  Labeled rows
# without matching predictions are also dropped with a warning because
# prevalence estimators need complete label/prediction pairs.
# ---------------------------------------------------------------------------


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
    joining.  Rows with missing/blank EIN2 are dropped with a warning that
    includes the dropped-row count.  Labeled rows without predictions are also
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
