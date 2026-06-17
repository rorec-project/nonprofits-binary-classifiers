"""Gated training-arm data and loss specifications.

This module defines the training-data preparation arms for the binary
classifier pipeline.  Each arm produces a normalized training frame and a
:class:`LossSpec` that tells the encoder fine-tuning stage what target
semantics to use (soft vote-shares vs. hard majority labels) and whether
to apply class weighting.

Design rationale
----------------
The default active arms are ``hard`` and ``class_weighted``.  Soft
vote-shares are the default target because they already down-weight the
disagreement band (rows where the silver ensemble is split) without needing
to drop data (arXiv:2511.14117; arXiv:2605.20642; PMC12148080).  The
``pruned`` arm, which uses cleanlab to flag potential label issues in the
disagreement band, is kept as an opt-in diagnostic extra because the same
variance reduction can be achieved by keeping the soft targets and
inverse-frequency class weighting (King & Zeng 2001; "Balancing the Scales",
arXiv:2409.19751).  The pruned arm is therefore gated behind the optional
``diagnostics`` dependency and is only invoked when explicitly requested.

Citations
---------
- Soft labels / vote-share smoothing: arXiv:2511.14117, arXiv:2605.20642,
  PMC12148080.
- Inverse-frequency class weighting (rare-event bias): King & Zeng (2001);
  "Balancing the Scales" (arXiv:2409.19751).
- Confident learning / data-pruning (opt-in): Northcutt, Jiang & Chuang (2021).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ArmName = Literal["hard", "pruned", "class_weighted"]
TargetKind = Literal["soft", "hard"]

# Training frame schema required by every arm.
_TRAINING_COLUMNS = {"EIN2", "text", "ntee_major_group", "p_pos", "hard_label"}
# OOF probability frame schema required by the pruned arm.
_OOF_COLUMNS = {"EIN2", "p0", "p1"}
# Disagreement band: vote-shares strictly between these bounds are considered
# low-confidence.  Used only by the opt-in pruned arm.
_DISAGREEMENT_LOWER = 0.34
_DISAGREEMENT_UPPER = 0.66


@dataclass(frozen=True)
class LossSpec:
    """Loss configuration returned by a gated arm runner.

    Args:
        targets: Target semantics to pass to encoder fine-tuning.
        arm: Training-data arm name recorded in result rows.
        class_weights: Optional inverse-frequency weights ordered as ``(w0, w1)``.

    """

    targets: TargetKind
    arm: str
    class_weights: tuple[float, float] | None = None

    def finetune_kwargs(self) -> dict[str, str]:
        """Return keyword arguments consumed by encoder fine-tuning.

        Returns:
            Dict containing the ``targets`` and ``arm`` entries.

        """
        return {"targets": self.targets, "arm": self.arm}


def run_arm(
    arm: ArmName,
    train_df: pd.DataFrame,
    *,
    oof_probs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, LossSpec]:
    """Run one gated arm preparation step.

    Args:
        arm: Arm name: ``"hard"``, ``"pruned"``, or ``"class_weighted"``.
        train_df: Training frame with ``EIN2``, text, ``p_pos``, and
            ``hard_label`` columns.
        oof_probs: Out-of-fold probabilities required for the ``pruned`` arm.

    Returns:
        Pair of modified training frame and loss specification.

    Raises:
        ValueError: If the arm name is unknown or the required schemas are
            invalid.

    """
    if arm == "hard":
        return hard_arm(train_df)
    if arm == "pruned":
        if oof_probs is None:
            raise ValueError("pruned arm requires oof_probs.")
        return pruned_arm(train_df, oof_probs)
    if arm == "class_weighted":
        return class_weighted_arm(train_df)
    raise ValueError(f"Unknown gated arm: {arm!r}.")


def hard_arm(train_df: pd.DataFrame) -> tuple[pd.DataFrame, LossSpec]:
    """Return a hard-majority-vote training frame and loss spec.

    Binarizes the silver vote-share ``p_pos`` to the hard majority label.
    This arm is the simplest baseline: it ignores ensemble uncertainty and
    treats every row as a certain 0/1 label.

    Args:
        train_df: Training frame containing ``hard_label``.

    Returns:
        Frame with ``p_pos`` binarized to the hard labels, plus a hard-target
        loss specification.

    """
    work = _validated_training_frame(train_df)
    work["p_pos"] = work["hard_label"].astype(float)
    return work.reset_index(drop=True), LossSpec(targets="hard", arm="hard")


def pruned_arm(
    train_df: pd.DataFrame,
    oof_probs: pd.DataFrame,
) -> tuple[pd.DataFrame, LossSpec]:
    """Drop confident-learning issues that fall in the disagreement band.

    Cleanlab flags are computed from hard labels and OOF probabilities, then
    intersected with rows whose vote share lies strictly between 0.34 and 0.66.
    This arm is opt-in and requires the ``diagnostics`` optional dependency
    because the same variance reduction is achieved by the default
    ``class_weighted`` soft-target arm (King & Zeng 2001; "Balancing the Scales",
    arXiv:2409.19751; Northcutt, Jiang & Chuang 2021).

    Args:
        train_df: Training frame containing ``EIN2``, ``p_pos``, and
            ``hard_label``.
        oof_probs: OOF probability frame with one row per ``EIN2`` and columns
            ``p0`` and ``p1``.

    Returns:
        Pruned training frame and soft-target loss specification for the pruned
        arm.

    Raises:
        ValueError: If frame schemas are invalid or OOF rows do not match the
            training ``EIN2`` set exactly.

    """
    work = _validated_training_frame(train_df)
    aligned_oof = _aligned_oof_probs(work, oof_probs)
    drop_ein2 = prune_ein2s(work, aligned_oof)
    keep_mask = ~work["EIN2"].isin(drop_ein2)
    pruned = work.loc[keep_mask].reset_index(drop=True)
    logger.info(
        "Pruned %d/%d low-vote-share cleanlab issue rows",
        len(drop_ein2),
        len(work),
    )
    return pruned, LossSpec(targets="soft", arm="pruned")


def class_weighted_arm(train_df: pd.DataFrame) -> tuple[pd.DataFrame, LossSpec]:
    """Return an unchanged frame and inverse-frequency class weights.

    Inverse-frequency weights compensate for rare-class bias in the silver
    sample (King & Zeng 2001; "Balancing the Scales", arXiv:2409.19751).
    The frame itself is left intact so that soft vote-shares still carry
    ensemble uncertainty; only the loss function is re-weighted.

    Args:
        train_df: Training frame containing binary ``hard_label`` values.

    Returns:
        Schema-normalized frame and loss specification with ``(w0, w1)`` class
        weights.

    """
    work = _validated_training_frame(train_df)
    return work.reset_index(drop=True), LossSpec(
        targets="soft",
        arm="class_weighted",
        class_weights=class_weights(work),
    )


def class_weights(train_df: pd.DataFrame) -> tuple[float, float]:
    """Compute inverse-frequency class weights from hard labels.

    Weights follow the standard ``n / (2 * nk)`` formula so that the
    minority class receives a proportionally larger loss contribution.
    This mitigates the rare-class bias documented in King & Zeng (2001)
    and "Balancing the Scales" (arXiv:2409.19751).

    Args:
        train_df: Training frame containing binary ``hard_label`` values.

    Returns:
        ``(w0, w1)`` where ``wk = n / (2 * nk)``.

    Raises:
        ValueError: If either binary class is absent.

    """
    work = _validated_training_frame(train_df)
    counts = work["hard_label"].value_counts().to_dict()
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))
    if n0 == 0 or n1 == 0:
        raise ValueError("class_weighted arm requires both hard-label classes.")
    total = n0 + n1
    return total / (2 * n0), total / (2 * n1)


def prune_ein2s(train_df: pd.DataFrame, oof_probs: pd.DataFrame) -> set[str]:
    """Return ``EIN2`` values dropped by the pruned arm.

    Args:
        train_df: Valid or raw training frame.
        oof_probs: Valid or raw OOF probability frame.

    Returns:
        Set of cleanlab-flagged ``EIN2`` values whose ``p_pos`` lies in the
        open disagreement band ``(0.34, 0.66)``.

    """
    work = _validated_training_frame(train_df)
    aligned_oof = _aligned_oof_probs(work, oof_probs)
    labels = work["hard_label"].to_numpy(dtype=int)
    pred_probs = aligned_oof[["p0", "p1"]].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # cleanlab is an optional ``diagnostics`` extra; the pruned arm is
    # opt-in (Northcutt, Jiang & Chuang 2021).  We import locally so that
    # the module can be imported when cleanlab is absent.
    # ------------------------------------------------------------------
    from cleanlab.filter import find_label_issues

    issue_mask = np.asarray(
        find_label_issues(labels=labels, pred_probs=pred_probs),
        dtype=bool,
    )
    if issue_mask.shape != (len(work),):
        raise ValueError("cleanlab returned an unexpected label-issue mask shape.")
    disagreement_mask = (
        work["p_pos"]
        .between(
            _DISAGREEMENT_LOWER,
            _DISAGREEMENT_UPPER,
            inclusive="neither",
        )
        .to_numpy(dtype=bool)
    )
    drop_mask = issue_mask & disagreement_mask
    return set(work.loc[drop_mask, "EIN2"].astype(str).tolist())


def _validated_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a training frame.

    Validates presence of required columns, uniqueness of ``EIN2``,
    and finiteness of ``p_pos`` and ``hard_label``.

    Args:
        frame: Raw training frame.

    Returns:
        A validated and type-cast copy of the input frame.

    Raises:
        ValueError: If any validation check fails.

    """
    missing = _TRAINING_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"training frame missing columns: {sorted(missing)}.")
    work = frame.copy().reset_index(drop=True)
    work["EIN2"] = work["EIN2"].astype(str).str.strip()
    if work["EIN2"].duplicated().any():
        duplicates = int(work["EIN2"].duplicated().sum())
        raise ValueError(
            f"training frame contains duplicate EIN2 values: {duplicates}."
        )

    p_pos = pd.to_numeric(work["p_pos"], errors="coerce")
    if p_pos.isna().any() or ((p_pos < 0) | (p_pos > 1)).any():
        raise ValueError("training frame p_pos values must be finite probabilities.")
    work["p_pos"] = p_pos.astype(float)

    labels = pd.to_numeric(work["hard_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("training frame hard_label values must be coded as 0/1.")
    work["hard_label"] = labels.astype(int)
    return work


def _aligned_oof_probs(train_df: pd.DataFrame, oof_probs: pd.DataFrame) -> pd.DataFrame:
    """Return OOF probabilities aligned to ``train_df`` row order.

    Validates the OOF schema, checks that the EIN2 set matches the training
    frame exactly, and reorders OOF rows to match ``train_df``.

    Args:
        train_df: Validated training frame (used for the expected EIN2 set).
        oof_probs: Raw OOF probability frame.

    Returns:
        OOF probabilities aligned and reordered to match ``train_df``.

    Raises:
        ValueError: If the OOF schema is invalid or the EIN2 sets mismatch.

    """
    missing = _OOF_COLUMNS - set(oof_probs.columns)
    if missing:
        raise ValueError(f"OOF probabilities missing columns: {sorted(missing)}.")
    oof = oof_probs.copy().reset_index(drop=True)
    oof["EIN2"] = oof["EIN2"].astype(str).str.strip()
    if oof["EIN2"].duplicated().any():
        duplicates = int(oof["EIN2"].duplicated().sum())
        raise ValueError(
            f"OOF probabilities contain duplicate EIN2 values: {duplicates}."
        )

    expected = train_df["EIN2"].astype(str).str.strip()
    if set(oof["EIN2"]) != set(expected):
        raise ValueError("OOF probabilities must match the training EIN2 set exactly.")

    probs = oof.set_index("EIN2").loc[expected, ["p0", "p1"]].reset_index()
    p0 = pd.to_numeric(probs["p0"], errors="coerce")
    p1 = pd.to_numeric(probs["p1"], errors="coerce")
    if p0.isna().any() or p1.isna().any():
        raise ValueError("OOF probabilities must be numeric.")
    p0_arr = p0.to_numpy(dtype=float)
    p1_arr = p1.to_numpy(dtype=float)
    if ((p0_arr < 0) | (p0_arr > 1) | (p1_arr < 0) | (p1_arr > 1)).any():
        raise ValueError("OOF probabilities must lie within [0, 1].")
    if not np.allclose(p0_arr + p1_arr, 1.0, atol=1e-6):
        raise ValueError("OOF p0 and p1 must sum to 1.")
    probs["p0"] = p0_arr
    probs["p1"] = p1_arr
    return probs


__all__ = [
    "LossSpec",
    "class_weighted_arm",
    "class_weights",
    "hard_arm",
    "prune_ein2s",
    "pruned_arm",
    "run_arm",
]
