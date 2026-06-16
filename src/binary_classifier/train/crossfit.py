"""Out-of-fold prediction probabilities for stage 06 training rows."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig, EncoderArm
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

_OOF_COLUMNS = ["EIN2", "fold", "p0", "p1"]
_REQUIRED_FRAME_COLUMNS = {"EIN2", "text", "hard_label"}


def compute_oof_pred_probs(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    frame: pd.DataFrame,
    *,
    finetune_fn: Callable[..., Any] | None = None,
) -> Path:
    """Compute stratified K-fold out-of-fold prediction probabilities.

    The injected ``finetune_fn`` follows the encoder fine-tune call shape and
    must return either a predictor directly, a callable predictor, or a mapping
    with a ``"predictor"`` entry. Predictors must expose
    ``predict_proba(eval_df)`` or be callable as ``predictor(eval_df)``. Returned
    probabilities may be a two-column ``(p0, p1)`` array/DataFrame or a one-
    dimensional positive-class probability vector.

    Args:
        cfg: Validated task configuration.
        registry: Path registry that owns the OOF artifact location.
        frame: Training frame containing at least ``EIN2``, ``text``, and
            ``hard_label``.
        finetune_fn: Optional injected fine-tuning function, primarily for
            tests. Defaults to :func:`binary_classifier.train.encoder.finetune`.

    Returns:
        Path to ``registry.oof_pred_probs`` with columns ``EIN2, fold, p0, p1``.

    Raises:
        ValueError: If the input frame, fold configuration, shard schema, or
            predictor output is invalid.

    """
    work = _validated_frame(frame)
    n_folds = _validated_fold_count(cfg, work)
    encoder = _selected_encoder(cfg)
    training_arm = _selected_training_arm(cfg)
    targets = cfg.training.targets
    trainer = finetune_fn if finetune_fn is not None else _default_finetune

    registry.oof_pred_probs.parent.mkdir(parents=True, exist_ok=True)
    fold_dir = registry.oof_pred_probs.parent / "oof_pred_probs_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    labels = work["hard_label"].to_numpy(dtype=int)
    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=int(cfg.SEED),
    )

    shards: list[pd.DataFrame] = []
    for fold_id, (train_idx, heldout_idx) in enumerate(splitter.split(work, labels)):
        heldout_df = work.iloc[heldout_idx].reset_index(drop=True)
        expected_ein2 = heldout_df["EIN2"].astype(str).tolist()
        shard_path = fold_dir / f"fold_{fold_id}.parquet"

        if shard_path.exists():
            shard = pd.read_parquet(shard_path)
            if _valid_fold_shard(shard, fold_id=fold_id, expected_ein2=expected_ein2):
                logger.info("Skipping completed OOF fold %d at %s", fold_id, shard_path)
                shards.append(shard[_OOF_COLUMNS].copy())
                continue
            logger.warning("Recomputing invalid OOF fold shard at %s", shard_path)

        train_df = work.iloc[train_idx].reset_index(drop=True)
        logger.info(
            "Training OOF fold %d/%d on %d rows; scoring %d held-out rows",
            fold_id + 1,
            n_folds,
            len(train_df),
            len(heldout_df),
        )
        result = trainer(
            cfg,
            registry,
            encoder,
            train_df,
            heldout_df,
            targets=targets,
            arm=training_arm,
            train_fraction=1.0,
            seed=int(cfg.SEED) + fold_id,
            run_root=registry.runs_dir / "crossfit" / f"fold_{fold_id}",
        )
        p0, p1 = _predict_fold_probs(_extract_predictor(result), heldout_df)
        shard = pd.DataFrame(
            {
                "EIN2": expected_ein2,
                "fold": np.full(len(heldout_df), fold_id, dtype=int),
                "p0": p0,
                "p1": p1,
            },
            columns=_OOF_COLUMNS,
        )
        _validate_fold_shard(shard, fold_id=fold_id, expected_ein2=expected_ein2)
        shard.to_parquet(shard_path, index=False)
        shards.append(shard)

    final = pd.concat(shards, ignore_index=True)[_OOF_COLUMNS].copy()
    final["EIN2"] = _normalize_ein2(final["EIN2"])
    final["fold"] = final["fold"].astype(int)
    final["p0"] = final["p0"].astype(float)
    final["p1"] = final["p1"].astype(float)
    final = final.sort_values(["fold", "EIN2"], kind="mergesort").reset_index(
        drop=True,
    )
    _validate_final_oof(final, expected_ein2=work["EIN2"].astype(str).tolist())
    final.to_parquet(registry.oof_pred_probs, index=False)
    logger.info("Wrote OOF prediction probabilities to %s", registry.oof_pred_probs)
    return registry.oof_pred_probs


def _default_finetune(*args: Any, **kwargs: Any) -> Any:
    """Lazily import and call the encoder OOF predictor implementation.

    Cross-fit scoring needs a predictor (``predict_proba``), not a metrics row,
    so this defaults to :func:`binary_classifier.train.encoder.finetune_predictor`
    rather than ``finetune``.
    """
    from binary_classifier.train.encoder import finetune_predictor

    return finetune_predictor(*args, **kwargs)


def _validated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a valid cross-fit training frame."""
    missing = _REQUIRED_FRAME_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}.")
    if frame.empty:
        raise ValueError("frame must contain at least one row.")

    work = frame.copy().reset_index(drop=True)
    work["EIN2"] = _normalize_ein2(work["EIN2"])
    if work["EIN2"].duplicated().any():
        duplicates = int(work["EIN2"].duplicated().sum())
        raise ValueError(f"frame contains duplicate EIN2 values: {duplicates}.")

    labels = pd.to_numeric(work["hard_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("frame hard_label values must be coded as 0/1.")
    work["hard_label"] = labels.astype(int)
    return work


def _validated_fold_count(cfg: BinaryClassifierConfig, frame: pd.DataFrame) -> int:
    """Return a feasible stratified K-fold count."""
    n_folds = int(cfg.training.crossfit_folds)
    if n_folds < 2:
        raise ValueError("training.crossfit_folds must be at least 2.")
    if n_folds > len(frame):
        raise ValueError("training.crossfit_folds cannot exceed the frame row count.")

    counts = frame["hard_label"].value_counts().to_dict()
    if set(counts) != {0, 1}:
        raise ValueError("frame hard_label must contain both binary classes.")
    min_class_count = min(int(counts[0]), int(counts[1]))
    if n_folds > min_class_count:
        raise ValueError(
            "training.crossfit_folds cannot exceed the smallest class count "
            f"({min_class_count}).",
        )
    return n_folds


def _selected_encoder(cfg: BinaryClassifierConfig) -> EncoderArm:
    """Return the configured primary encoder, falling back to the first arm."""
    if not cfg.training.encoders:
        raise ValueError("training.encoders must contain at least one encoder.")
    for encoder in cfg.training.encoders:
        if encoder.arm == "primary":
            return encoder
    return cfg.training.encoders[0]


def _selected_training_arm(cfg: BinaryClassifierConfig) -> str:
    """Return the default soft-label recipe arm for OOF fitting."""
    del cfg
    return "default"


def _extract_predictor(result: Any) -> Any:
    """Extract a predictor from a fine-tune result or return the result itself."""
    if isinstance(result, Mapping):
        if "predictor" not in result:
            raise ValueError(
                "finetune_fn returned a mapping without a 'predictor' entry; "
                "cross-fit requires a predictor with predict_proba(eval_df).",
            )
        return result["predictor"]
    return result


def _predict_fold_probs(
    predictor: Any, eval_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated ``(p0, p1)`` arrays from a predictor."""
    if hasattr(predictor, "predict_proba"):
        raw = predictor.predict_proba(eval_df)
    elif callable(predictor):
        raw = predictor(eval_df)
    else:
        raise ValueError(
            "Predictor must expose predict_proba(eval_df) or be callable.",
        )
    return _coerce_probabilities(raw, n_rows=len(eval_df))


def _coerce_probabilities(raw: Any, *, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert predictor output into two valid probability vectors."""
    if isinstance(raw, pd.DataFrame):
        if {"p0", "p1"}.issubset(raw.columns):
            p0 = raw["p0"].to_numpy(dtype=float)
            p1 = raw["p1"].to_numpy(dtype=float)
        elif "p1" in raw.columns:
            p1 = raw["p1"].to_numpy(dtype=float)
            p0 = 1.0 - p1
        else:
            raise ValueError("predict_proba DataFrame must contain p1 or p0/p1.")
    elif isinstance(raw, pd.Series):
        p1 = raw.to_numpy(dtype=float)
        p0 = 1.0 - p1
    else:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            p1 = arr
            p0 = 1.0 - p1
        elif arr.ndim == 2 and arr.shape[1] == 2:
            p0 = arr[:, 0]
            p1 = arr[:, 1]
        else:
            raise ValueError(
                "predict_proba output must be shape (n,), (n, 2), or p1/p0-p1 DataFrame.",
            )

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if len(p0) != n_rows or len(p1) != n_rows:
        raise ValueError(
            f"predict_proba returned {len(p1)} rows for an eval frame of {n_rows}.",
        )
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError("predict_proba returned non-finite probabilities.")
    if ((p0 < 0) | (p0 > 1) | (p1 < 0) | (p1 > 1)).any():
        raise ValueError("predict_proba probabilities must be within [0, 1].")
    if not np.allclose(p0 + p1, 1.0, atol=1e-6):
        raise ValueError("predict_proba p0 and p1 must sum to 1.")
    return p0.astype(float), p1.astype(float)


def _valid_fold_shard(
    shard: pd.DataFrame,
    *,
    fold_id: int,
    expected_ein2: list[str],
) -> bool:
    """Return whether an existing shard is usable for resume."""
    try:
        _validate_fold_shard(shard, fold_id=fold_id, expected_ein2=expected_ein2)
    except ValueError:
        return False
    return True


def _validate_fold_shard(
    shard: pd.DataFrame,
    *,
    fold_id: int,
    expected_ein2: list[str],
) -> None:
    """Validate one per-fold shard against its expected held-out EIN2s."""
    if list(shard.columns) != _OOF_COLUMNS:
        raise ValueError(f"OOF shard schema must be exactly {_OOF_COLUMNS}.")
    if len(shard) != len(expected_ein2):
        raise ValueError("OOF shard row count does not match held-out fold size.")
    fold_values = pd.to_numeric(shard["fold"], errors="coerce")
    if fold_values.isna().any() or set(fold_values.astype(int).tolist()) != {fold_id}:
        raise ValueError(f"OOF shard fold values must all equal {fold_id}.")
    actual_ein2 = set(_normalize_ein2(shard["EIN2"]))
    expected_set = set(expected_ein2)
    if actual_ein2 != expected_set or len(actual_ein2) != len(shard):
        raise ValueError("OOF shard EIN2 set does not match the held-out fold.")
    _coerce_probabilities(shard[["p0", "p1"]], n_rows=len(shard))


def _validate_final_oof(final: pd.DataFrame, *, expected_ein2: list[str]) -> None:
    """Validate the final OOF artifact before writing it."""
    if list(final.columns) != _OOF_COLUMNS:
        raise ValueError(f"OOF artifact schema must be exactly {_OOF_COLUMNS}.")
    actual = _normalize_ein2(final["EIN2"])
    if len(final) != len(expected_ein2) or actual.duplicated().any():
        raise ValueError("OOF artifact must contain every EIN2 exactly once.")
    if set(actual) != set(expected_ein2):
        raise ValueError("OOF artifact EIN2 set does not match the training frame.")
    _coerce_probabilities(final[["p0", "p1"]], n_rows=len(final))


def _normalize_ein2(series: pd.Series) -> pd.Series:
    """Normalize EIN2 values for artifact comparisons and persisted output."""
    return series.astype(str).str.strip()
