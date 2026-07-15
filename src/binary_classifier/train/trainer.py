"""Stage-06 training entrypoint."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from binary_classifier.train.data import (
    build_training_frame,
    load_human_split,
    split_dev,
)
from binary_classifier.train.sweep import (
    build_run_matrix,
    execute_run_matrix,
    print_selected_model_skeleton,
    write_selection_report,
)

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def run_training(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    *,
    baselines_only: bool = False,
    sweep: bool = True,
    final: bool = False,
    encoder: str | None = None,
    limit: int | None = None,
    subset: float | None = None,
    oof_finetune_fn: Callable[..., Any] | None = None,
) -> None:
    """Run stage 06: baselines, sweep matrix, selection, and optional finals.

    Args:
        cfg: Validated task configuration.
        registry: Resolved path registry.
        baselines_only: Run only configured baselines.
        sweep: Run the documentation curve and model-selection arm matrix.
        final: Run final seeds for the selected model cell.
        encoder: Optional encoder-id filter for local tiers.
        limit: Optional row limit applied before the dev split.
        subset: Optional training-fraction override for encoder runs.
        oof_finetune_fn: Optional injected fine-tune predictor used for cross-fit
            OOF probabilities; defaults to the encoder OOF predictor. Primarily
            for tests.

    """
    if sweep and final:
        run_training(
            cfg,
            registry,
            baselines_only=baselines_only,
            sweep=True,
            final=False,
            encoder=encoder,
            limit=limit,
            subset=subset,
            oof_finetune_fn=oof_finetune_fn,
        )
        run_training(
            cfg,
            registry,
            baselines_only=baselines_only,
            sweep=False,
            final=True,
            encoder=encoder,
            limit=limit,
            subset=subset,
            oof_finetune_fn=oof_finetune_fn,
        )
        return

    recommendation: dict[str, object] | None = None
    if final:
        if not registry.selection_report.exists():
            raise FileNotFoundError(
                f"Final-seed refit requires an existing selection report at "
                f"{registry.selection_report}. Run stage 06 sweep first.",
            )
        recommendation = _load_recommendation(registry)
        if recommendation is None:
            raise ValueError(
                f"Final-seed refit requires a usable recommendation in "
                f"{registry.selection_report}. Run stage 06 sweep first.",
            )

    frame, validation_df = _load_training_inputs(cfg, registry)
    if limit is not None:
        frame = frame.iloc[: int(limit)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Stage 06 requires at least one silver training row.")

    train_df, dev_df = split_dev(frame, cfg.training.dev_fraction, cfg.SEED)
    logger.info(
        "Stage 06 split %d silver rows into %d train / %d dev rows",
        len(frame),
        len(train_df),
        len(dev_df),
    )

    if not baselines_only and _needs_oof(cfg):
        _ensure_oof_pred_probs(cfg, registry, train_df, oof_finetune_fn)

    specs = build_run_matrix(
        cfg,
        baselines_only=baselines_only,
        sweep=sweep,
        final=final,
        encoder=encoder,
        subset=subset,
        recommendation=recommendation,
    )
    rows = execute_run_matrix(cfg, registry, specs, train_df, dev_df, validation_df)

    if baselines_only:
        return

    sweep_cells = [spec.cell_key for spec in specs if spec.phase == "sweep"]
    if sweep_cells:
        report = write_selection_report(
            rows,
            registry.selection_report,
            cfg=cfg,
            registry=registry,
            eligible_cells=sweep_cells,
        )
        print_selected_model_skeleton(report)
    elif final:
        report = _load_selection_report(registry)
        print_selected_model_skeleton(report)


def _load_training_inputs(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load stage-06 training and validation frames."""
    try:
        frame = build_training_frame(cfg, registry)
        validation_df = load_human_split(cfg, registry, "validation")
        return frame, validation_df
    except FileNotFoundError:
        if not cfg.data.allow_synthetic:
            raise
        logger.warning(
            "Stage-06 inputs missing with data.allow_synthetic=true; using an "
            "in-memory synthetic smoke frame and validation split.",
        )
        return _synthetic_training_inputs(cfg)


def _synthetic_training_inputs(
    cfg: BinaryClassifierConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build in-memory synthetic stage-06 inputs without human artifacts."""
    from binary_classifier.data.load import load_missions

    missions = load_missions(cfg).copy().reset_index(drop=True)
    missions["text"] = missions["mission_text"].fillna("").astype(str)
    labels = missions["text"].str.contains(
        "church|prayer|worship|faith|ministry|christian|religious",
        case=False,
        regex=True,
    )
    if labels.nunique() < 2:
        labels = pd.Series([i % 2 == 1 for i in range(len(missions))])
    missions["hard_label"] = labels.astype(int)
    missions["p_pos"] = missions["hard_label"].map({0: 0.1, 1: 0.9}).astype(float)

    n_train = min(int(cfg.sample_sizes.silver), max(20, len(missions) - 20))
    train = missions.iloc[:n_train][
        ["EIN2", "text", "ntee_major_group", "p_pos", "hard_label"]
    ].reset_index(drop=True)
    validation = missions.iloc[n_train : n_train + 40][["EIN2", "text", "hard_label"]]
    if validation.empty:
        validation = train.tail(min(20, len(train)))[["EIN2", "text", "hard_label"]]
    validation = validation.rename(columns={"hard_label": "human_label"}).reset_index(
        drop=True,
    )
    return train, validation


def _load_recommendation(registry: PathRegistry) -> dict[str, object] | None:
    """Load the existing selection recommendation for final runs."""
    if not registry.selection_report.exists():
        return None
    report = _load_selection_report(registry)
    recommendation = report.get("recommendation")
    if not isinstance(recommendation, dict):
        return None
    return cast(dict[str, object], recommendation)


def _load_selection_report(registry: PathRegistry) -> dict[str, object]:
    """Load ``selection_report.json`` as an object."""
    raw = json.loads(registry.selection_report.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {registry.selection_report}.")
    return raw


def _needs_oof(cfg: BinaryClassifierConfig) -> bool:
    """Return whether stage 06 must produce out-of-fold probabilities.

    True when the pruned arm is configured (it consumes true OOF instead of the
    vote-share proxy) or when CROWDLAB is a configured aggregation comparison arm
    (stage 11 reads ``registry.oof_pred_probs``).
    """
    return (
        "pruned" in cfg.training.arms or "crowdlab" in cfg.aggregation.comparison_arms
    )


def _ensure_oof_pred_probs(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    train_df: pd.DataFrame,
    finetune_fn: Callable[..., Any] | None,
) -> None:
    """Compute cross-fit OOF probabilities for the pruned arm / CROWDLAB.

    Skipped when ``registry.oof_pred_probs`` already exists so re-runs stay
    idempotent. Runs before the sweep so the pruned arm loads true OOF rather
    than the vote-share proxy.
    """
    if registry.oof_pred_probs.exists():
        logger.info(
            "Skipping OOF probabilities; %s already exists",
            registry.oof_pred_probs,
        )
        return
    from binary_classifier.train.crossfit import compute_oof_pred_probs

    logger.info("Computing cross-fit OOF probabilities (pruned arm / CROWDLAB)...")
    compute_oof_pred_probs(cfg, registry, train_df, finetune_fn=finetune_fn)
