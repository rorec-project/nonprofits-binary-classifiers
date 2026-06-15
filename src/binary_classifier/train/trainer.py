"""Stage-06 training entrypoint."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

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
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    *,
    baselines_only: bool = False,
    sweep: bool = True,
    final: bool = False,
    encoder: str | None = None,
    limit: int | None = None,
    subset: float | None = None,
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

    """
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

    recommendation = _load_recommendation(registry) if final else None
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
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
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
    cfg: "BinaryClassifierConfig",
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
        drop=True
    )
    return train, validation


def _load_recommendation(registry: "PathRegistry") -> dict[str, object] | None:
    """Load the existing selection recommendation for final runs."""
    if not registry.selection_report.exists():
        return None
    report = _load_selection_report(registry)
    recommendation = report.get("recommendation")
    if not isinstance(recommendation, dict):
        return None
    return cast(dict[str, object], recommendation)


def _load_selection_report(registry: "PathRegistry") -> dict[str, object]:
    """Load ``selection_report.json`` as an object."""
    raw = json.loads(registry.selection_report.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {registry.selection_report}.")
    return raw
