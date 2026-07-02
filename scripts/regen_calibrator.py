"""Regenerate calibrator threshold metadata from saved anchor OOF scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from binary_classifier.config import load_config
from binary_classifier.evaluation.thresholds import pick_threshold
from binary_classifier.log_utils import setup_logging
from binary_classifier.paths import PathRegistry

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig


def regen_calibrator(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> dict[str, Any]:
    """Rewrite only ``calibrator.json`` using saved anchor OOF probabilities."""
    scores = pd.read_parquet(registry.anchor_oof_scores)
    required = {"prob_calibrated_oof", "human_label"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(
            f"{registry.anchor_oof_scores} missing columns {sorted(missing)}",
        )
    threshold_report = pick_threshold(
        scores["prob_calibrated_oof"].astype(float).tolist(),
        scores["human_label"].astype(int).tolist(),
        cfg.evaluation.threshold_policy,
        float(cfg.evaluation.precision_floor),
    )
    calibrator = json.loads(registry.calibrator_path.read_text())
    calibrator.update(
        {
            "threshold": threshold_report["threshold"],
            "threshold_policy": threshold_report["policy"],
            "precision_floor": float(cfg.evaluation.precision_floor),
            "achieved_precision": threshold_report["achieved_precision"],
            "achieved_recall": threshold_report["achieved_recall"],
            "max_f1_threshold": threshold_report["max_f1_threshold"],
            "pr_curve_points": threshold_report["pr_curve_points"],
            "anchor_oof_scores_path": str(registry.anchor_oof_scores),
        },
    )
    registry.calibrator_path.write_text(
        json.dumps(calibrator, indent=2, sort_keys=True) + "\n",
    )
    return calibrator


def main() -> None:
    """Regenerate calibrator.json PR-curve points from saved anchor OOF scores."""
    setup_logging(stem="regen_calibrator")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    regen_calibrator(cfg, registry)


if __name__ == "__main__":
    main()
