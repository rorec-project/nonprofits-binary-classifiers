"""Thin CLI wrapper for Stage 4: quality control and aggregation.

Calls :func:`binary_classifier.qc.agreement.run_quality_check`.
"""

import argparse
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.paths import PathRegistry
from binary_classifier.qc.agreement import run_quality_check


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate labels and run the QC gate "
        "(agreement, Krippendorff alpha).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("data/annotation_store.csv"),
        help="Path to the long/tidy annotation store.",
    )
    parser.add_argument(
        "--human-validation",
        type=Path,
        default=None,
        help="CSV of human validation labels (EIN2, human_label).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_test_datasets/silver_labels.csv"),
        help="Path to write the frozen silver labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    run_quality_check(
        cfg,
        registry,
        store_path=args.store_path,
        human_validation_path=args.human_validation,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
