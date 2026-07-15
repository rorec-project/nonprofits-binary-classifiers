"""Thin CLI wrapper for Stage 4: QC and majority-vote silver-label freeze.

Calls :func:`binary_classifier.qc.agreement.run_quality_check`.
"""

import argparse
import logging
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.paths import PathRegistry
from binary_classifier.qc.agreement import run_quality_check

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze majority-vote labels after the QC agreement gate.",
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
        default=None,
        help="Path to the long/tidy annotation store "
        "(defaults to registry.annotation_store).",
    )
    parser.add_argument(
        "--human-validation",
        type=Path,
        default=None,
        help="Coded human validation labels (defaults to the gold coding "
        "template; the validation split is used).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the frozen silver labels "
        "(defaults to processed_dir/silver_labels.csv).",
    )
    return parser.parse_args()


def main() -> None:
    """Load stage-04 inputs and delegate QC to the package implementation.

    The CLI remains a thin wrapper so standalone runs and the orchestrator share
    the same majority-vote freeze gate and fabricated-evidence abstention
    behavior.
    """
    setup_logging(stem="04_quality_check")

    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    logger.info("[04] Running quality check...")
    run_quality_check(
        cfg,
        registry,
        store_path=args.store_path,
        human_validation_path=args.human_validation,
        output_path=args.output,
    )
    logger.info("[04] Done.")


if __name__ == "__main__":
    main()
