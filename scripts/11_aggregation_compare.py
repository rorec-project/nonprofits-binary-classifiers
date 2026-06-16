"""Thin CLI wrapper for stage 11 aggregation comparison."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.paths import PathRegistry
from binary_classifier.qc.aggregation_compare import run_aggregation_compare


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run stage 11 aggregation comparison.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and call the package aggregation-comparison entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    run_aggregation_compare(cfg, registry)


if __name__ == "__main__":
    main()
