"""Thin CLI wrapper for stage 07 frozen-test evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.evaluation.evaluate import run_evaluation
from binary_classifier.paths import PathRegistry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage 07 evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and call the package evaluation entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    run_evaluation(cfg, registry)


if __name__ == "__main__":
    main()
