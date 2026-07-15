"""Thin CLI wrapper for stage 09 prevalence estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.paths import PathRegistry
from binary_classifier.prevalence.estimate import run_prevalence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage 09 prevalence estimation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and call the package prevalence entrypoint."""
    setup_logging(stem="09_prevalence")

    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    run_prevalence(cfg, registry)


if __name__ == "__main__":
    main()
