"""Thin CLI wrapper for stage 08 batch inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.inference.predict import run_inference
from binary_classifier.log_utils import setup_logging
from binary_classifier.paths import PathRegistry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage 08 batch inference.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke-test inference runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and call the package inference entrypoint."""
    setup_logging(stem="08_infer")

    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    run_inference(cfg, registry, limit=args.limit)


if __name__ == "__main__":
    main()
