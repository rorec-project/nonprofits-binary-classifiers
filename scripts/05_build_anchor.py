"""Thin CLI wrapper for Stage 05 — build the full-frame anchor sample.

Usage::

    uv run python scripts/05_build_anchor.py --config config/smoke.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.data.anchor import build_anchor
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full-frame anchor sample and coding template.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing anchor coding template with human labels.",
    )
    return parser.parse_args()


def main() -> int:
    """Run stage 05 from the configured paths."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    registry = PathRegistry(args.config)

    logger.info("[05] Building anchor sample...")
    build_anchor(cfg, registry, force=args.force)
    logger.info("[05] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
