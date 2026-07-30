"""Build the names-arm panel and BMF-only cross-sections."""

import argparse
import logging
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.names.frame import build_name_frame
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build names-arm input frames.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run names stage N1."""
    setup_logging(stem="N1_build_name_frame")
    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    logger.info("[N1] Building names frames...")
    build_name_frame(cfg, registry)
    logger.info("[N1] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
