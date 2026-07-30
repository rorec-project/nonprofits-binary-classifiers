"""Score cleaned organization names with cross-field transfer."""

import argparse
import logging
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.names.score import score_names
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def main() -> int:
    """Run stage N3."""
    parser = argparse.ArgumentParser(description="Score names-arm input frames.")
    parser.add_argument(
        "--config", type=Path, default=Path("config/religious_missions.yaml")
    )
    args = parser.parse_args()
    setup_logging(stem="N3_score_names")
    cfg = load_config(args.config)
    score_names(cfg, PathRegistry(args.config))
    logger.info("[N3] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
