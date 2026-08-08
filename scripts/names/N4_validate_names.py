"""Validate mission-to-name cross-field transfer on paired organizations."""

import argparse
import logging
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.names.validation import run_name_validation
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def main() -> int:
    """Run stage N4."""
    parser = argparse.ArgumentParser(description="Validate names-arm transfer scores.")
    parser.add_argument(
        "--config", type=Path, default=Path("config/religious_missions.yaml")
    )
    args = parser.parse_args()
    setup_logging(stem="N4_validate_names")
    cfg = load_config(args.config)
    run_name_validation(cfg, PathRegistry(args.config))
    logger.info("[N4] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
