"""Clean both names-arm populations and run the divergence gate."""

import argparse
import logging
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.names.cleaner import clean_names
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def main() -> int:
    """Run stage N2."""
    parser = argparse.ArgumentParser(description="Clean names-arm input frames.")
    parser.add_argument(
        "--config", type=Path, default=Path("config/religious_missions.yaml")
    )
    args = parser.parse_args()
    setup_logging(stem="N2_clean_names")
    cfg = load_config(args.config)
    clean_names(cfg, PathRegistry(args.config))
    logger.info("[N2] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
