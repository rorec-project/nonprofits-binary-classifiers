"""Draw the seeded, stratified BMF-only names gold sample."""

import argparse
import logging
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.names.gold import draw_name_gold
from binary_classifier.paths import PathRegistry


logger = logging.getLogger(__name__)


def main() -> int:
    """Run stage N6."""
    parser = argparse.ArgumentParser(description="Draw BMF-only names gold sample.")
    parser.add_argument(
        "--config", type=Path, default=Path("config/religious_missions.yaml")
    )
    args = parser.parse_args()
    setup_logging(stem="N6_draw_name_gold")
    cfg = load_config(args.config)
    draw_name_gold(cfg, PathRegistry(args.config))
    logger.info("[N6] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
