"""Run diagnostic-only synthetic probes and legal-name/DBA case study."""

import argparse
import logging
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.names.diagnostics import run_name_diagnostics
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def main() -> int:
    """Run stage N5."""
    parser = argparse.ArgumentParser(description="Run names-arm diagnostics.")
    parser.add_argument(
        "--config", type=Path, default=Path("config/religious_missions.yaml")
    )
    args = parser.parse_args()
    setup_logging(stem="N5_name_diagnostics")
    cfg = load_config(args.config)
    run_name_diagnostics(cfg, PathRegistry(args.config))
    logger.info("[N5] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
