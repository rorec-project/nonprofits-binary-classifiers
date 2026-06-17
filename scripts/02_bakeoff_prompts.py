"""Thin CLI wrapper for Stage 2.2: prompt bake-off.

Calls :func:`binary_classifier.annotate.bakeoff_prompts.run_bakeoff`.
"""

import argparse
import logging
from pathlib import Path

from binary_classifier.annotate.bakeoff_prompts import run_bakeoff
from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt / model bake-off on prompt-dev vs human labels.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        type=Path,
        default=[
            Path("src/binary_classifier/annotate/prompts/v1.txt"),
            Path("src/binary_classifier/annotate/prompts/v2.txt"),
            Path("src/binary_classifier/annotate/prompts/v3.txt"),
        ],
        help="Prompt text files to evaluate.",
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        default=None,
        help="Coded human labels (defaults to the gold coding template; "
        "the prompt_dev rows are used).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the bake-off results JSON "
        "(defaults to registry.bakeoff_results).",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=None,
        help="Path to write the long/tidy bake-off label store "
        "(defaults to registry.bakeoff_store).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompt-dev records for a quick test.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(stem="02_bakeoff_prompts")

    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    logger.info("[02] Running prompt bake-off...")
    run_bakeoff(
        cfg,
        registry,
        prompt_paths=args.prompts,
        human_labels_path=args.human_labels,
        output_path=args.output,
        store_path=args.store_path,
        limit=args.limit,
    )
    logger.info("[02] Done.")


if __name__ == "__main__":
    main()
