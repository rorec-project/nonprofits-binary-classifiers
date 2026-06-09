"""Thin CLI wrapper for Stage 2.3: full annotation run.

Calls :func:`binary_classifier.annotate.run_annotation.run_annotation`.
"""

import argparse
from pathlib import Path

from binary_classifier.annotate.run_annotation import run_annotation
from binary_classifier.config import load_config
from binary_classifier.paths import PathRegistry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full annotation matrix over the silver pool.",
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
        help="Prompt text files to use.",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=None,
        help="Path to the long/tidy label store "
        "(defaults to registry.annotation_store).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of silver records for a quick test.",
    )
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Run only the canary EIN2 set (drift detection).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing store and start from scratch.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Flush the store to disk every N records.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    run_annotation(
        cfg,
        registry,
        limit=args.limit,
        prompt_paths=args.prompts,
        store_path=args.store_path,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        canary_only=args.canary,
    )


if __name__ == "__main__":
    main()
