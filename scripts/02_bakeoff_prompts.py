"""Thin CLI wrapper for Stage 2.2: prompt bake-off.

Calls :func:`binary_classifier.annotate.bakeoff_prompts.run_bakeoff`.
"""

import argparse
from pathlib import Path

from binary_classifier.annotate.bakeoff_prompts import run_bakeoff
from binary_classifier.config import load_config
from binary_classifier.paths import PathRegistry


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
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="Override the model slate (defaults to config model_slate).",
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        default=None,
        help="CSV of human labels for prompt-dev (EIN2, human_label, source_type).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bakeoff_results.json"),
        help="Path to write the bake-off results JSON.",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("data/bakeoff_labels.csv"),
        help="Path to write the long/tidy bake-off label store.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompt-dev records for a quick test.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    run_bakeoff(
        cfg,
        registry,
        prompt_paths=args.prompts,
        model_ids=args.models,
        human_labels_path=args.human_labels,
        output_path=args.output,
        store_path=args.store_path,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
