"""Thin CLI wrapper for stage 06 training."""

from __future__ import annotations

import argparse
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.log_utils import setup_logging
from binary_classifier.paths import PathRegistry
from binary_classifier.train.trainer import run_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage 06 training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Run only TF-IDF/MiniLM baselines.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=None,
        help="Run the documentation curve and model-selection sweep.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Run final seeds for the current selection-report winner.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Limit to encoder ID.",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Override encoder training fraction, e.g. 0.5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs for local tiers.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed override for sweep and final runs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    return parser.parse_args()


def main() -> None:
    """Load config, apply CLI overrides, and call the package entrypoint."""
    setup_logging(stem="06_train")

    args = _parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg.training.epochs = int(args.epochs)
    if args.seeds is not None:
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        cfg.training.sweep_seeds = seeds
        cfg.training.final_seeds = seeds

    # Bind the registry to the in-memory (override-applied) config so the
    # registry and cfg cannot silently diverge if downstream code consults
    # ``registry.cfg``.
    registry = PathRegistry.from_config(cfg)

    sweep = not args.final if args.sweep is None else bool(args.sweep)
    run_training(
        cfg,
        registry,
        baselines_only=bool(args.baselines_only),
        sweep=sweep,
        final=bool(args.final),
        encoder=args.encoder,
        limit=args.limit,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
