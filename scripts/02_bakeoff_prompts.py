"""Thin CLI wrapper for Stage 2.2: prompt bake-off.

Calls :func:`binary_classifier.annotate.bakeoff_prompts.run_bakeoff`.
"""

import argparse
import logging
from pathlib import Path

from binary_classifier.annotate.bakeoff_prompts import (
    rebuild_bakeoff_artifacts_from_store,
    rebuild_proposed_slate_from_results,
    run_bakeoff,
)
from binary_classifier.config import (
    BakeoffCandidate,
    BinaryClassifierConfig,
    load_config,
)
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
    parser.add_argument(
        "--only-model",
        default=None,
        help="Restrict annotation to one configured bake-off model id.",
    )
    rebuild_group = parser.add_mutually_exclusive_group()
    rebuild_group.add_argument(
        "--rebuild-from-store",
        action="store_true",
        help="Do not annotate; rebuild bakeoff_results.json and proposed_slate.json "
        "from the existing full bake-off label store.",
    )
    rebuild_group.add_argument(
        "--rebuild-slate-from-results",
        action="store_true",
        help="Do not annotate; only rebuild proposed_slate.json from existing "
        "bakeoff_results.json.",
    )
    return parser.parse_args()


def _select_candidates(
    cfg: BinaryClassifierConfig,
    only_model: str | None,
) -> list[BakeoffCandidate] | None:
    """Return a candidate override for ``--only-model``, if requested."""
    if only_model is None:
        return None

    matches = [
        spec for spec in cfg.model_slate.bakeoff_candidates if spec.id == only_model
    ]
    if not matches:
        configured = ", ".join(spec.id for spec in cfg.model_slate.bakeoff_candidates)
        raise ValueError(
            f"--only-model {only_model!r} is not configured. "
            f"Configured bake-off models: {configured}",
        )
    return matches


def main() -> None:
    setup_logging(stem="02_bakeoff_prompts")

    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    selected_candidates = _select_candidates(cfg, args.only_model)
    if args.rebuild_slate_from_results:
        logger.info("[02] Rebuilding proposed slate from existing bake-off results...")
        rebuild_proposed_slate_from_results(
            registry,
            kappa_threshold=cfg.qc.kappa_threshold,
            f1_ci_floor=cfg.qc.f1_ci_floor,
            agreement_threshold=cfg.qc.agreement_threshold,
        )
        logger.info("[02] Done.")
        return
    if args.rebuild_from_store:
        logger.info("[02] Rebuilding bake-off artifacts from existing label store...")
        rebuild_bakeoff_artifacts_from_store(
            cfg,
            registry,
            prompt_paths=args.prompts,
            candidates=selected_candidates,
            human_labels_path=args.human_labels,
            output_path=args.output,
            store_path=args.store_path,
            limit=args.limit,
        )
        logger.info("[02] Done.")
        return

    logger.info("[02] Running prompt bake-off...")
    run_bakeoff(
        cfg,
        registry,
        prompt_paths=args.prompts,
        human_labels_path=args.human_labels,
        output_path=args.output,
        store_path=args.store_path,
        limit=args.limit,
        candidates=selected_candidates,
    )
    logger.info("[02] Done.")


if __name__ == "__main__":
    main()
