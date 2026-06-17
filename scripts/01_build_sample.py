"""Thin CLI wrapper for Stage 2.1 — Build the silver pool, gold set, and human splits.

Calls :func:`binary_classifier.data.sample.build_sample`, then runs the
plan-specified validation assertions on the resulting manifests.

Usage::

    uv run python scripts/01_build_sample.py
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from binary_classifier.config import load_config
from binary_classifier.data.sample import build_sample
from binary_classifier.log_utils import setup_logging
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the silver pool, gold set, and human splits.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing gold coding template "
        "(discards any human-entered labels).",
    )
    return parser.parse_args()


def main() -> int:
    """Orchestrate sample construction and run assertions."""
    setup_logging(stem="01_build_sample")

    args = _parse_args()

    cfg = load_config(args.config)
    registry = PathRegistry(args.config)

    logger.info("[01] Building sample...")
    build_sample(cfg, registry, force=args.force)
    logger.info("[01] Sample built.")

    # Load manifests for assertions
    silver = pd.read_csv(registry.silver_manifest)
    gold = pd.read_csv(registry.gold_manifest)
    prompt_dev = pd.read_csv(registry.prompt_dev_manifest)
    validation = pd.read_csv(registry.validation_manifest)
    test = pd.read_csv(registry.test_manifest)

    _run_assertions(silver, gold, prompt_dev, validation, test)
    logger.info("[01] Done.")
    return 0


def _run_assertions(
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    prompt_dev: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Run plan-specified checks and raise on failure."""
    errors: list[str] = []

    # EIN2 uniqueness per split
    for name, df in [
        ("silver", silver),
        ("gold", gold),
        ("prompt_dev", prompt_dev),
        ("validation", validation),
        ("test", test),
    ]:
        if df["EIN2"].duplicated().any():
            errors.append(f"Duplicate EIN2 in {name}")

    # All 26 groups in silver
    silver_groups = set(silver["stratum"].unique())
    missing_silver = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - silver_groups
    if missing_silver:
        errors.append(f"Silver missing strata: {missing_silver}")

    # Positive share ~30-40 % in silver
    pos_share = silver["is_positive_enriched"].mean()
    if not (0.25 <= pos_share <= 0.50):
        errors.append(f"Silver positive share {pos_share:.2%} outside 25-50 % band")

    # Gold has stratum + inclusion_prob
    for col in ["stratum", "inclusion_prob"]:
        if col not in gold.columns:
            errors.append(f"Gold manifest missing column {col}")

    # Human splits are disjoint
    all_eins = set(prompt_dev["EIN2"]) | set(validation["EIN2"]) | set(test["EIN2"])
    total_human = len(prompt_dev) + len(validation) + len(test)
    if len(all_eins) != total_human:
        errors.append("Human splits overlap")

    if errors:
        for e in errors:
            logger.error("[ASSERTION FAILED] %s", e)
        raise AssertionError(f"{len(errors)} assertion(s) failed")

    logger.info("[01] All assertions passed.")


if __name__ == "__main__":
    sys.exit(main())
