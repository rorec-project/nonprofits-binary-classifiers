"""Orchestrator entrypoint that chains stages 01 through 04.

Stages are thin wrappers around the ``src/binary_classifier/`` package.
Each stage is executed in order and can be selectively enabled via the
``--stages`` flag.
"""

import argparse
import importlib
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.paths import PathRegistry

# ── Stage registry ───────────────────────────────────────────────────────────

_STAGE_MODULES = {
    "01": ("binary_classifier.data.sample", "build_sample"),
    "02": ("binary_classifier.annotate.bakeoff_prompts", "run_bakeoff"),
    "03": ("binary_classifier.annotate.run_annotation", "run_annotation"),
    "04": ("binary_classifier.qc.agreement", "run_quality_check"),
}

# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline "
        "(01 sample → 02 bake-off → 03 annotate → 04 QC).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="01,02,03,04",
        help="Comma-separated list of stages to run.",
    )
    parser.add_argument(
        "--annotate-limit",
        type=int,
        default=None,
        help="Optional limit for stage 03 (smoke-test).",
    )
    return parser.parse_args()


# ── Stage runner ─────────────────────────────────────────────────────────────


def _run_stage(
    stage_id: str,
    cfg,
    registry,
    annotate_limit: int | None,
) -> None:
    """Import and execute a single stage."""
    module_name, func_name = _STAGE_MODULES[stage_id]

    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    if stage_id == "03" and annotate_limit is not None:
        func(cfg, registry, limit=annotate_limit)
    else:
        func(cfg, registry)


# ── Main entrypoint ───────────────────────────────────────────────────────────


def main() -> None:
    """Run the requested pipeline stages in order."""
    args = _parse_args()

    cfg = load_config(args.config)
    registry = PathRegistry(args.config)

    requested = {s.strip() for s in args.stages.split(",")}
    invalid = requested - set(_STAGE_MODULES.keys())
    if invalid:
        print(f"Invalid stage(s): {invalid}", file=sys.stderr)
        sys.exit(1)

    for stage_id in ("01", "02", "03", "04"):
        if stage_id in requested:
            print(f"Running stage {stage_id} ...")
            _run_stage(stage_id, cfg, registry, args.annotate_limit)


if __name__ == "__main__":
    main()
