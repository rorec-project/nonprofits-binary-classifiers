"""Orchestrator entrypoint that chains configured pipeline stages.

Stages are thin wrappers around the ``src/binary_classifier/`` package. The
orchestrator inserts human checkpoints that fail gracefully (clear message,
non-zero exit, no wasted GPU/API work):

* **G1 (labels)** is checked after stage 01 and before any model stage: the
  human coding template must carry strict ``0/1`` labels for every requested
  label-dependent stage (``prompt_dev``→02, ``validation``→04/06,
  ``test``→07).
* **G2 (slate)** is checked after stage 02 and before stage 03: a human must
  have confirmed ``production_slate.json``. Running ``01,02,03,04`` with no
  confirmed slate therefore executes 02 (producing the *proposed* slate) and
  then stops gracefully before 03 — no annotation work is wasted.
* **G4 (anchor labels)** is checked before stages 07/09: the anchor coding
  template from stage 05 must exist and be fully coded strict ``0/1``.
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

from binary_classifier.config import load_config
from binary_classifier.paths import PathRegistry
from binary_classifier.qc.preflight import validate_gates

# ── Stage registry ───────────────────────────────────────────────────────────

_STAGE_MODULES = {
    "01": ("binary_classifier.data.sample", "build_sample"),
    "02": ("binary_classifier.annotate.bakeoff_prompts", "run_bakeoff"),
    "03": ("binary_classifier.annotate.run_annotation", "run_annotation"),
    "04": ("binary_classifier.qc.agreement", "run_quality_check"),
    "05": ("binary_classifier.data.anchor", "build_anchor"),
}

# Graceful gate-failure exit code (distinct from CLI misuse, which is 1).
_GATE_EXIT = 2


# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline "
        "(01 sample → 02 bake-off → 03 annotate → 04 QC → 05 anchor sample) "
        "with human gates.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help=("Pass --force to stages 01 and 05 (overwrite human coding templates)."),
    )
    return parser.parse_args()


# ── Stage runner ─────────────────────────────────────────────────────────────


def _run_stage(
    stage_id: str,
    cfg,
    registry,
    annotate_limit: int | None,
    force: bool,
) -> None:
    """Import and execute a single stage."""
    module_name, func_name = _STAGE_MODULES[stage_id]
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    if stage_id in {"01", "05"}:
        func(cfg, registry, force=force)
    elif stage_id == "03" and annotate_limit is not None:
        func(cfg, registry, limit=annotate_limit)
    else:
        func(cfg, registry)


def _report_gate(title: str, problems: list[str], hint: str | None = None) -> None:
    """Print a gate failure to stderr."""
    print(
        f"\n✗ Gate {title} failed — stopping before any further work:",
        file=sys.stderr,
    )
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    if hint:
        print(f"\n  Next: {hint}", file=sys.stderr)


# ── Orchestration core ───────────────────────────────────────────────────────


def run_pipeline(
    cfg,
    registry,
    requested: set[str],
    annotate_limit: int | None = None,
    force: bool = False,
) -> None:
    """Run the requested stages in order, enforcing human gates.

    Exits the process with code 2 if a gate fails (graceful stop). Stage 01,
    when requested, always runs first so the coding template exists for the G1
    check.
    """
    # Stage 01 — produces manifests + the human coding template.
    if "01" in requested:
        print("Running stage 01 ...")
        _run_stage("01", cfg, registry, annotate_limit, force)

    # Gate G1 (labels) — before any model stage. Only the label gates are
    # checked here (not G2), so stage 02 can still run to produce the proposed
    # slate even when no confirmed slate exists yet.
    label_stages = requested & {"02", "04", "06", "07"}
    if label_stages:
        problems = validate_gates(cfg, registry, label_stages)
        if problems:
            _report_gate(
                "G1 (human labels)",
                problems,
                hint=(
                    f"open {registry.gold_coding_template}, fill human_label "
                    f"(0/1) for the listed split(s), then re-run."
                ),
            )
            sys.exit(_GATE_EXIT)

    # Stage 02 — bake-off; writes the *proposed* (unconfirmed) slate.
    if "02" in requested:
        print("Running stage 02 ...")
        _run_stage("02", cfg, registry, annotate_limit, force)

    # Gate G2 (confirmed slate) — after 02, before 03. No annotation work has
    # been done yet, so stopping here wastes nothing.
    if "03" in requested:
        problems = validate_gates(cfg, registry, {"03"})
        if problems:
            _report_gate(
                "G2 (confirmed production slate)",
                problems,
                hint=(
                    f"review {registry.bakeoff_results}, copy "
                    f"{registry.proposed_slate} to {registry.production_slate}, "
                    f"set 'confirmed': true, then run --stages 03,04."
                ),
            )
            sys.exit(_GATE_EXIT)

    # Stages 03, 04, and 05 — annotation, QC freeze, and anchor sampling.
    for stage_id in ("03", "04", "05"):
        if stage_id in requested:
            print(f"Running stage {stage_id} ...")
            _run_stage(stage_id, cfg, registry, annotate_limit, force)

    # Gate G4 (anchor labels) — before future anchor-dependent stages.
    anchor_stages = requested & {"07", "09"}
    if anchor_stages:
        problems = validate_gates(cfg, registry, anchor_stages)
        if problems:
            _report_gate(
                "G4 (anchor labels)",
                problems,
                hint=(
                    f"open {registry.anchor_coding_template}, fill human_label "
                    "(0/1) for every anchor row, then re-run."
                ),
            )
            sys.exit(_GATE_EXIT)

    # TODO(PR-3): add G3 test-unlock validation before stage 07 test evaluation.


# ── Main entrypoint ───────────────────────────────────────────────────────────


def main() -> None:
    """Run the requested pipeline stages in order."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    cfg = load_config(args.config)
    registry = PathRegistry(args.config)

    requested = {s.strip() for s in args.stages.split(",")}
    invalid = requested - set(_STAGE_MODULES.keys())
    if invalid:
        print(f"Invalid stage(s): {invalid}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(cfg, registry, requested, args.annotate_limit, args.force)


if __name__ == "__main__":
    main()
