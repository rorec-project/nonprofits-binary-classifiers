"""Pre-flight gates that fail gracefully before any GPU/API work.

Two human checkpoints guard the pipeline:

* **G1 (labels)** — before stage 02, every label-dependent requested stage must
  have its human coding complete: ``prompt_dev`` for stage 02, ``validation``
  for stage 04. ``human_label`` is strict ``{0, 1}`` (no human abstain).
* **G2 (slate)** — before stage 03, a human-confirmed ``production_slate.json``
  must exist and list at least one model resolvable to a configured candidate.

:func:`validate_gates` is a pure check returning a list of human-readable
problems; the orchestrator decides when to call it and exits non-zero on any
problem (see ``scripts/run_pipeline.py``).
"""

from typing import TYPE_CHECKING
from collections.abc import Iterable

import pandas as pd

from binary_classifier.config import load_slate

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

# Which human split each label-dependent stage needs coded.
_STAGE_SPLITS: dict[str, str] = {"02": "prompt_dev", "04": "validation"}

_TEMPLATE_COLS = {"EIN2", "split", "text", "human_label"}


def validate_gates(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    stages: Iterable[str],
) -> list[str]:
    """Validate the human gates relevant to the requested stages.

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved artifact paths.
        stages: The stage ids about to run (e.g. ``{"02", "03", "04"}``).

    Returns:
        A list of problem strings. Empty means all relevant gates pass.

    """
    stages = set(stages)
    problems: list[str] = []

    # G1 — labels for each label-dependent requested stage.
    needed_splits = [split for stage, split in _STAGE_SPLITS.items() if stage in stages]
    if needed_splits:
        problems.extend(_validate_labels(registry, needed_splits))

    # G2 — confirmed production slate (stage 03 only).
    if "03" in stages:
        problems.extend(_validate_slate(cfg, registry))

    return problems


# ── G1: human labels ─────────────────────────────────────────────────────────


def _validate_labels(registry: "PathRegistry", needed_splits: list[str]) -> list[str]:
    """Check the gold coding template for the needed coded splits."""
    path = registry.gold_coding_template
    if not path.exists():
        return [
            f"G1: gold coding template not found at {path}. Run stage 01, then "
            f"code human_label (0/1) for the {needed_splits} split(s).",
        ]

    df = pd.read_csv(path)
    missing_cols = _TEMPLATE_COLS - set(df.columns)
    if missing_cols:
        return [f"G1: {path} missing columns {sorted(missing_cols)}."]

    problems: list[str] = []

    # EIN2 set must match the gold manifest (no drift / partial template).
    if registry.gold_manifest.exists():
        manifest_eins = set(pd.read_csv(registry.gold_manifest)["EIN2"])
        template_eins = set(df["EIN2"])
        if template_eins != manifest_eins:
            n_extra = len(template_eins - manifest_eins)
            n_missing = len(manifest_eins - template_eins)
            problems.append(
                f"G1: {path} EIN2 set does not match {registry.gold_manifest} "
                f"({n_extra} extra, {n_missing} missing).",
            )

    for split in needed_splits:
        sub = df[df["split"] == split]
        if sub.empty:
            problems.append(f"G1: no '{split}' rows in {path}.")
            continue
        n_bad = sum(not _is_strict_binary(v) for v in sub["human_label"])
        if n_bad:
            problems.append(
                f"G1: '{split}' has {n_bad}/{len(sub)} row(s) with a blank or "
                f"non-{{0,1}} human_label (strict 0/1 required, no abstain).",
            )

    return problems


def _is_strict_binary(value: object) -> bool:
    """Return True only for a value equal to 0 or 1 (int/float/str forms)."""
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return False
    try:
        as_float = float(text)
    except ValueError:
        return False
    return as_float in (0.0, 1.0)


# ── G2: confirmed production slate ───────────────────────────────────────────


def _validate_slate(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
) -> list[str]:
    """Check that a human-confirmed production slate exists and resolves."""
    path = registry.production_slate
    if not path.exists():
        return [
            f"G2: no confirmed production slate at {path}. Review "
            f"{registry.bakeoff_results}, copy {registry.proposed_slate} to "
            f"{path}, edit it, and set 'confirmed': true.",
        ]

    try:
        slate = load_slate(path)
    except Exception as exc:
        return [f"G2: {path} is not valid slate JSON: {exc}"]

    problems: list[str] = []
    if not slate.confirmed:
        problems.append(
            f"G2: {path} is not confirmed (set 'confirmed': true after review).",
        )
    if not slate.models:
        problems.append(f"G2: {path} lists no models under 'models'.")

    configured = {c.id for c in cfg.model_slate.bakeoff_candidates}
    for model in slate.models:
        if model.id not in configured:
            problems.append(
                f"G2: production model {model.id!r} is not among the configured "
                f"bakeoff_candidates {sorted(configured)}.",
            )

    return problems
