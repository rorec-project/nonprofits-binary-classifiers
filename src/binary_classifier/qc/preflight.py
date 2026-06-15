"""Pre-flight gates that fail gracefully before any GPU/API work.

Human checkpoints guard the pipeline:

* **G1 (labels)** — before stage 02, every label-dependent requested stage must
  have its human coding complete: ``prompt_dev`` for stage 02, ``validation``
  for stages 04/06, and ``test`` for stage 07. ``human_label`` is strict
  ``{0, 1}`` (no human abstain).
* **G2 (slate)** — before stage 03, a human-confirmed ``production_slate.json``
  must exist and list at least one model resolvable to a configured candidate.
* **G3 (test unlock)** — before stage 07, a human-confirmed
  ``test_unlock.json`` must match the current acceptance criteria and selected
  checkpoint SHA.
* **G4 (anchor labels)** — before stages 07/09, the anchor coding template must
  exist and every anchor row must have a strict ``{0, 1}`` human label.

:func:`validate_gates` is a pure check returning a list of human-readable
problems; the orchestrator decides when to call it and exits non-zero on any
problem (see ``scripts/run_pipeline.py``).
"""

import json
from typing import TYPE_CHECKING
from collections.abc import Iterable

import pandas as pd

from binary_classifier.config import load_slate, load_test_unlock

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

# Which human split each label-dependent stage needs coded.
_STAGE_SPLITS: dict[str, str] = {
    "02": "prompt_dev",
    "04": "validation",
    "06": "validation",
    "07": "test",
}

_TEMPLATE_COLS = {"EIN2", "split", "text", "human_label"}
_ANCHOR_TEMPLATE_COLS = {"EIN2", "tier", "text", "human_label"}


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

    # G4 — anchor labels for anchor-dependent stages.
    if stages & {"07", "09"}:
        problems.extend(_validate_anchor_labels(cfg, registry))

    # G3 — human test-unlock for frozen test evaluation.
    if "07" in stages:
        problems.extend(_validate_test_unlock(cfg, registry))

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


# ── G4: anchor human labels ──────────────────────────────────────────────────


def _validate_anchor_labels(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
) -> list[str]:
    """Check that the anchor coding template exists and is fully coded.

    Args:
        cfg: Validated configuration object (reserved for future G4 checks).
        registry: Path registry with resolved artifact paths.

    Returns:
        A list of G4 problem strings. Empty means the anchor labels are complete.

    """
    _ = cfg
    path = registry.anchor_coding_template
    if not path.exists():
        return [
            f"G4: anchor coding template not found at {path}. Run stage 05, then "
            "code human_label (0/1) for every anchor row.",
        ]

    df = pd.read_csv(path)
    missing_cols = _ANCHOR_TEMPLATE_COLS - set(df.columns)
    if missing_cols:
        return [f"G4: {path} missing columns {sorted(missing_cols)}."]

    problems: list[str] = []

    if registry.anchor_manifest.exists():
        manifest = pd.read_csv(registry.anchor_manifest)
        if "EIN2" not in manifest.columns:
            problems.append(f"G4: {registry.anchor_manifest} missing column 'EIN2'.")
        else:
            manifest_eins = set(manifest["EIN2"].astype(str).str.strip())
            template_eins = set(df["EIN2"].astype(str).str.strip())
            if template_eins != manifest_eins:
                n_extra = len(template_eins - manifest_eins)
                n_missing = len(manifest_eins - template_eins)
                problems.append(
                    f"G4: {path} EIN2 set does not match "
                    f"{registry.anchor_manifest} ({n_extra} extra, "
                    f"{n_missing} missing).",
                )

    n_bad = sum(not _is_strict_binary(v) for v in df["human_label"])
    if n_bad:
        problems.append(
            f"G4: anchor template has {n_bad}/{len(df)} row(s) with a blank or "
            "non-{0,1} human_label (strict 0/1 required, no abstain).",
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


# ── G3: human test unlock ────────────────────────────────────────────────────


def _validate_test_unlock(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
) -> list[str]:
    """Check that the frozen-test unlock matches current acceptance settings.

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved test-unlock and selected-model
            paths.

    Returns:
        A list of G3 problem strings. Empty means the frozen test may be run.

    """
    path = registry.test_unlock
    if not path.exists():
        return [
            f"G3: no confirmed test unlock at {path}. Review "
            f"{registry.selected_model}, create {path}, and set "
            "'confirmed': true.",
        ]

    try:
        unlock = load_test_unlock(path)
    except Exception as exc:
        return [f"G3: {path} is not valid test-unlock JSON: {exc}"]

    problems: list[str] = []
    if not unlock.confirmed:
        problems.append(
            f"G3: {path} is not confirmed (set 'confirmed': true after review).",
        )

    current_acceptance = cfg.evaluation.acceptance.model_dump()
    unlock_acceptance = unlock.acceptance.model_dump()
    drifted_fields = [
        field
        for field, current_value in current_acceptance.items()
        if unlock_acceptance.get(field) != current_value
    ]
    if drifted_fields:
        problems.append(
            f"G3: {path} acceptance snapshot differs from current "
            "cfg.evaluation.acceptance for field(s) "
            f"{sorted(drifted_fields)}.",
        )

    selected_path = registry.selected_model
    if not selected_path.exists():
        problems.append(
            f"G3: selected model artifact not found at {selected_path}. "
            "Run stage 06 and copy the selected-model skeleton after review.",
        )
        return problems

    try:
        selected_model = json.loads(selected_path.read_text())
    except Exception as exc:
        problems.append(f"G3: {selected_path} is not valid JSON: {exc}")
        return problems
    if not isinstance(selected_model, dict):
        problems.append(f"G3: {selected_path} must be a JSON object.")
        return problems

    selected_sha = str(selected_model.get("checkpoint_sha256", "")).strip()
    if not selected_sha:
        problems.append(f"G3: {selected_path} missing 'checkpoint_sha256'.")
    elif unlock.checkpoint_sha256 != selected_sha:
        problems.append(
            f"G3: {path} checkpoint_sha256 does not match "
            f"{selected_path} checkpoint_sha256.",
        )

    return problems


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
