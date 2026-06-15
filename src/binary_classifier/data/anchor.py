"""Stage 05 full-frame anchor sampling.

The anchor sample is drawn from the full missions frame, including LOW-quality
rows that stage 01 deliberately excludes from silver/gold sampling. It is the
design-weighted human-coded sample used by later DSL/PPI-style prevalence and
calibration steps, so it writes its own manifest schema rather than reusing the
stage-01 manifest helper.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from binary_classifier.data.load import load_missions
from binary_classifier.data.quality import assign_tier, compute_quality_score

if TYPE_CHECKING:
    from pathlib import Path

    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


logger = logging.getLogger(__name__)

_ANCHOR_MANIFEST_COLUMNS = [
    "EIN2",
    "stratum",
    "tier",
    "ntee_major_group",
    "sample_prob",
    "split",
]
_ANCHOR_TEMPLATE_COLUMNS = ["EIN2", "tier", "text", "human_label"]


def build_anchor(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    force: bool = False,
) -> None:
    """Build the full-frame anchor sample and human-coding template.

    Args:
        cfg: Validated binary-classifier configuration.
        registry: Path registry exposing anchor output locations and upstream
            stage-01 manifests to exclude.
        force: Overwrite an existing anchor coding template, including any
            human-entered labels. Defaults to ``False``.

    Raises:
        RuntimeError: If the anchor coding template already contains human
            labels and ``force`` is ``False``.
        ValueError: If the requested sample is impossible for the available
            post-exclusion frame.
    """
    logger.info("Loading missions for anchor sample...")
    df = load_missions(cfg).copy()

    logger.info("Computing quality scores and tiers...")
    df["Q"] = df["mission_text"].map(compute_quality_score)
    df["tier"] = df["Q"].map(lambda q: assign_tier(float(q), cfg.q_thresholds))
    df["ntee_major_group"] = df["ntee_major_group"].astype(str).str.strip()
    df["stratum"] = df["tier"] + "|" + df["ntee_major_group"]

    excluded_ein2s = _load_stage01_ein2s(registry)
    ein2_norm = _normalize_ein2(df["EIN2"])
    keep_mask = ~ein2_norm.isin(excluded_ein2s)
    excluded_count = int((~keep_mask).sum())
    logger.info(
        "Excluded %d rows already present in silver/gold manifests "
        "(documented estimand note: anchor excludes prior human/silver rows).",
        excluded_count,
    )
    frame = df.loc[keep_mask].copy()

    target_n = int(cfg.anchor.n)
    if target_n < 0:
        raise ValueError("cfg.anchor.n must be non-negative")
    if target_n > len(frame):
        raise ValueError(
            f"Requested anchor sample n={target_n}, but only {len(frame)} rows "
            "remain after silver/gold exclusions.",
        )

    sampled = _sample_anchor_frame(
        frame,
        target_n=target_n,
        oversample_low_factor=cfg.anchor.oversample_low_factor,
        min_stratum_frame=cfg.anchor.min_stratum_frame,
        seed=cfg.SEED,
    )

    _raise_if_coded_template_exists(registry.anchor_coding_template, force=force)

    registry.ensure_dirs()
    _write_anchor_manifest(sampled, registry.anchor_manifest)
    _write_anchor_coding_template(sampled, registry.anchor_coding_template)
    logger.info("Anchor sample built (%d rows).", len(sampled))


def _sample_anchor_frame(
    frame: pd.DataFrame,
    target_n: int,
    oversample_low_factor: float,
    min_stratum_frame: int,
    seed: int,
) -> pd.DataFrame:
    """Draw the anchor sample with tier × NTEE allocation and design weights."""
    if target_n == 0:
        result = frame.iloc[0:0].copy()
        result["sample_prob"] = pd.Series(dtype=float)
        return result
    if frame.empty:
        raise ValueError("Cannot draw a non-empty anchor sample from an empty frame")

    stratum_counts = frame["stratum"].value_counts(sort=False).sort_index()
    targets = _allocate_anchor_targets(
        stratum_counts=stratum_counts,
        target_n=target_n,
        oversample_low_factor=oversample_low_factor,
        min_stratum_frame=min_stratum_frame,
    )

    rng = np.random.default_rng(seed=seed)
    sampled_parts: list[pd.DataFrame] = []
    for stratum, n_drawn in targets.items():
        if n_drawn <= 0:
            continue
        stratum_df = frame[frame["stratum"] == stratum]
        n_frame = int(stratum_counts.loc[stratum])
        sample = stratum_df.sample(n=int(n_drawn), random_state=rng).copy()
        sample["sample_prob"] = float(n_drawn) / float(n_frame)
        sampled_parts.append(sample)

    if not sampled_parts:
        return frame.iloc[0:0].copy()

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=rng).reset_index(drop=True)


def _allocate_anchor_targets(
    stratum_counts: pd.Series,
    target_n: int,
    oversample_low_factor: float,
    min_stratum_frame: int,
) -> pd.Series:
    """Allocate integer anchor draws across tier × NTEE strata.

    Allocation is proportional to stratum frame size, with LOW strata weighted by
    ``oversample_low_factor``. Eligible strata receive a lower bound of one draw
    when they contain at least ``min_stratum_frame`` frame rows, then remaining
    draws are renormalized proportionally while respecting each stratum's size.
    """
    if target_n == 0:
        return pd.Series(0, index=stratum_counts.index, dtype=int)

    counts = stratum_counts.sort_index().astype(int)
    if target_n > int(counts.sum()):
        raise ValueError("Anchor target cannot exceed the post-exclusion frame size")

    weights = counts.astype(float)
    is_low = counts.index.to_series().astype(str).str.startswith("LOW|")
    weights.loc[is_low.to_numpy()] *= float(oversample_low_factor)

    lower = pd.Series(0, index=counts.index, dtype=int)
    lower.loc[counts >= int(min_stratum_frame)] = 1
    if int(lower.sum()) > target_n:
        raise ValueError(
            "Anchor floor allocation exceeds cfg.anchor.n; lower "
            "min_stratum_frame or increase anchor.n.",
        )

    return _bounded_largest_remainder(
        weights=weights,
        lower=lower,
        capacity=counts,
        target_n=target_n,
    )


def _bounded_largest_remainder(
    weights: pd.Series,
    lower: pd.Series,
    capacity: pd.Series,
    target_n: int,
) -> pd.Series:
    """Largest-remainder integer allocation with lower and upper bounds."""
    allocation = lower.copy().astype(int)
    if target_n > int(capacity.sum()):
        raise ValueError("Target allocation exceeds total capacity")

    remaining = target_n - int(allocation.sum())
    while remaining > 0:
        spare = capacity - allocation
        active = (spare > 0) & (weights > 0)
        if not bool(active.any()):
            raise ValueError("No stratum capacity remains for anchor allocation")

        active_weights = weights.loc[active]
        raw = active_weights / float(active_weights.sum()) * remaining
        base = np.floor(raw).astype(int)
        base = pd.Series(base, index=active_weights.index).clip(
            upper=spare.loc[active],
        )

        if int(base.sum()) > 0:
            allocation.loc[base.index] += base.astype(int)
            remaining = target_n - int(allocation.sum())
            if remaining == 0:
                break

        spare = capacity - allocation
        active = (spare > 0) & (weights > 0)
        if not bool(active.any()):
            break
        remainders = (raw - np.floor(raw)).loc[active]
        for stratum in sorted(
            remainders.index,
            key=lambda idx: (-float(remainders.loc[idx]), str(idx)),
        ):
            if remaining == 0:
                break
            if allocation.loc[stratum] < capacity.loc[stratum]:
                allocation.loc[stratum] += 1
                remaining -= 1

    if int(allocation.sum()) != target_n:
        raise ValueError("Failed to allocate the requested anchor sample size")
    return allocation.astype(int)


def _load_stage01_ein2s(registry: "PathRegistry") -> set[str]:
    """Load silver/gold EIN2s to exclude, tolerating absent manifests."""
    excluded: set[str] = set()
    for path in [registry.silver_manifest, registry.gold_manifest]:
        if not path.exists():
            continue
        manifest = pd.read_csv(path, usecols=["EIN2"])
        excluded.update(_normalize_ein2(manifest["EIN2"]).to_list())
    return excluded


def _normalize_ein2(values: pd.Series) -> pd.Series:
    """Normalize EIN2 values for cross-artifact set comparison."""
    return values.astype(str).str.strip()


def _raise_if_coded_template_exists(path: "Path", force: bool) -> None:
    """Refuse to overwrite a human-filled anchor template unless forced."""
    if force or not path.exists():
        return

    existing = pd.read_csv(path)
    if "human_label" not in existing.columns:
        return

    label_text = existing["human_label"].astype("string").str.strip()
    coded = label_text.notna() & (label_text != "")
    n_coded = int(coded.sum())
    if n_coded > 0:
        raise RuntimeError(
            f"Anchor coding template at {path} contains {n_coded} human labels; "
            "rerun with force=True to overwrite.",
        )


def _write_anchor_manifest(sampled: pd.DataFrame, path: "Path") -> None:
    """Write the stage-05 anchor manifest schema."""
    manifest = sampled.copy()
    manifest["split"] = "anchor"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest[_ANCHOR_MANIFEST_COLUMNS].to_csv(path, index=False)
    logger.info("Wrote anchor manifest (%d rows) to %s", len(manifest), path)


def _write_anchor_coding_template(sampled: pd.DataFrame, path: "Path") -> None:
    """Write the human-coding template for anchor rows."""
    template = pd.DataFrame(
        {
            "EIN2": sampled["EIN2"].to_numpy(),
            "tier": sampled["tier"].to_numpy(),
            "text": sampled["mission_text"].to_numpy(),
            "human_label": "",
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    template[_ANCHOR_TEMPLATE_COLUMNS].to_csv(path, index=False)
    logger.info("Wrote anchor coding template (%d rows) to %s", len(template), path)
