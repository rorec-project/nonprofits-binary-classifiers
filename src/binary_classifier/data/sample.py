"""Stratified, positive-enriched sampling for silver, gold, and human splits.

All sampling uses a fixed random seed for reproducibility. The silver pool is
drawn from HIGH+MEDIUM missions (``Q`` ≥3.0), stratified across the 26 NTEE
major groups with proportional allocation, floors for thin strata (V/Y/U), caps
for fat strata (B/P), and positive enrichment to ~30-40 % per stratum.

The gold set deliberately retains boundary cases (saint-named secular,
spiritual-not-religious, generic ministry, faith-heritage) and is carved into
prompt-dev, validation, frozen-test, and monitor splits.

"""

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from binary_classifier.config import QThresholdsConfig
from binary_classifier.data.load import load_missions
from binary_classifier.data.quality import (
    assign_tier,
    compute_quality_score,
    has_religious_lexicon,
    is_positive_rescue,
)

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_FLOOR_GROUPS = {"V", "Y", "U"}
_CAP_GROUPS = {"B", "P"}
_FLOOR_SIZE = 200
_CAP_SIZE = 2_500
_TARGET_POSITIVE_SHARE = 0.35


# ── Public API ─────────────────────────────────────────────────────────────────


def build_silver_pool(
    df: pd.DataFrame,
    target_size: int,
    seed: int,
    thresholds: QThresholdsConfig,
    floor_groups: set[str] | None = None,
    cap_groups: set[str] | None = None,
    floor_size: int = _FLOOR_SIZE,
    cap_size: int = _CAP_SIZE,
    target_positive_share: float = _TARGET_POSITIVE_SHARE,
) -> pd.DataFrame:
    """Construct the silver pool (~20 k) from HIGH+MEDIUM missions.

    The function performs stratified random sampling across all 26 NTEE major
    groups (A-Z). Proportional allocation is adjusted with floors for thin
    strata and caps for fat strata. Within each stratum, positive missions
    (religious-lexicon hit or rescued) are enriched to approximately
    ``target_positive_share``.
    ``inclusion_prob`` is stored at the stratum × enrichment-cell level because
    positives and negatives are sampled at different rates; downstream design
    weights must invert that cell probability rather than a stratum marginal.

    Args:
        df: DataFrame with columns ``EIN2``, ``mission_text``,
            ``ntee_major_group``, and ``Q``.
        target_size: Total rows desired in the silver pool.
        seed: Random seed for reproducibility.
        thresholds: Configured ``Q`` tier thresholds; passed explicitly so the
            sampling frame matches the YAML rather than hidden defaults.
        floor_groups: Strata that receive a floor.
        cap_groups: Strata that receive a cap.
        floor_size: Minimum rows per floor group.
        cap_size: Maximum rows per cap group.
        target_positive_share: Target share of positives per stratum
            (default 0.35, i.e. 35 %).

    Returns:
        Sampled DataFrame with the same columns as ``df`` plus
        ``tier``, ``inclusion_prob``, and ``is_positive_enriched``.

    """
    if floor_groups is None:
        floor_groups = _FLOOR_GROUPS
    if cap_groups is None:
        cap_groups = _CAP_GROUPS

    df = df.copy()
    df["tier"] = df["Q"].apply(lambda q: assign_tier(q, thresholds))

    # Restrict to HIGH + MEDIUM (Q ≥ 3.0)
    pool = df[df["tier"].isin(["HIGH", "MEDIUM"])].copy()

    # Only A-Z strata
    pool = pool[pool["ntee_major_group"].str.match(r"^[A-Z]$")]

    # Mark likely positives (lexicon or rescue rule)
    pool["is_positive_enriched"] = pool.apply(
        lambda row: (
            has_religious_lexicon(row["mission_text"])
            or is_positive_rescue(row["mission_text"], row["Q"])
        ),
        axis=1,
    )

    # Compute stratum-level allocation
    stratum_counts = pool["ntee_major_group"].value_counts().sort_index()
    targets = _allocate_stratum_targets(
        stratum_counts,
        target_size,
        floor_groups,
        cap_groups,
        floor_size,
        cap_size,
    )

    rng = np.random.default_rng(seed=seed)
    sampled_rows: list[pd.DataFrame] = []

    for stratum, n_target in targets.items():
        stratum_df = pool[pool["ntee_major_group"] == stratum].copy()
        if stratum_df.empty:
            continue

        pos_df = stratum_df[stratum_df["is_positive_enriched"]]
        neg_df = stratum_df[~stratum_df["is_positive_enriched"]]

        n_pos_target = min(
            math.floor(n_target * target_positive_share),
            len(pos_df),
        )
        n_neg_target = min(n_target - n_pos_target, len(neg_df))

        # If negatives insufficient, backfill with positives
        if n_neg_target < (n_target - n_pos_target) and len(pos_df) > n_pos_target:
            n_pos_target = min(n_target - n_neg_target, len(pos_df))

        # If still short, take everything available
        if n_pos_target + n_neg_target < n_target:
            n_pos_target = len(pos_df)
            n_neg_target = min(n_target - n_pos_target, len(neg_df))

        pos_inclusion_prob = _cell_inclusion_prob(pos_df, n_pos_target)
        neg_inclusion_prob = _cell_inclusion_prob(neg_df, n_neg_target)

        sampled_pos = (
            pos_df.sample(n=n_pos_target, random_state=rng).copy()
            if n_pos_target > 0
            else pos_df.iloc[0:0].copy()
        )
        sampled_neg = (
            neg_df.sample(n=n_neg_target, random_state=rng).copy()
            if n_neg_target > 0
            else neg_df.iloc[0:0].copy()
        )
        sampled_pos["inclusion_prob"] = pos_inclusion_prob
        sampled_neg["inclusion_prob"] = neg_inclusion_prob
        stratum_sample = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
        sampled_rows.append(stratum_sample)

    silver = pd.concat(sampled_rows, ignore_index=True)
    silver = silver.sample(frac=1, random_state=rng).reset_index(drop=True)
    return silver


def build_gold_set(
    df: pd.DataFrame,
    target_size: int,
    seed: int,
    thresholds: QThresholdsConfig,
) -> pd.DataFrame:
    """Construct the configured gold set from HIGH+MEDIUM missions.

    The gold set is deliberately diverse: each stratum receives a mix of
    clear positives, clear negatives, and boundary cases (saint-named
    secular, spiritual-not-religious, generic ministry, faith-heritage).
    Boundary cases are heuristically flagged so that they are intentionally
    retained for human coding.
    ``inclusion_prob`` records the quota-cell draw rate for audit diagnostics;
    gold boundary-quota weights are diagnostic, not for population estimation.

    Args:
        df: DataFrame with columns ``EIN2``, ``mission_text``,
            ``ntee_major_group``, and ``Q``.
        target_size: Total rows desired in the gold set.
        seed: Random seed for reproducibility.
        thresholds: Configured ``Q`` tier thresholds; passed explicitly so the
            human-coding frame follows the same YAML boundary as stage 01.

    Returns:
        Sampled DataFrame with columns ``tier``, ``inclusion_prob``,
        and ``is_positive_enriched``.

    """
    df = df.copy()
    df["tier"] = df["Q"].apply(lambda q: assign_tier(q, thresholds))
    pool = df[df["tier"].isin(["HIGH", "MEDIUM"])].copy()
    pool = pool[pool["ntee_major_group"].str.match(r"^[A-Z]$")]

    # Classification for sampling diversity
    pool["is_positive_enriched"] = pool["mission_text"].apply(has_religious_lexicon)
    pool["is_rescue"] = pool.apply(
        lambda row: is_positive_rescue(row["mission_text"], row["Q"]),
        axis=1,
    )
    pool["boundary_type"] = pool["mission_text"].apply(_classify_boundary)
    pool.loc[pool["is_rescue"], "boundary_type"] = "rescue"

    rng = np.random.default_rng(seed=seed)

    # Target ~15 per stratum (26 * 15 = 390), distribute remainder
    n_strata = 26
    base_per_stratum = target_size // n_strata
    remainder = target_size % n_strata

    # Give the extra rows to the largest strata (by pool size)
    stratum_order = pool["ntee_major_group"].value_counts().index.tolist()
    extra_strata = set(stratum_order[:remainder])

    sampled_rows: list[pd.DataFrame] = []

    for stratum in sorted(pool["ntee_major_group"].unique()):
        stratum_df = pool[pool["ntee_major_group"] == stratum].copy()
        if stratum_df.empty:
            continue

        n_target = base_per_stratum + (1 if stratum in extra_strata else 0)

        # Diversity targets per stratum
        n_boundary = max(1, math.ceil(n_target * 0.35))
        n_clear_pos = max(1, math.floor(n_target * 0.25))
        n_clear_neg = max(1, math.floor(n_target * 0.25))
        n_other = n_target - n_boundary - n_clear_pos - n_clear_neg

        # Draw from each category. Gold's boundary-quota weights are diagnostic
        # only, not for population estimation, because this set deliberately
        # over-represents ambiguous cases for human review.
        pos_df = stratum_df[stratum_df["is_positive_enriched"]]
        neg_df = stratum_df[~stratum_df["is_positive_enriched"]]
        bnd_df = stratum_df[stratum_df["boundary_type"] != "none"]
        other_df = stratum_df[stratum_df["boundary_type"] == "none"]

        draws: list[pd.DataFrame] = []
        draws.append(_safe_sample_with_inclusion_prob(pos_df, n_clear_pos, rng))
        draws.append(_safe_sample_with_inclusion_prob(neg_df, n_clear_neg, rng))
        draws.append(_safe_sample_with_inclusion_prob(bnd_df, n_boundary, rng))
        draws.append(_safe_sample_with_inclusion_prob(other_df, n_other, rng))

        stratum_sample = pd.concat(draws, ignore_index=True).drop_duplicates(
            subset=["EIN2"],
            keep="first",
        )
        sampled_rows.append(stratum_sample)

    gold = pd.concat(sampled_rows, ignore_index=True)
    gold = gold.sample(frac=1, random_state=rng).reset_index(drop=True)
    return gold


def split_human_sets(
    gold_df: pd.DataFrame,
    prompt_dev_size: int,
    monitor_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the gold set into prompt-dev, validation, test, and monitor.

    The prompt-dev split is drawn first (fixed size), then the monitor split is
    drawn as an incremental held-out slice for drift checks. The remaining rows
    are divided evenly between validation and frozen test so bumping ``gold`` by
    ``monitor_size`` preserves the validation/test sizes. Prompt-dev and monitor
    are stratified by ``ntee_major_group`` to preserve sector coverage.

    Adding a monitor split changes deterministic split tags. If an existing
    coding template is protected from clobbering, regenerate it before human
    coding or after any re-split so manifests and labels stay aligned.

    Args:
        gold_df: The gold-set DataFrame.
        prompt_dev_size: Number of rows for prompt-dev.
        monitor_size: Number of rows reserved for drift-monitor canaries.
        seed: Random seed.

    Returns:
        Tuple of (prompt_dev, validation, test, monitor) DataFrames.

    """
    rng = np.random.default_rng(seed=seed)
    df = gold_df.copy()

    prompt_dev = _sample_stratified_by_group(df, prompt_dev_size, rng)

    remaining = df[~df["EIN2"].isin(prompt_dev["EIN2"])].copy()

    # Draw monitor before validation/test because it is an incremental gold
    # holdout, not a fourth-way shrink of the human validation/test frame.
    monitor = _sample_stratified_by_group(remaining, monitor_size, rng)
    remaining = remaining[~remaining["EIN2"].isin(monitor["EIN2"])].copy()

    # Split remaining evenly between validation and test
    n_val = len(remaining) // 2
    val = remaining.sample(n=n_val, random_state=rng).reset_index(drop=True)
    test = remaining[~remaining["EIN2"].isin(val["EIN2"])].reset_index(drop=True)

    prompt_dev["split"] = "prompt_dev"
    val["split"] = "validation"
    test["split"] = "test"
    monitor["split"] = "monitor"

    return prompt_dev, val, test, monitor


def build_sample(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    force: bool = False,
) -> None:
    """Build the silver pool, gold set, and human splits and write manifests.

    Steps:
        1. Load missions via :func:`load_missions`.
        2. Compute quality scores ``Q`` and assign tiers.
        3. Flag religious-lexicon hits and apply the positive-protective
           rescue rule.
        4. Build the silver pool (~20 k) stratified across 26 NTEE major
           groups with proportional allocation, floors for thin strata,
           caps for fat strata, and positive enrichment to ~30--40 %.
        5. Build the gold set deliberately retaining boundary cases plus the
           incremental monitor slice.
        6. Split gold into prompt-dev (~50), monitor (~50), validation (~175),
           and frozen test (~175).
        7. Write manifest CSVs to the registry paths.
        8. Emit the in-place human-coding template ``gold_to_code.csv`` (unless
           it already exists and ``force`` is ``False`` — clobber protection
           for human-entered labels).

    Re-split caveat: adding the monitor slice changes split tags. If a protected
    coding template already exists, regenerate it before coding or after any
    re-split so the manifests and template do not disagree.

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved manifest paths.
        force: Overwrite an existing gold coding template (discarding any
            human-entered labels). Defaults to ``False``.

    Returns:
        None. Writes CSV manifests to ``registry`` paths.

    """
    logger.info("Loading missions...")
    df = load_missions(cfg)

    logger.info("Computing Q scores...")
    df["Q"] = df["mission_text"].apply(compute_quality_score)

    logger.info("Assigning tiers...")
    df["tier"] = df["Q"].apply(lambda q: assign_tier(q, cfg.q_thresholds))

    logger.info("Flagging religious lexicon and rescue rule...")
    df["has_religious_lexicon"] = df["mission_text"].apply(has_religious_lexicon)
    df["is_positive_rescue"] = df.apply(
        lambda row: is_positive_rescue(row["mission_text"], row["Q"]),
        axis=1,
    )

    logger.info(
        "Building silver pool (target=%d)...",
        cfg.sample_sizes.silver,
    )
    silver = build_silver_pool(
        df,
        target_size=cfg.sample_sizes.silver,
        seed=cfg.SEED,
        thresholds=cfg.q_thresholds,
    )

    logger.info(
        "Building gold set (target=%d)...",
        cfg.sample_sizes.gold,
    )
    gold = build_gold_set(
        df,
        target_size=cfg.sample_sizes.gold,
        seed=cfg.SEED,
        thresholds=cfg.q_thresholds,
    )

    logger.info("Splitting gold into human sets...")
    prompt_dev, validation, test, monitor = split_human_sets(
        gold,
        prompt_dev_size=cfg.sample_sizes.prompt_dev,
        monitor_size=cfg.sample_sizes.monitor,
        seed=cfg.SEED,
    )

    registry.ensure_dirs()

    _write_manifest(silver, "silver", registry.silver_manifest)
    _write_split_manifest(prompt_dev, "prompt_dev", registry)
    _write_split_manifest(validation, "validation", registry)
    _write_split_manifest(test, "test", registry)
    _write_split_manifest(monitor, "monitor", registry)

    gold_all = pd.concat([prompt_dev, validation, test, monitor], ignore_index=True)
    _write_manifest(gold_all, None, registry.gold_manifest)

    # Clobber protection can preserve an already-coded template even after a
    # monitor re-split changes manifests, so humans should regenerate before
    # coding or intentionally use --force after changing split geometry.
    _write_gold_coding_template(gold_all, registry, force=force)
    logger.info("Done.")


def _write_gold_coding_template(
    gold_all: pd.DataFrame,
    registry: "PathRegistry",
    force: bool = False,
) -> None:
    """Write the in-place human-coding template ``gold_to_code.csv``.

    Columns: ``EIN2, split, text, human_label`` (the last blank for the human
    to fill with strict ``0``/``1``). The text is carried inline so the human
    can code without re-joining the upstream parquet.

    Clobber protection: if the template already exists it is **not** rewritten
    unless ``force`` is ``True`` — this preserves any human-entered labels.

    Args:
        gold_all: Concatenated prompt-dev + validation + test + monitor rows
            (with ``mission_text`` and ``split`` columns).
        registry: Path registry exposing ``gold_coding_template``.
        force: Overwrite an existing template.

    """
    path = registry.gold_coding_template
    if path.exists() and not force:
        existing = pd.read_csv(path)
        n_coded = (
            int(existing["human_label"].notna().sum())
            if "human_label" in existing.columns
            else 0
        )
        logger.info(
            "Gold coding template present at %s, %d coded — skipping "
            "(use --force to regenerate).",
            path,
            n_coded,
        )
        return

    template = pd.DataFrame(
        {
            "EIN2": gold_all["EIN2"].to_numpy(),
            "split": gold_all["split"].to_numpy(),
            "text": gold_all["mission_text"].to_numpy(),
            "human_label": "",
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    logger.info("Wrote gold coding template (%d rows) to %s", len(template), path)


def _write_manifest(
    df: pd.DataFrame,
    split_name: str | None,
    path: Path,
) -> None:
    """Write a single manifest CSV with canonical columns."""
    cols = [
        "EIN2",
        "ntee_major_group",
        "tier",
        "inclusion_prob",
        "is_positive_enriched",
    ]
    if split_name is None and "split" in df.columns:
        cols.append("split")
    manifest = df[cols].rename(columns={"ntee_major_group": "stratum"}).copy()
    manifest["is_positive_enriched"] = manifest["is_positive_enriched"].astype(int)
    if split_name is not None:
        manifest["split"] = split_name
    elif "split" not in manifest.columns:
        manifest["split"] = "gold"
    manifest.to_csv(path, index=False)


def _write_split_manifest(
    df: pd.DataFrame,
    split_name: str,
    registry: "PathRegistry",
) -> None:
    """Write a split manifest using registry path lookup."""
    path = getattr(registry, f"{split_name}_manifest")
    _write_manifest(df, split_name, path)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _sample_stratified_by_group(
    df: pd.DataFrame,
    target_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw a fixed human split while preserving NTEE sector coverage.

    Prompt-dev and monitor need deterministic, disjoint sub-slices that touch
    as many NTEE major groups as the target size allows. This helper keeps the
    allocation logic identical for both slices before validation/test consume
    the remainder.
    """
    if target_size <= 0 or df.empty:
        return df.iloc[0:0].copy()

    groups = sorted(df["ntee_major_group"].unique())
    n_groups = len(groups)
    per_group = target_size // n_groups
    extra = target_size % n_groups

    parts: list[pd.DataFrame] = []
    for i, grp in enumerate(groups):
        g = df[df["ntee_major_group"] == grp]
        n_sample = per_group + (1 if i < extra else 0)
        n_sample = min(n_sample, len(g))
        if n_sample > 0:
            parts.append(g.sample(n=n_sample, random_state=rng))

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _allocate_stratum_targets(
    pool_counts: pd.Series,
    total_target: int,
    floor_groups: set[str],
    cap_groups: set[str],
    floor_size: int,
    cap_size: int,
) -> pd.Series:
    """Proportional allocation with floor / cap adjustments.

    Computes a target count for each stratum. Capped groups are clamped,
    floor groups are boosted, and the residual is redistributed
    proportionally among the remaining strata.
    """
    total_pool = pool_counts.sum()
    targets = (pool_counts / total_pool * total_target).round().astype(int)

    # Apply caps first
    for g in cap_groups:
        if g in targets.index:
            targets[g] = min(targets[g], cap_size)

    # Apply floors
    for g in floor_groups:
        if g in targets.index:
            targets[g] = max(targets[g], floor_size)
            # Never exceed available pool
            targets[g] = min(targets[g], pool_counts.get(g, 0))

    # Rebalance deficit/surplus among non-capped, non-floor strata
    current_total = targets.sum()
    deficit = total_target - current_total

    adjustable = [
        g for g in targets.index if g not in floor_groups and g not in cap_groups
    ]

    if deficit != 0 and adjustable:
        adj_pool = pool_counts[adjustable]
        adj_total = adj_pool.sum()
        for g in adjustable:
            delta = round(deficit * pool_counts[g] / adj_total)
            targets[g] = max(0, targets[g] + delta)

        # Final clean-up: nudge the largest adjustable group
        current_total = targets.sum()
        deficit = total_target - current_total
        if deficit != 0 and adjustable:
            largest = max(adjustable, key=lambda g: targets[g])
            targets[largest] += deficit

    # Ensure no target exceeds pool
    targets = targets.clip(upper=pool_counts.reindex(targets.index).fillna(0))

    return targets


def _classify_boundary(text: str) -> str:
    """Heuristic boundary-case classifier for gold-set diversity.

    Returns one of ``saint``, ``spiritual``, ``ministry``, ``faith_heritage``,
    ``boilerplate_religious``, or ``none``.
    """
    if not isinstance(text, str):
        return "none"

    t = text.lower()

    # Saint-named secular
    if re.search(r"\bst\.?\s+\w+", t) or re.search(r"\bsaint\s+\w+", t):
        if not re.search(r"\b(church|diocese|catholic|parish|christian)\b", t):
            return "saint"

    # Spiritual but not religious
    if re.search(r"\bspiritual\b", t) and not re.search(
        r"\b(christ|christian|jesus|gospel|bible|church|synagogue|mosque|temple|islam|muslim|jewish|hindu|buddhist)\b",
        t,
    ):
        return "spiritual"

    # Generic ministry without strong tradition
    if re.search(r"\bministr(y|ies)\b", t) and not re.search(
        r"\b(christ|christian|jesus|gospel|bible|church|catholic|baptist|methodist|lutheran|presbyterian|episcopal|pentecostal|jewish|synagogue|torah|rabbi|islam|muslim|mosque|quran|buddhist|hindu|temple|salvation army|seminary|chapel|diocese|orthodox)\b",
        t,
    ):
        return "ministry"

    # Faith-heritage / values framing
    if re.search(r"\bfaith\b", t) or re.search(r"\bjudeo-christian\b", t):
        if not re.search(
            r"\b(christ|christian|jesus|gospel|bible|church|synagogue|mosque|temple)\b",
            t,
        ):
            return "faith_heritage"

    # Boilerplate religious legal clause
    if re.search(r"\breligious,\s+charitable\s+and\s+educational\b", t):
        return "boilerplate_religious"

    return "none"


def _safe_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample ``n`` rows from ``df`` without error if ``n`` > len(df)."""
    n = min(n, max(len(df), 0))
    if n <= 0:
        return df.iloc[0:0]
    return df.sample(n=n, random_state=rng)


def _safe_sample_with_inclusion_prob(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample a quota cell and attach its realized inclusion probability.

    Gold uses positive, negative, boundary, and filler quota cells to stress-test
    human coding and prompt selection. The manifest therefore needs the draw
    rate for the cell that selected the row, not a stratum-wide marginal rate.
    """
    sample = _safe_sample(df, n, rng).copy()
    sample["inclusion_prob"] = _cell_inclusion_prob(df, n)
    return sample


def _cell_inclusion_prob(df: pd.DataFrame, n_target: int) -> float:
    """Return the realized per-cell sampling probability.

    Positive enrichment and boundary quotas sample cells within a stratum at
    different rates. Persisting that cell-level probability lets downstream
    estimators invert the manifest value when a design weight is appropriate.
    """
    if df.empty or n_target <= 0:
        return 0.0

    # design weight = 1/π, per-cell inclusion probability;
    # Horvitz & Thompson 1952
    return min(n_target, len(df)) / len(df)
