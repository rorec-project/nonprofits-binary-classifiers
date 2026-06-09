"""Stratified, positive-enriched sampling for silver, gold, and human splits.

All sampling uses a fixed random seed for reproducibility. The silver pool is
drawn from HIGH+MEDIUM missions (``Q`` ≥3.0), stratified across the 26 NTEE
major groups with proportional allocation, floors for thin strata (V/Y/U), caps
for fat strata (B/P), and positive enrichment to ~30-40 % per stratum.

The gold set deliberately retains boundary cases (saint-named secular,
spiritual-not-religious, generic ministry, faith-heritage) and is carved into
prompt-dev, validation, and frozen-test splits.

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

    Args:
        df: DataFrame with columns ``EIN2``, ``mission_text``,
            ``ntee_major_group``, and ``Q``.
        target_size: Total rows desired in the silver pool.
        seed: Random seed for reproducibility.
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

    thresholds = QThresholdsConfig()
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

        sampled_pos = (
            pos_df.sample(n=n_pos_target, random_state=rng)
            if n_pos_target > 0
            else pos_df.iloc[0:0]
        )
        sampled_neg = (
            neg_df.sample(n=n_neg_target, random_state=rng)
            if n_neg_target > 0
            else neg_df.iloc[0:0]
        )
        stratum_sample = pd.concat([sampled_pos, sampled_neg], ignore_index=True)

        # Inclusion probability = n_target / pool_size for this stratum
        inclusion_prob = n_target / max(len(stratum_df), 1)
        stratum_sample["inclusion_prob"] = inclusion_prob
        sampled_rows.append(stratum_sample)

    silver = pd.concat(sampled_rows, ignore_index=True)
    silver = silver.sample(frac=1, random_state=rng).reset_index(drop=True)
    return silver


def build_gold_set(
    df: pd.DataFrame,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    """Construct the gold set (~400) from HIGH+MEDIUM missions.

    The gold set is deliberately diverse: each stratum receives a mix of
    clear positives, clear negatives, and boundary cases (saint-named
    secular, spiritual-not-religious, generic ministry, faith-heritage).
    Boundary cases are heuristically flagged so that they are intentionally
    retained for human coding.

    Args:
        df: DataFrame with columns ``EIN2``, ``mission_text``,
            ``ntee_major_group``, and ``Q``.
        target_size: Total rows desired in the gold set.
        seed: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with columns ``tier``, ``inclusion_prob``,
        and ``is_positive_enriched``.

    """
    thresholds = QThresholdsConfig()
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

        # Draw from each category
        pos_df = stratum_df[stratum_df["is_positive_enriched"]]
        neg_df = stratum_df[~stratum_df["is_positive_enriched"]]
        bnd_df = stratum_df[stratum_df["boundary_type"] != "none"]
        other_df = stratum_df[stratum_df["boundary_type"] == "none"]

        draws: list[pd.DataFrame] = []
        draws.append(_safe_sample(pos_df, n_clear_pos, rng))
        draws.append(_safe_sample(neg_df, n_clear_neg, rng))
        draws.append(_safe_sample(bnd_df, n_boundary, rng))
        draws.append(_safe_sample(other_df, n_other, rng))

        stratum_sample = pd.concat(draws, ignore_index=True).drop_duplicates(
            subset=["EIN2"], keep="first"
        )
        inclusion_prob = n_target / max(len(stratum_df), 1)
        stratum_sample["inclusion_prob"] = inclusion_prob
        sampled_rows.append(stratum_sample)

    gold = pd.concat(sampled_rows, ignore_index=True)
    gold = gold.sample(frac=1, random_state=rng).reset_index(drop=True)
    return gold


def split_human_sets(
    gold_df: pd.DataFrame,
    prompt_dev_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the gold set into prompt-dev, validation, and frozen test.

    The prompt-dev split is drawn first (fixed size), then the remainder
    is divided evenly between validation and frozen test. All splits are
    stratified by ``ntee_major_group`` to preserve sector coverage.

    Args:
        gold_df: The gold-set DataFrame.
        prompt_dev_size: Number of rows for prompt-dev.
        seed: Random seed.

    Returns:
        Tuple of (prompt_dev, validation, test) DataFrames.

    """
    rng = np.random.default_rng(seed=seed)
    df = gold_df.copy()
    n_groups = df["ntee_major_group"].nunique()
    per_group = prompt_dev_size // n_groups
    extra = prompt_dev_size % n_groups

    # Stratified prompt-dev sample — deterministic proportional allocation
    prompt_dev_parts: list[pd.DataFrame] = []
    groups = sorted(df["ntee_major_group"].unique())
    for i, grp in enumerate(groups):
        g = df[df["ntee_major_group"] == grp]
        n_sample = per_group + (1 if i < extra else 0)
        n_sample = min(n_sample, len(g))
        if n_sample > 0:
            prompt_dev_parts.append(g.sample(n=n_sample, random_state=rng))
    prompt_dev = pd.concat(prompt_dev_parts, ignore_index=True)

    remaining = df[~df["EIN2"].isin(prompt_dev["EIN2"])].copy()

    # Split remaining evenly between validation and test
    n_val = len(remaining) // 2
    val = remaining.sample(n=n_val, random_state=rng).reset_index(drop=True)
    test = remaining[~remaining["EIN2"].isin(val["EIN2"])].reset_index(drop=True)

    prompt_dev["split"] = "prompt_dev"
    val["split"] = "validation"
    test["split"] = "test"

    return prompt_dev, val, test


def build_sample(cfg: "BinaryClassifierConfig", registry: "PathRegistry") -> None:
    """Build the silver pool, gold set, and human splits and write manifests.

    Steps:
        1. Load missions via :func:`load_missions`.
        2. Compute quality scores ``Q`` and assign tiers.
        3. Flag religious-lexicon hits and apply the positive-protective
           rescue rule.
        4. Build the silver pool (~20 k) stratified across 26 NTEE major
           groups with proportional allocation, floors for thin strata,
           caps for fat strata, and positive enrichment to ~30--40 %.
        5. Build the gold set (~400) deliberately retaining boundary cases.
        6. Split gold into prompt-dev (~50), validation (~175), and frozen
           test (~175).
        7. Write manifest CSVs to the registry paths.

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved manifest paths.

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
    )

    logger.info(
        "Building gold set (target=%d)...",
        cfg.sample_sizes.gold,
    )
    gold = build_gold_set(
        df,
        target_size=cfg.sample_sizes.gold,
        seed=cfg.SEED,
    )

    logger.info("Splitting gold into human sets...")
    prompt_dev, validation, test = split_human_sets(
        gold,
        prompt_dev_size=cfg.sample_sizes.prompt_dev,
        seed=cfg.SEED,
    )

    registry.ensure_dirs()

    _write_manifest(silver, "silver", registry.silver_manifest)
    _write_split_manifest(prompt_dev, "prompt_dev", registry)
    _write_split_manifest(validation, "validation", registry)
    _write_split_manifest(test, "test", registry)

    gold_all = pd.concat([prompt_dev, validation, test])
    _write_manifest(gold_all, None, registry.gold_manifest)
    logger.info("Done.")


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
