"""Training-frame construction and deterministic data splits.

The stage-06 training stack consumes frozen silver labels from stage 04, but it
must rehydrate source text from the corpus and recompute soft targets from the
long/tidy annotation store. This module keeps those joins explicit and guards
against leakage from human-held-out manifests.
"""

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from binary_classifier.annotate.schema import AnnotationStore
from binary_classifier.data.load import load_missions

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

_TRAINING_COLUMNS = ["EIN2", "text", "ntee_major_group", "p_pos", "hard_label"]


def build_training_frame(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
) -> pd.DataFrame:
    """Build the silver-label frame used by training stages.

    Args:
        cfg: Validated task configuration.
        registry: Resolved path registry for the run.

    Returns:
        DataFrame with columns ``EIN2``, ``text``, ``ntee_major_group``,
        ``p_pos``, and ``hard_label``.

    Raises:
        ValueError: If required columns are missing, any retained silver row has
            no non-abstain votes in the annotation store, source text cannot be
            rehydrated, or the frame overlaps held-out gold/anchor manifests.
        FileNotFoundError: If required input artifacts are absent.

    """
    silver = _read_silver_labels(registry.silver_labels)
    silver = silver.dropna(subset=["silver_label"]).copy()
    if silver.empty:
        logger.info("No non-abstain silver labels found at %s", registry.silver_labels)
        return pd.DataFrame(columns=_TRAINING_COLUMNS)

    silver["EIN2"] = _normalize_ein2_series(silver["EIN2"])
    silver["hard_label"] = silver["silver_label"].astype(int)

    p_pos = _compute_p_pos(registry.annotation_store)
    missing_vote_mask = ~silver["EIN2"].isin(p_pos.index)
    if missing_vote_mask.any():
        missing_count = int(missing_vote_mask.sum())
        raise ValueError(
            "Silver labels contain non-abstain EIN2s with zero non-abstain "
            f"votes in the annotation store: {missing_count} EIN2(s).",
        )
    silver["p_pos"] = silver["EIN2"].map(p_pos).astype(float)

    missions = load_missions(cfg)[["EIN2", "mission_text", "ntee_major_group"]].copy()
    missions["EIN2"] = _normalize_ein2_series(missions["EIN2"])
    missions = missions.rename(columns={"mission_text": "text"})

    frame = silver[["EIN2", "p_pos", "hard_label"]].merge(
        missions[["EIN2", "text", "ntee_major_group"]],
        on="EIN2",
        how="left",
    )
    missing_source = frame["text"].isna() | frame["ntee_major_group"].isna()
    if missing_source.any():
        raise ValueError(
            "Could not rehydrate text/ntee_major_group for "
            f"{int(missing_source.sum())} silver EIN2(s).",
        )

    frame = frame[_TRAINING_COLUMNS].copy()
    _assert_no_manifest_overlap(frame, registry.gold_manifest, "gold")
    _assert_no_manifest_overlap(frame, registry.anchor_manifest, "anchor")

    logger.info("Built training frame with %d rows", len(frame))
    return frame


def split_dev(
    frame: pd.DataFrame,
    dev_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into train/dev partitions stratified by hard label.

    Args:
        frame: Training frame containing ``hard_label``.
        dev_fraction: Fraction of each label stratum assigned to dev.
        seed: Random seed for the split.

    Returns:
        Tuple of ``(train_df, dev_df)`` with reset indexes.

    Raises:
        ValueError: If ``dev_fraction`` is not in ``(0, 1)`` or the frame lacks
            ``hard_label``.

    """
    if "hard_label" not in frame.columns:
        raise ValueError("frame must contain a hard_label column.")
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    dev_indices: list[int] = []
    for _, group in frame.groupby("hard_label", sort=True):
        idx = group.index.to_numpy()
        shuffled = rng.permutation(idx)
        n_dev = int(round(len(shuffled) * dev_fraction))
        if len(shuffled) > 1:
            n_dev = min(max(n_dev, 1), len(shuffled) - 1)
        dev_indices.extend(shuffled[:n_dev].tolist())

    dev_index = pd.Index(dev_indices)
    dev_df = frame.loc[dev_index].copy()
    train_df = frame.drop(index=dev_index).copy()
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)


def subset_fraction(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Return a deterministic, label-stratified nested subset.

    Rows are sorted within each ``hard_label`` stratum by a stable hash of
    ``(seed, EIN2)`` and the requested prefix is returned. Therefore, for a
    fixed seed, smaller fractions are strict prefixes of larger fractions.

    Args:
        frame: Training frame containing ``EIN2`` and ``hard_label``.
        fraction: Fraction in ``(0, 1]`` to keep.
        seed: Seed included in the stable hash.

    Returns:
        Deterministic subset with reset index.

    Raises:
        ValueError: If required columns are missing or ``fraction`` is outside
            ``(0, 1]``.

    """
    required = {"EIN2", "hard_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    selected: list[pd.DataFrame] = []
    work = frame.copy()
    work["_ein2_norm"] = _normalize_ein2_series(work["EIN2"])
    work["_stable_hash"] = work["_ein2_norm"].map(
        lambda ein2: _stable_hash(ein2, seed),
    )

    for _, group in work.groupby("hard_label", sort=True):
        ordered = group.sort_values(["_stable_hash", "_ein2_norm"], kind="mergesort")
        n_keep = (
            len(ordered) if fraction == 1 else int(np.ceil(len(ordered) * fraction))
        )
        selected.append(ordered.iloc[:n_keep])

    if not selected:
        return frame.copy().reset_index(drop=True)
    subset = pd.concat(selected, axis=0).sort_index(kind="mergesort")
    return subset.drop(columns=["_ein2_norm", "_stable_hash"]).reset_index(drop=True)


def load_human_split(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    split: str,
) -> pd.DataFrame:
    """Load a coded non-test human split from ``gold_to_code.csv``.

    Args:
        cfg: Validated task configuration. Included for API symmetry with other
            training data loaders.
        registry: Resolved path registry for the run.
        split: Human split name to load. ``"test"`` is deliberately blocked.

    Returns:
        DataFrame with ``EIN2``, ``text``, and integer ``human_label``.

    Raises:
        ValueError: If ``split`` is ``"test"``, required columns are missing,
            the requested split is empty, or labels are not coded as 0/1.
        FileNotFoundError: If ``gold_to_code.csv`` is absent.

    """
    del cfg
    if split == "test":
        raise ValueError("The human test split is only readable inside stage 07.")

    path = registry.gold_coding_template
    if not path.exists():
        raise FileNotFoundError(f"No human coding template at {path}.")

    df = pd.read_csv(path)
    required = {"EIN2", "split", "text", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}.")

    split_values = df["split"].astype(str).str.strip()
    sub = df.loc[split_values == split, ["EIN2", "text", "human_label"]].copy()
    if sub.empty:
        raise ValueError(f"No rows for split {split!r} in {path}.")
    if sub["human_label"].isna().any():
        raise ValueError(f"Split {split!r} contains uncoded human_label values.")

    labels = pd.to_numeric(sub["human_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"Split {split!r} human_label values must be 0/1.")

    sub["EIN2"] = _normalize_ein2_series(sub["EIN2"])
    sub["human_label"] = labels.astype(int)
    return sub.reset_index(drop=True)


def _read_silver_labels(path: Path) -> pd.DataFrame:
    """Read and validate the frozen silver-label artifact."""
    if not path.exists():
        raise FileNotFoundError(f"No silver labels at {path}. Run stage 04 first.")
    df = pd.read_csv(path)
    required = {"EIN2", "silver_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}.")
    return df


def _compute_p_pos(store_path: Path) -> pd.Series:
    """Compute positive-vote shares from the long annotation store."""
    store_df = AnnotationStore(store_path).to_frame()
    if store_df.empty:
        return pd.Series(dtype=float)

    store_df = store_df.copy()
    store_df["EIN2"] = _normalize_ein2_series(store_df["EIN2"])
    store_df["label"] = pd.to_numeric(store_df["label"], errors="coerce")
    votes = store_df[store_df["label"].isin([0, 1])].copy()
    if votes.empty:
        return pd.Series(dtype=float)

    grouped = votes.groupby("EIN2")["label"]
    numerator = grouped.sum()
    denominator = grouped.count()
    return (numerator / denominator).astype(float)


def _assert_no_manifest_overlap(
    frame: pd.DataFrame,
    manifest_path: Path,
    manifest_name: str,
) -> None:
    """Raise if ``frame`` overlaps a held-out manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No {manifest_name} manifest at {manifest_path}; cannot verify "
            "training leak guard.",
        )

    manifest = pd.read_csv(manifest_path)
    if "EIN2" not in manifest.columns:
        raise ValueError(f"{manifest_path} missing required EIN2 column.")

    manifest_ein2s = set(_normalize_ein2_series(manifest["EIN2"]).dropna())
    frame_ein2s = set(_normalize_ein2_series(frame["EIN2"]).dropna())
    overlap = frame_ein2s & manifest_ein2s
    if overlap:
        raise ValueError(
            f"Training frame overlaps {manifest_name} manifest: "
            f"{len(overlap)} EIN2(s).",
        )


def _normalize_ein2_series(series: pd.Series) -> pd.Series:
    """Normalize EIN2 values for robust CSV/set comparisons."""
    return series.astype("string").str.strip()


def _stable_hash(ein2: str, seed: int) -> int:
    """Return a deterministic hash integer for one EIN2 and seed."""
    payload = f"{seed}\0{ein2}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")
