"""Label aggregation and denoising.

Provides majority-vote aggregation as the default silver-label builder, plus
drop-in hooks for ``crowd-kit`` (Dawid-Skene) and ``cleanlab`` (CROWDLAB) as
comparison arms. The long/tidy store is the natural input for all three
methods.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Majority vote ────────────────────────────────────────────────────────────


def majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-EIN2 labels by majority vote.

    Args:
        df: Long/tidy dataframe with columns ``EIN2``, ``label``,
            ``confidence``, ``source_id``.

    Returns:
        Wide dataframe with one row per EIN2 and columns:
        ``silver_label``, ``silver_confidence``, ``num_votes``,
        ``num_abstain``, ``agreement``, ``tie``.

    """
    results = []
    for ein2, group in df.groupby("EIN2"):
        labels = group["label"].dropna()
        abstains = group["label"].isna().sum()

        if len(labels) == 0:
            # All abstain → route to human review
            results.append(
                {
                    "EIN2": ein2,
                    "silver_label": np.nan,
                    "silver_confidence": np.nan,
                    "num_votes": 0,
                    "num_abstain": abstains,
                    "agreement": np.nan,
                    "tie": False,
                },
            )
            continue

        counts = labels.value_counts()
        top_label = counts.index[0]
        top_count = counts.iloc[0]
        agreement = top_count / len(labels)

        # Tie detection: two or more classes with equal max count
        tie = (counts == top_count).sum() > 1
        silver_label = np.nan if tie else top_label

        # Confidence = mean confidence among votes for the silver label
        conf_for_label = group[group["label"] == top_label]["confidence"]
        silver_confidence = (
            conf_for_label.mean() if not conf_for_label.empty else np.nan
        )

        results.append(
            {
                "EIN2": ein2,
                "silver_label": silver_label,
                "silver_confidence": silver_confidence,
                "num_votes": len(labels),
                "num_abstain": abstains,
                "agreement": agreement,
                "tie": tie,
            },
        )

    return pd.DataFrame(results)


# ── Drop-in comparison arms ──────────────────────────────────────────────────


def aggregate_dawid_skene(
    df: pd.DataFrame,
    n_iter: int = 100,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """Dawid-Skene label aggregation via ``crowd-kit``.

    Args:
        df: Long/tidy dataframe with columns ``EIN2``, ``source_id``,
            ``label`` (0/1 numeric, NaN = abstain).
        n_iter: Maximum EM iterations.
        tol: Convergence tolerance.

    Returns:
        Wide dataframe with one row per EIN2 and columns:
        ``silver_label``, ``ds_prob``, ``ds_reliability``.

    """
    try:
        from crowdkit.aggregation import DawidSkene
    except ImportError:
        logger.exception("crowd-kit not installed; skipping Dawid-Skene")
        return pd.DataFrame()

    # Drop abstains
    clean = df.dropna(subset=["label"]).copy()
    clean["label"] = clean["label"].astype(int)

    ds = DawidSkene(n_iter=n_iter, tol=tol)
    aggregated = ds.fit_predict(clean)
    reliabilities = ds.fit_predict_proba(clean)

    out = aggregated.reset_index()
    out.columns = ["EIN2", "silver_label"]
    out["ds_prob"] = reliabilities.max(axis=1).values
    return out


def aggregate_crowdlab(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """CROWDLAB consensus + noise detection via ``cleanlab``.

    .. note::
        This is a **stub**. CROWDLAB requires ``pred_probs`` (model
        predicted-class probabilities) which are not yet wired into the
        pipeline. Call this only after a baseline model produces
        per-EIN2 probabilities.

    Args:
        df: Long/tidy dataframe with columns ``EIN2``, ``source_id``,
            ``label`` (0/1 numeric, NaN = abstain).

    Returns:
        Wide dataframe with one row per EIN2 and columns:
        ``silver_label``, ``crowdlab_consensus``, ``quality_score``.

    """
    logger.warning(
        "aggregate_crowdlab is a stub — pred_probs are not yet wired. "
        "Returning empty DataFrame."
    )
    return pd.DataFrame()


# ── Unified aggregator ─────────────────────────────────────────────────────


def aggregate_labels(
    df: pd.DataFrame,
    method: str = "majority",
) -> pd.DataFrame:
    """Dispatch to the requested aggregation method.

    Args:
        df: Long/tidy label dataframe.
        method: One of ``majority``, ``dawid_skene``, ``crowdlab``.

    Returns:
        Wide dataframe with one row per EIN2.

    """
    dispatch: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "majority": majority_vote,
        "dawid_skene": aggregate_dawid_skene,
        "crowdlab": aggregate_crowdlab,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown aggregation method: {method}")
    return dispatch[method](df)
