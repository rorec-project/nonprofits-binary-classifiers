"""Label aggregation and denoising.

Provides majority-vote aggregation as the default silver-label builder. The
Dawid-Skene and CROWDLAB dispatch arms are intentionally quarantined until they
are validated for this pipeline: Dawid-Skene (Dawid & Skene 1979) is not yet
verified for correlated LLM ensembles, and CROWDLAB needs classifier
``pred_probs`` from a later fine-tuning stage (Goh, Mueller et al. 2022;
cleanlab multiannotator docs).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

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
    """Quarantined Dawid-Skene comparison arm.

    Dawid-Skene estimates worker reliability by EM (Dawid & Skene 1979), but
    this pipeline's model-by-prompt annotators are correlated LLM ensemble
    members rather than independent human coders. Majority vote remains the
    default until this arm is validated for that dependence structure.

    Args:
        df: Long/tidy dataframe with columns ``EIN2``, ``source_id``,
            ``label`` (0/1 numeric, NaN = abstain).
        n_iter: Reserved for the future EM implementation.
        tol: Reserved for the future EM implementation.

    Returns:
        This function currently raises instead of returning labels.

    """
    raise NotImplementedError(
        "Dawid-Skene is quarantined: unverified for correlated LLM ensembles; "
        "majority vote is the default.",
    )


def aggregate_crowdlab(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Quarantined CROWDLAB comparison arm.

    CROWDLAB combines multiple annotators with classifier ``pred_probs`` to
    estimate consensus and label quality (Goh, Mueller et al. 2022; cleanlab
    multiannotator docs). This pipeline has not reached the fine-tuning stage
    that will produce those probabilities, so selecting this method must fail
    explicitly rather than silently returning an empty result.

    Args:
        df: Long/tidy dataframe with columns ``EIN2``, ``source_id``,
            ``label`` (0/1 numeric, NaN = abstain).

    Returns:
        This function currently raises instead of returning labels.

    """
    raise NotImplementedError(
        "CROWDLAB is quarantined: requires pred_probs from a trained "
        "classifier (fine-tuning stage), not yet available.",
    )


# ── Unified aggregator ─────────────────────────────────────────────────────


def aggregate_labels(
    df: pd.DataFrame,
    method: str = "majority",
) -> pd.DataFrame:
    """Dispatch to the requested aggregation method.

    The dormant Dawid-Skene and CROWDLAB arms remain in this dispatch table so
    experimental selection raises an explicit quarantine error instead of
    silently producing empty labels. Majority vote is the production default.

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
