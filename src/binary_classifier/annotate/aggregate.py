"""Label aggregation and denoising.

Provides majority-vote aggregation as the default silver-label builder, plus
drop-in Dawid-Skene and CROWDLAB comparison arms. Dawid-Skene (Dawid & Skene
1979) is useful for diagnostics but remains a comparison method because this
pipeline's model-by-prompt annotators are correlated LLM ensemble members.
CROWDLAB requires classifier ``pred_probs`` from a later fine-tuning stage (Goh,
Mueller et al. 2022; cleanlab multiannotator docs), so it still fails
explicitly when those probabilities are not supplied.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SILVER_COLUMNS = [
    "EIN2",
    "silver_label",
    "silver_confidence",
    "num_votes",
    "num_abstain",
    "agreement",
    "tie",
]

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


def _label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long annotation store to an EIN2 × source_id label matrix.

    Args:
        df: Long/tidy dataframe with columns ``EIN2``, ``source_id``, and
            ``label``.

    Returns:
        Dataframe indexed by ``EIN2`` with one column per ``source_id``. Missing
        labels and missing annotator-task pairs are represented as ``NaN``.

    """
    labels = df[["EIN2", "source_id", "label"]].drop_duplicates(
        subset=["EIN2", "source_id"],
        keep="last",
    )
    return labels.pivot(index="EIN2", columns="source_id", values="label").sort_index()


def _agreement_with_consensus(
    labels: pd.DataFrame,
    silver_labels: pd.Series,
) -> pd.Series:
    """Compute observed-vote agreement with a consensus label.

    Args:
        labels: EIN2 × source_id label matrix.
        silver_labels: Consensus labels indexed by ``EIN2``.

    Returns:
        Per-EIN2 share of non-missing votes equal to the consensus label, or
        ``NaN`` when no consensus/non-missing votes exist.

    """
    agreements = []
    for ein2, row in labels.iterrows():
        consensus = silver_labels.loc[ein2]
        observed = row.dropna()
        if pd.isna(consensus) or observed.empty:
            agreements.append(np.nan)
        else:
            agreements.append((observed == consensus).sum() / len(observed))
    return pd.Series(agreements, index=labels.index, dtype="float64")


def _wide_result(
    labels: pd.DataFrame,
    silver_labels: pd.Series,
    silver_confidence: pd.Series,
) -> pd.DataFrame:
    """Map consensus labels and qualities to the majority-vote schema.

    Args:
        labels: EIN2 × source_id label matrix.
        silver_labels: Consensus labels indexed by ``EIN2``.
        silver_confidence: Consensus-quality scores indexed by ``EIN2``.

    Returns:
        Wide silver-label dataframe with the exact majority-vote columns.

    """
    silver_labels = silver_labels.reindex(labels.index)
    silver_confidence = silver_confidence.reindex(labels.index)
    result = pd.DataFrame(
        {
            "EIN2": labels.index.to_list(),
            "silver_label": silver_labels.to_numpy(),
            "silver_confidence": silver_confidence.to_numpy(),
            "num_votes": labels.notna().sum(axis=1).astype(int).to_numpy(),
            "num_abstain": labels.isna().sum(axis=1).astype(int).to_numpy(),
            "agreement": _agreement_with_consensus(labels, silver_labels).to_numpy(),
            "tie": False,
        },
    )
    return result[SILVER_COLUMNS]


def aggregate_dawid_skene(
    df: pd.DataFrame,
    n_iter: int = 100,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """Aggregate labels with the Dawid-Skene comparison arm.

    Dawid-Skene estimates worker reliability by EM (Dawid & Skene 1979), but
    this pipeline's model-by-prompt annotators are correlated LLM ensemble
    members rather than independent human coders. Majority vote remains the
    default until this arm is validated for that dependence structure.

    Args:
        df: Long/tidy dataframe with columns ``EIN2``, ``source_id``,
            ``label`` (0/1 numeric, NaN = abstain).
        n_iter: Maximum EM iterations.
        tol: EM convergence tolerance.

    Returns:
        Wide dataframe with the exact majority-vote silver-label columns.

    """
    labels = _label_matrix(df)
    if labels.shape[0] < 2 or labels.shape[1] < 2:
        raise NotImplementedError(
            "Dawid-Skene is quarantined: unverified for correlated LLM ensembles; "
            "majority vote is the default.",
        )

    from crowdkit.aggregation.classification import DawidSkene

    long_labels = (
        labels.stack()
        .dropna()
        .rename("label")
        .reset_index()
        .rename(columns={"EIN2": "task", "source_id": "worker"})
    )
    if long_labels.empty:
        empty = pd.Series(np.nan, index=labels.index, dtype="float64")
        return _wide_result(labels, empty, empty)

    long_labels["label"] = long_labels["label"].astype(int)
    probabilities = DawidSkene(n_iter=n_iter, tol=tol).fit_predict_proba(long_labels)
    silver_labels = pd.Series(np.nan, index=labels.index, dtype="float64")
    silver_confidence = pd.Series(np.nan, index=labels.index, dtype="float64")
    if not probabilities.empty:
        silver_labels.loc[probabilities.index] = probabilities.idxmax(axis=1)
        silver_confidence.loc[probabilities.index] = probabilities.max(axis=1)
    return _wide_result(labels, silver_labels, silver_confidence)


def aggregate_crowdlab(
    df: pd.DataFrame,
    pred_probs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate labels with the CROWDLAB comparison arm.

    CROWDLAB combines multiple annotators with classifier ``pred_probs`` to
    estimate consensus and label quality (Goh, Mueller et al. 2022; cleanlab
    multiannotator docs). Selecting this method without probabilities fails
    explicitly rather than silently returning an empty result.

    Args:
        df: Long/tidy dataframe with columns ``EIN2``, ``source_id``,
            ``label`` (0/1 numeric, NaN = abstain).
        pred_probs: Classifier probabilities with columns ``p0`` and ``p1``,
            indexed by ``EIN2`` or carrying an ``EIN2`` column.

    Returns:
        Wide dataframe with the exact majority-vote silver-label columns.

    """
    if pred_probs is None:
        raise NotImplementedError(
            "CROWDLAB is quarantined: requires pred_probs from a trained "
            "classifier (fine-tuning stage), not yet available.",
        )

    if not {"p0", "p1"}.issubset(pred_probs.columns):
        raise ValueError("pred_probs must include p0 and p1 columns")

    labels = _label_matrix(df)
    probs = pred_probs.set_index("EIN2") if "EIN2" in pred_probs.columns else pred_probs
    probs = probs[["p0", "p1"]]
    probs = probs[~probs.index.duplicated(keep="last")]

    missing_predictions = labels.index.difference(probs.index)
    if len(missing_predictions) > 0:
        logger.warning(
            "Dropping %d EIN2 rows missing classifier predictions for CROWDLAB.",
            len(missing_predictions),
        )
    labels = labels.loc[labels.index.intersection(probs.index)]
    probs = probs.loc[labels.index]
    if labels.empty:
        raise ValueError("No overlapping EIN2 rows between labels and pred_probs")

    from cleanlab.multiannotator import get_label_quality_multiannotator

    quality = get_label_quality_multiannotator(
        labels,
        probs.to_numpy(),
        quality_method="crowdlab",
        verbose=False,
    )
    label_quality = quality["label_quality"].set_index(labels.index)
    silver_labels = label_quality["consensus_label"]
    silver_confidence = label_quality["consensus_quality_score"]
    return _wide_result(labels, silver_labels, silver_confidence)


# ── Unified aggregator ─────────────────────────────────────────────────────


def aggregate_labels(
    df: pd.DataFrame,
    method: str = "majority",
    pred_probs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dispatch to the requested aggregation method.

    Majority vote is the production default. Dawid-Skene and CROWDLAB are
    drop-in comparison arms; CROWDLAB requires classifier probabilities.

    Args:
        df: Long/tidy label dataframe.
        method: One of ``majority``, ``dawid_skene``, ``crowdlab``.
        pred_probs: Optional class probabilities for CROWDLAB.

    Returns:
        Wide dataframe with one row per EIN2.

    """
    if method == "majority":
        return majority_vote(df)
    if method == "dawid_skene":
        return aggregate_dawid_skene(df)
    if method == "crowdlab":
        return aggregate_crowdlab(df, pred_probs=pred_probs)
    raise ValueError(f"Unknown aggregation method: {method}")
