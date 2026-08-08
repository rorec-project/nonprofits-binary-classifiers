"""N-gram diagnostic plots for silver-label artifacts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from binary_classifier.viz.style import (
    LIGHT_GREY,
    MUTED_GREY,
    OKABE_ITO_BLACK,
    OKABE_ITO_BLUE,
    OKABE_ITO_VERMILLION,
    pad_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_TEXT_COLUMNS = ("mission_text", "text")
_LABEL_COLUMNS = ("silver_label", "label", "hard_label")


def ngram_log_odds(
    silver_df_with_text: pd.DataFrame,
    ax: Axes,
    *,
    top_k: int = 30,
) -> None:
    """Plot signed religious-vs-nonreligious n-gram log odds.

    The input is an already-materialized stage artifact joined with text. The
    function detects a text column (``mission_text`` or ``text``) and a binary
    label column (``silver_label``, ``label``, or ``hard_label``), counts 1--2
    grams with ``min_df=5``, and plots the largest absolute signed log-odds.

    Args:
        silver_df_with_text: DataFrame containing mission text and binary silver
            labels where ``1`` is religious and ``0`` is nonreligious.
        ax: Matplotlib axes to draw into.
        top_k: Maximum number of n-grams to display.

    Raises:
        ValueError: If required columns are missing, labels are malformed, both
            classes are not present, or no n-grams survive vectorization.

    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    text_col = _detect_column(silver_df_with_text, _TEXT_COLUMNS, "text")
    label_col = _detect_column(silver_df_with_text, _LABEL_COLUMNS, "label")
    frame = silver_df_with_text[[text_col, label_col]].dropna().copy()
    if frame.empty:
        raise ValueError("No non-missing text/label rows to plot.")
    terms, log_odds, _, _ = compute_ngram_scores(
        frame[text_col].astype(str),
        frame[label_col],
        ngram_range=(1, 2),
        method="naive",
        min_df=5,
    )

    order = _top_absolute_order(log_odds, top_k)
    selected_terms = terms[order].astype(str)
    selected_odds = log_odds[order]
    colors = np.where(selected_odds >= 0.0, OKABE_ITO_BLUE, OKABE_ITO_VERMILLION)

    ax.barh(selected_terms, selected_odds, color=colors)
    ax.axvline(0.0, color=OKABE_ITO_BLACK, linewidth=0.8)
    ax.set_xlabel("Log odds: religious vs nonreligious")
    ax.set_ylabel("N-gram")
    ax.set_title(f"Top {len(selected_terms)} silver-label n-gram log odds")
    logger.info("Rendered n-gram log-odds plot with %d terms", len(selected_terms))


def ngram_weighted_log_odds(
    silver_df_with_text: pd.DataFrame,
    ax: Axes,
    *,
    ngram_range: tuple[int, int],
    top_k: int = 30,
) -> None:
    """Plot Monroe-style weighted n-gram log-odds z-scores."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    text_col = _detect_column(silver_df_with_text, _TEXT_COLUMNS, "text")
    label_col = _detect_column(silver_df_with_text, _LABEL_COLUMNS, "label")
    frame = silver_df_with_text[[text_col, label_col]].dropna().copy()
    terms, z_scores, _, _ = compute_ngram_scores(
        frame[text_col].astype(str),
        frame[label_col],
        ngram_range=ngram_range,
        method="weighted",
        min_df=5,
    )

    order = _top_absolute_order(z_scores, top_k)
    selected_terms = terms[order].astype(str)
    selected_scores = z_scores[order]
    colors = np.where(selected_scores >= 0.0, OKABE_ITO_BLUE, OKABE_ITO_VERMILLION)

    ax.barh(selected_terms, selected_scores, color=colors)
    ax.axvline(0.0, color=OKABE_ITO_BLACK, linewidth=0.8)
    ax.set_xlabel("Weighted log-odds z-score: religious vs nonreligious")
    ax.set_ylabel("N-gram")
    ax.set_title(
        f"Top {len(selected_terms)} weighted log-odds "
        f"{ngram_range[0]}-{ngram_range[1]}-grams",
    )
    logger.info(
        "Rendered weighted n-gram log-odds plot with %d terms",
        len(selected_terms),
    )


def compute_ngram_scores(
    texts: pd.Series | list[str],
    labels: pd.Series | list[int],
    *,
    ngram_range: tuple[int, int],
    method: str,
    min_df: int | float,
    stopwords: set[str] | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute signed n-gram scores and class-specific term counts.

    Positive scores indicate terms that are more distinctive of class 1
    (religious); negative scores indicate class 0 (non-religious).
    """
    if method not in {"naive", "weighted"}:
        raise ValueError("method must be 'naive' or 'weighted'.")

    frame = pd.DataFrame({"text": texts, "label": labels}).dropna()
    if frame.empty:
        raise ValueError("No non-missing text/label rows to score.")

    label_arr = _validate_binary_labels(frame["label"])
    if not ({0, 1} <= set(label_arr.tolist())):
        raise ValueError("Both religious (1) and nonreligious (0) rows are required.")

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=sorted(stopwords) if stopwords is not None else None,
    )
    counts = vectorizer.fit_transform(frame["text"].astype(str).tolist())
    terms = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    if len(terms) == 0:
        raise ValueError(f"No n-grams survived CountVectorizer(min_df={min_df}).")

    pos_counts = np.asarray(counts[label_arr == 1].sum(axis=0)).ravel().astype(float)
    neg_counts = np.asarray(counts[label_arr == 0].sum(axis=0)).ravel().astype(float)
    if method == "naive":
        scores = _naive_log_odds(pos_counts, neg_counts, vocabulary_size=len(terms))
    else:
        scores = _weighted_log_odds_z_scores(pos_counts, neg_counts)
    return terms, scores, pos_counts, neg_counts


def compute_keyness_frame(
    texts: pd.Series | list[str],
    *,
    ngram_range: tuple[int, int],
    min_df: int | float,
    labels: pd.Series | list[int] | None = None,
    probabilities: pd.Series | list[float] | None = None,
    stopwords: set[str] | list[str] | None = None,
) -> pd.DataFrame:
    """Return term-level keyness statistics for hard or probability labels.

    Hard-label comparisons use binary 0/1 labels. Probability-weighted
    comparisons use calibrated probabilities as expected positive membership
    weights and ``1 - p`` as expected negative membership weights.
    """
    if (labels is None) == (probabilities is None):
        raise ValueError("Provide exactly one of labels or probabilities.")

    frame = pd.DataFrame({"text": texts}).dropna().copy()
    if probabilities is not None:
        frame["probability"] = pd.Series(probabilities, index=frame.index)
        frame = frame.dropna(subset=["probability"])
        positive_weights = pd.to_numeric(frame["probability"], errors="coerce")
        if positive_weights.isna().any():
            raise ValueError("Probabilities must be numeric.")
        if not positive_weights.between(0.0, 1.0).all():
            raise ValueError("Probabilities must be in [0, 1].")
        pos_weight_arr = positive_weights.to_numpy(dtype=float)
        neg_weight_arr = 1.0 - pos_weight_arr
    else:
        frame["label"] = pd.Series(labels, index=frame.index)
        frame = frame.dropna(subset=["label"])
        label_arr = _validate_binary_labels(frame["label"])
        if not ({0, 1} <= set(label_arr.tolist())):
            raise ValueError(
                "Both religious (1) and nonreligious (0) rows are required."
            )
        pos_weight_arr = (label_arr == 1).astype(float)
        neg_weight_arr = (label_arr == 0).astype(float)

    if frame.empty:
        raise ValueError("No non-missing text rows to score.")

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=sorted(stopwords) if stopwords is not None else None,
    )
    counts = vectorizer.fit_transform(frame["text"].astype(str).tolist())
    terms = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    if len(terms) == 0:
        raise ValueError(f"No n-grams survived CountVectorizer(min_df={min_df}).")

    pos_counts = np.asarray(counts.T @ pos_weight_arr).ravel().astype(float)
    neg_counts = np.asarray(counts.T @ neg_weight_arr).ravel().astype(float)
    total_counts = pos_counts + neg_counts
    doc_frequency = np.asarray((counts > 0).sum(axis=0)).ravel().astype(float)
    z_scores = _weighted_log_odds_z_scores(pos_counts, neg_counts)

    pos_total = float(pos_counts.sum())
    neg_total = float(neg_counts.sum())
    vocabulary_size = float(len(terms))
    pos_rate = (pos_counts + 0.5) / (pos_total + 0.5 * vocabulary_size)
    neg_rate = (neg_counts + 0.5) / (neg_total + 0.5 * vocabulary_size)
    log2_rate_ratio = np.log2(pos_rate / neg_rate)

    return pd.DataFrame(
        {
            "term": terms.astype(str),
            "z_score": z_scores,
            "log2_rate_ratio": log2_rate_ratio,
            "positive_count": pos_counts,
            "negative_count": neg_counts,
            "total_count": total_counts,
            "document_frequency": doc_frequency,
            "positive_rate": pos_rate,
            "negative_rate": neg_rate,
        }
    ).sort_values("z_score", ascending=False, ignore_index=True)


def term_scatter_plot(
    keyness: pd.DataFrame,
    ax: Axes,
    *,
    title: str,
    top_k: int = 12,
) -> None:
    """Plot a static Scattertext-style term frequency comparison."""
    frame = _validated_keyness_frame(keyness)
    sizes = 8.0 + 42.0 * np.sqrt(frame["total_count"] / frame["total_count"].max())
    colors = np.where(frame["z_score"] >= 0.0, OKABE_ITO_BLUE, OKABE_ITO_VERMILLION)
    ax.scatter(
        np.log10(frame["negative_rate"]),
        np.log10(frame["positive_rate"]),
        s=sizes,
        c=colors,
        alpha=0.45,
        linewidths=0,
    )
    left = min(ax.get_xlim()[0], ax.get_ylim()[0])
    right = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([left, right], [left, right], color=LIGHT_GREY, linewidth=0.9)
    ax.set_xlim(left, right)
    ax.set_ylim(left, right)
    ax.set_xlabel("log10 rate in predicted non-religious corpus")
    ax.set_ylabel("log10 rate in predicted religious corpus")
    ax.set_title(title)
    _label_extreme_terms(frame, ax, "negative_rate", "positive_rate", top_k=top_k)


def keyness_volcano_plot(
    keyness: pd.DataFrame,
    ax: Axes,
    *,
    title: str,
    top_k: int = 12,
) -> None:
    """Plot distinctiveness against corpus frequency."""
    frame = _validated_keyness_frame(keyness)
    x = frame["z_score"]
    y = np.log10(frame["total_count"] + 1.0)
    colors = np.where(x >= 0.0, OKABE_ITO_BLUE, OKABE_ITO_VERMILLION)
    ax.scatter(x, y, s=18, c=colors, alpha=0.50, linewidths=0)
    ax.axvline(0.0, color=OKABE_ITO_BLACK, linewidth=0.8)
    for threshold in (-3.0, 3.0):
        ax.axvline(threshold, color=MUTED_GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Weighted log-odds z-score")
    ax.set_ylabel("log10(total term count + 1)")
    ax.set_title(title)
    _label_extreme_terms(frame, ax, "z_score", "total_count", top_k=top_k)
    pad_axes(ax, x=0.04, y=0.04)


def top_terms_lollipop_plot(
    keyness: pd.DataFrame,
    ax: Axes,
    *,
    title: str,
    top_k: int = 15,
) -> None:
    """Plot top positive and negative distinctive terms as a lollipop chart."""
    frame = _top_signed_terms(_validated_keyness_frame(keyness), top_k=top_k)
    colors = np.where(frame["z_score"] >= 0.0, OKABE_ITO_BLUE, OKABE_ITO_VERMILLION)
    y = np.arange(len(frame))
    ax.hlines(y, 0.0, frame["z_score"], color=colors, linewidth=1.1)
    ax.scatter(frame["z_score"], y, color=colors, s=18, zorder=3)
    ax.axvline(0.0, color=OKABE_ITO_BLACK, linewidth=0.8)
    ax.set_yticks(y, labels=frame["term"].to_list())
    ax.set_xlabel("Weighted log-odds z-score")
    ax.set_title(title)
    pad_axes(ax, x=0.05, y=0.02)


def keyness_sensitivity_heatmap(
    sensitivity: pd.DataFrame,
    ax: Axes,
    *,
    title: str,
) -> None:
    """Plot z-score stability for selected terms across label definitions."""
    if sensitivity.empty:
        raise ValueError("Sensitivity frame is empty.")
    required = {"term", "comparison", "z_score"}
    missing = required - set(sensitivity.columns)
    if missing:
        raise ValueError(f"Sensitivity frame is missing columns: {sorted(missing)}.")
    matrix = sensitivity.pivot(index="term", columns="comparison", values="z_score")
    matrix = matrix.fillna(0.0)
    max_abs = float(np.nanmax(np.abs(matrix.to_numpy())))
    limit = max(1.0, max_abs)
    image = ax.imshow(
        matrix.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
    )
    ax.set_xticks(
        np.arange(matrix.shape[1]),
        labels=matrix.columns.to_list(),
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index.to_list())
    ax.set_title(title)
    ax.set_xlabel("Population label definition")
    ax.set_ylabel("Term")
    ax.figure.colorbar(image, ax=ax, label="Weighted log-odds z-score")


def _validated_keyness_frame(keyness: pd.DataFrame) -> pd.DataFrame:
    required = {
        "term",
        "z_score",
        "positive_rate",
        "negative_rate",
        "total_count",
    }
    missing = required - set(keyness.columns)
    if missing:
        raise ValueError(f"Keyness frame is missing columns: {sorted(missing)}.")
    frame = keyness.dropna(subset=list(required)).copy()
    if frame.empty:
        raise ValueError("Keyness frame is empty after dropping missing values.")
    return frame


def _label_extreme_terms(
    frame: pd.DataFrame,
    ax: Axes,
    x_col: str,
    y_col: str,
    *,
    top_k: int,
) -> None:
    selected = _top_signed_terms(frame, top_k=max(1, top_k // 2))
    for _, row in selected.iterrows():
        x = np.log10(row[x_col]) if x_col.endswith("rate") else row[x_col]
        y = (
            np.log10(row[y_col])
            if y_col.endswith("rate")
            else np.log10(row[y_col] + 1.0)
        )
        ax.annotate(
            str(row["term"]),
            (x, y),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=6,
            alpha=0.90,
        )


def _top_signed_terms(frame: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    positive = frame.nlargest(top_k, "z_score")
    negative = frame.nsmallest(top_k, "z_score")
    selected = pd.concat([negative, positive], ignore_index=True)
    return selected.sort_values("z_score", ascending=True, ignore_index=True)


def _naive_log_odds(
    pos_counts: np.ndarray,
    neg_counts: np.ndarray,
    *,
    vocabulary_size: int,
) -> np.ndarray:
    pos_total = float(pos_counts.sum())
    neg_total = float(neg_counts.sum())
    vocabulary_size_float = float(vocabulary_size)
    return np.log((pos_counts + 1.0) / (pos_total + vocabulary_size_float)) - np.log(
        (neg_counts + 1.0) / (neg_total + vocabulary_size_float),
    )


def _weighted_log_odds_z_scores(
    pos_counts: np.ndarray,
    neg_counts: np.ndarray,
) -> np.ndarray:
    pooled_counts = pos_counts + neg_counts
    prior_total = float(len(pooled_counts))
    alpha = pooled_counts * (prior_total / float(pooled_counts.sum()))
    alpha_total = float(alpha.sum())
    pos_total = float(pos_counts.sum())
    neg_total = float(neg_counts.sum())
    pos_odds = np.log(
        (pos_counts + alpha) / (pos_total + alpha_total - pos_counts - alpha)
    )
    neg_odds = np.log(
        (neg_counts + alpha) / (neg_total + alpha_total - neg_counts - alpha)
    )
    variance = (1.0 / (pos_counts + alpha)) + (1.0 / (neg_counts + alpha))
    return (pos_odds - neg_odds) / np.sqrt(variance)


def _top_absolute_order(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    order = np.argsort(np.abs(scores))[-top_k:]
    return order[np.argsort(scores[order])]


def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find {role} column; tried {list(candidates)}.")


def _validate_binary_labels(values: pd.Series) -> np.ndarray:
    labels = pd.to_numeric(values, errors="coerce")
    if labels.isna().any():
        raise ValueError("Labels must be numeric 0/1 values.")
    label_arr = labels.astype(int).to_numpy()
    if not np.array_equal(label_arr, labels.to_numpy(dtype=float)):
        raise ValueError("Labels must be integer 0/1 values.")
    if not np.isin(label_arr, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1 values.")
    return label_arr
