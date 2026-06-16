"""N-gram diagnostic plots for silver-label artifacts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_TEXT_COLUMNS = ("mission_text", "text")
_LABEL_COLUMNS = ("silver_label", "label", "hard_label")


def ngram_log_odds(
    silver_df_with_text: pd.DataFrame,
    ax: "Axes",
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

    labels = _validate_binary_labels(frame[label_col])
    texts = frame[text_col].astype(str).tolist()
    if not ({0, 1} <= set(labels.tolist())):
        raise ValueError("Both religious (1) and nonreligious (0) rows are required.")

    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5)
    counts = vectorizer.fit_transform(texts)
    terms = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    if len(terms) == 0:
        raise ValueError("No n-grams survived CountVectorizer(min_df=5).")

    pos_counts = np.asarray(counts[labels == 1].sum(axis=0)).ravel().astype(float)
    neg_counts = np.asarray(counts[labels == 0].sum(axis=0)).ravel().astype(float)
    vocabulary_size = float(len(terms))
    pos_total = float(pos_counts.sum())
    neg_total = float(neg_counts.sum())
    log_odds = np.log((pos_counts + 1.0) / (pos_total + vocabulary_size)) - np.log(
        (neg_counts + 1.0) / (neg_total + vocabulary_size)
    )

    order = np.argsort(np.abs(log_odds))[-top_k:]
    order = order[np.argsort(log_odds[order])]
    selected_terms = terms[order].astype(str)
    selected_odds = log_odds[order]
    colors = np.where(selected_odds >= 0.0, "tab:blue", "tab:orange")

    ax.barh(selected_terms, selected_odds, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Log odds: religious vs nonreligious")
    ax.set_ylabel("N-gram")
    ax.set_title(f"Top {len(selected_terms)} silver-label n-gram log odds")
    logger.info("Rendered n-gram log-odds plot with %d terms", len(selected_terms))


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
