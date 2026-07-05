"""Class-conditional wordcloud diagnostics for silver-label text."""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from PIL import Image, ImageFont
from wordcloud import WordCloud

from binary_classifier.viz.ngrams import (
    _detect_column,
    _validate_binary_labels,
    compute_ngram_scores,
)
from binary_classifier.viz.style import OKABE_ITO_BLUE, OKABE_ITO_VERMILLION

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_TEXT_COLUMNS = ("mission_text", "text")
_LABEL_COLUMNS = ("silver_label", "label", "hard_label")
_VALID_WEIGHTINGS = {"frequency", "distinctive"}
_RGB_RE = re.compile(
    r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)"
)
_SVG_NS = "http://www.w3.org/2000/svg"

CURATED_FREQUENCY_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "association",
    "charitable",
    "charity",
    "corporation",
    "foundation",
    "fund",
    "inc",
    "incorporated",
    "mission",
    "nonprofit",
    "organisation",
    "organization",
    "organizations",
    "program",
    "programs",
    "provide",
    "provides",
    "providing",
    "public",
    "service",
    "services",
    "support",
    "supports",
    "tax",
}


def class_wordclouds(
    silver_df_with_text: pd.DataFrame,
    ax: Axes,
    *,
    ngram_range: tuple[int, int],
    weighting: str,
    max_words: int = 150,
    min_df: int | float = 1,
    stopwords: set[str] | list[str] | None = None,
) -> None:
    """Render religious and non-religious wordcloud panels.

    Args:
        silver_df_with_text: DataFrame containing mission text and binary silver
            labels where ``1`` is religious and ``0`` is non-religious.
        ax: Single axes supplied by the stage-10 save wrapper. The function
            replaces it with two side-by-side panel axes.
        ngram_range: CountVectorizer n-gram range.
        weighting: ``"frequency"`` for per-class counts or ``"distinctive"``
            for positive-side weighted log-odds z-scores.
        max_words: Maximum number of words in each cloud.
        min_df: Minimum document frequency passed to ``CountVectorizer``.
        stopwords: Optional extra stopwords. Frequency clouds combine these with
            the curated generic/nonprofit list; distinctive clouds use them only
            when explicitly supplied.
    """
    if weighting not in _VALID_WEIGHTINGS:
        raise ValueError("weighting must be 'frequency' or 'distinctive'.")
    if max_words <= 0:
        raise ValueError("max_words must be positive.")

    text_col = _detect_column(silver_df_with_text, _TEXT_COLUMNS, "text")
    label_col = _detect_column(silver_df_with_text, _LABEL_COLUMNS, "label")
    frame = silver_df_with_text[[text_col, label_col]].dropna().copy()
    if frame.empty:
        raise ValueError("No non-missing text/label rows to plot.")
    labels = _validate_binary_labels(frame[label_col])
    if not ({0, 1} <= set(labels.tolist())):
        raise ValueError("Both religious (1) and non-religious (0) rows are required.")

    class_frequencies = _class_frequencies(
        frame[text_col].astype(str),
        labels,
        ngram_range=ngram_range,
        weighting=weighting,
        min_df=min_df,
        stopwords=stopwords,
    )

    fig = ax.figure
    ax.remove()
    axes = fig.subplots(1, 2, squeeze=False).ravel()
    panels = (
        (1, OKABE_ITO_BLUE),
        (0, OKABE_ITO_VERMILLION),
    )
    for panel_ax, (label_value, color) in zip(axes, panels, strict=True):
        frequencies = class_frequencies[label_value]
        if not frequencies:
            raise ValueError(f"No n-grams available for class {label_value} wordcloud.")
        cloud = _transparent_wordcloud(
            frequencies,
            max_words=max_words,
            color_func=_color_ramp(color),
        )
        panel_ax.imshow(cloud, interpolation="bilinear")
        panel_ax.axis("off")

    logger.info(
        "Rendered %s wordclouds for %d-%d-grams",
        weighting,
        ngram_range[0],
        ngram_range[1],
    )


def class_wordcloud(
    silver_df_with_text: pd.DataFrame,
    ax: Axes,
    *,
    ngram_range: tuple[int, int],
    weighting: str,
    class_label: int,
    max_words: int = 150,
    min_df: int | float = 1,
    stopwords: set[str] | list[str] | None = None,
) -> WordCloud:
    """Render one class-conditional wordcloud without in-figure text."""
    cloud = build_class_wordcloud(
        silver_df_with_text,
        ngram_range=ngram_range,
        weighting=weighting,
        class_label=class_label,
        max_words=max_words,
        min_df=min_df,
        stopwords=stopwords,
    )
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    return cloud


def build_class_wordcloud(
    silver_df_with_text: pd.DataFrame,
    *,
    ngram_range: tuple[int, int],
    weighting: str,
    class_label: int,
    max_words: int = 150,
    min_df: int | float = 1,
    stopwords: set[str] | list[str] | None = None,
) -> WordCloud:
    """Build one class-conditional wordcloud for raster or vector output."""
    if class_label not in {0, 1}:
        raise ValueError("class_label must be 0 or 1.")
    if weighting not in _VALID_WEIGHTINGS:
        raise ValueError("weighting must be 'frequency' or 'distinctive'.")
    if max_words <= 0:
        raise ValueError("max_words must be positive.")

    text_col = _detect_column(silver_df_with_text, _TEXT_COLUMNS, "text")
    label_col = _detect_column(silver_df_with_text, _LABEL_COLUMNS, "label")
    frame = silver_df_with_text[[text_col, label_col]].dropna().copy()
    if frame.empty:
        raise ValueError("No non-missing text/label rows to plot.")
    labels = _validate_binary_labels(frame[label_col])
    if not ({0, 1} <= set(labels.tolist())):
        raise ValueError("Both religious (1) and non-religious (0) rows are required.")

    class_frequencies = _class_frequencies(
        frame[text_col].astype(str),
        labels,
        ngram_range=ngram_range,
        weighting=weighting,
        min_df=min_df,
        stopwords=stopwords,
    )
    frequencies = class_frequencies[class_label]
    if not frequencies:
        raise ValueError(f"No n-grams available for class {class_label} wordcloud.")

    color = OKABE_ITO_BLUE if class_label == 1 else OKABE_ITO_VERMILLION
    cloud = _transparent_wordcloud(
        frequencies,
        max_words=max_words,
        color_func=_color_ramp(color),
    )
    logger.info(
        "Rendered %s class-%d wordcloud for %d-%d-grams",
        weighting,
        class_label,
        ngram_range[0],
        ngram_range[1],
    )
    return cloud


def write_wordcloud_pdf(cloud: WordCloud, path: Path) -> None:
    """Write a selectable PDF from the WordCloud layout."""
    with _quiet_fonttools(), plt.rc_context({"pdf.fonttype": 42}):
        fig, _ = _render_layout_axes(cloud)
        try:
            fig.savefig(path, transparent=True)
        finally:
            plt.close(fig)


def write_wordcloud_svg(cloud: WordCloud, path: Path, *, embed_font: bool = True) -> None:
    """Write a renderer-stable SVG with visible paths and invisible text.

    Args:
        cloud: Generated WordCloud instance (must have ``layout_`` populated).
        path: Output SVG path.
        embed_font: Deprecated compatibility argument; SVG output now uses
            visible paths plus an invisible selectable text layer.
    """
    del embed_font
    path_svg = _render_layout_svg(cloud, {"svg.fonttype": "path"})
    text_svg = _render_layout_svg(cloud, {"svg.fonttype": "none"})

    ET.register_namespace("", _SVG_NS)
    path_root = ET.fromstring(path_svg)
    text_root = ET.fromstring(text_svg)
    _set_svg_canvas(path_root, cloud.width, cloud.height)

    invisible_group = ET.Element(
        f"{{{_SVG_NS}}}g",
        {"style": "fill-opacity:0;stroke-opacity:0"},
    )
    for child in list(text_root):
        if _contains_svg_text(child):
            invisible_group.append(child)
    path_root.append(invisible_group)

    path.write_text(ET.tostring(path_root, encoding="unicode"), encoding="utf-8")


def _render_layout_svg(cloud: WordCloud, rc: dict[str, str]) -> str:
    with _quiet_fonttools(), plt.rc_context(rc):
        fig, _ = _render_layout_axes(cloud)
        buffer = io.StringIO()
        try:
            fig.savefig(buffer, format="svg", transparent=True)
        finally:
            plt.close(fig)
    return buffer.getvalue()


@contextmanager
def _quiet_fonttools():
    fonttools_logger = logging.getLogger("fontTools")
    previous_level = fonttools_logger.level
    fonttools_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        fonttools_logger.setLevel(previous_level)


def _layout_glyphs(
    cloud: WordCloud,
) -> list[tuple[str, float, float, int, tuple[float, float, float, float]]]:
    """Return glyph baselines using WordCloud's PIL metric math."""
    cloud._check_generated()
    glyphs = []
    for (word, _count), font_size, (y, x), orientation, color in cloud.layout_:
        x *= cloud.scale
        y *= cloud.scale

        font = ImageFont.truetype(cloud.font_path, int(font_size * cloud.scale))
        (size_x, _size_y), (offset_x, offset_y) = font.font.getsize(
            word,
            "",
            None,
            None,
            None,
            None,
        )
        ascent, _descent = font.getmetrics()

        min_x = -offset_x
        max_x = size_x - offset_x
        max_y = ascent - offset_y

        if orientation == Image.Transpose.ROTATE_90:
            baseline_x = x + max_y
            baseline_y = y + max_x - min_x
            rotation_deg = 90
        else:
            baseline_x = x + min_x
            baseline_y = y + max_y
            rotation_deg = 0

        glyphs.append(
            (
                word,
                baseline_x,
                baseline_y,
                rotation_deg,
                _parse_rgba(color),
            )
        )
    return glyphs


def _render_layout_axes(cloud: WordCloud, dpi: int = 100):
    """Render WordCloud's packed layout into pixel-sized matplotlib axes."""
    fig = plt.figure(
        figsize=(cloud.width / dpi, cloud.height / dpi),
        dpi=dpi,
    )
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.patch.set_alpha(0)
    ax.set_xlim(0, cloud.width)
    ax.set_ylim(cloud.height, 0)
    ax.axis("off")

    font_properties = font_manager.FontProperties(fname=cloud.font_path)
    for glyph, layout_item in zip(_layout_glyphs(cloud), cloud.layout_, strict=True):
        word, x, y, rotation, rgba = glyph
        _layout_word, font_size, _position, _orientation, _color = layout_item
        ax.text(
            x,
            y,
            word,
            fontsize=font_size * cloud.scale * 72 / dpi,
            color=rgba,
            fontproperties=font_properties,
            ha="left",
            va="baseline",
            rotation=rotation,
            rotation_mode="anchor",
        )
    return fig, ax


def _parse_rgba(color: str) -> tuple[float, float, float, float]:
    match = _RGB_RE.fullmatch(color.strip())
    if match is None:
        red, green, blue = to_rgb(color)
        return red, green, blue, 1.0

    red, green, blue, alpha = match.groups()
    opacity = 1.0 if alpha is None else float(alpha)
    if opacity > 1.0:
        opacity /= 255.0
    return int(red) / 255.0, int(green) / 255.0, int(blue) / 255.0, opacity


def _contains_svg_text(element: ET.Element) -> bool:
    if element.tag == f"{{{_SVG_NS}}}text":
        return True
    return element.find(f".//{{{_SVG_NS}}}text") is not None


def _set_svg_canvas(root: ET.Element, width: int, height: int) -> None:
    root.set("width", str(width))
    root.set("height", str(height))
    if "viewBox" not in root.attrib:
        root.set("viewBox", f"0 0 {width} {height}")


def _class_frequencies(
    texts: pd.Series,
    labels: np.ndarray,
    *,
    ngram_range: tuple[int, int],
    weighting: str,
    min_df: int | float,
    stopwords: set[str] | list[str] | None,
) -> dict[int, dict[str, float]]:
    if weighting == "frequency":
        terms, _, pos_counts, neg_counts = compute_ngram_scores(
            texts,
            labels.tolist(),
            ngram_range=ngram_range,
            method="naive",
            min_df=min_df,
            stopwords=_frequency_stopwords(stopwords),
        )
        return {
            1: _positive_weights(terms, pos_counts),
            0: _positive_weights(terms, neg_counts),
        }

    terms, scores, _, _ = compute_ngram_scores(
        texts,
        labels.tolist(),
        ngram_range=ngram_range,
        method="weighted",
        min_df=min_df,
        stopwords=set(stopwords) if stopwords is not None else None,
    )
    return {
        1: _positive_weights(terms, scores),
        0: _positive_weights(terms, -scores),
    }


def _frequency_stopwords(stopwords: set[str] | list[str] | None) -> set[str]:
    combined = set(CURATED_FREQUENCY_STOPWORDS)
    if stopwords is not None:
        combined.update(str(word).lower() for word in stopwords)
    return combined


def _positive_weights(terms: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        str(term): float(weight)
        for term, weight in zip(terms, weights, strict=True)
        if np.isfinite(weight) and weight > 0.0
    }


def _transparent_wordcloud(
    frequencies: dict[str, float],
    *,
    max_words: int,
    color_func: Callable[..., str],
) -> WordCloud:
    cloud = _make_wordcloud(None, max_words=max_words, color_func=color_func)
    cloud.generate_from_frequencies(frequencies)
    if _has_transparent_background(cloud):
        return cloud

    cloud = _make_wordcloud(
        "rgba(255,255,255,0)",
        max_words=max_words,
        color_func=color_func,
    )
    cloud.generate_from_frequencies(frequencies)
    if not _has_transparent_background(cloud):
        raise RuntimeError("WordCloud rendered an opaque background after RGBA fallback.")
    return cloud


def _make_wordcloud(
    background_color: str | None,
    *,
    max_words: int,
    color_func: Callable[..., str],
) -> WordCloud:
    return WordCloud(
        width=1200,
        height=700,
        max_words=max_words,
        mode="RGBA",
        background_color=background_color,
        font_path=font_manager.findfont("DejaVu Sans"),
        color_func=color_func,
        random_state=13,
        collocations=False,
        prefer_horizontal=0.95,
        margin=8,
    )


def _has_transparent_background(cloud: WordCloud) -> bool:
    image = np.asarray(cloud.to_image())
    return image.ndim == 3 and image.shape[2] == 4 and bool(np.min(image[:, :, 3]) < 255)


def _color_ramp(base_color: str) -> Callable[..., str]:
    red, green, blue = (int(channel * 255) for channel in to_rgb(base_color))

    def color_func(
        word: str,
        font_size: int,
        position: tuple[int, int],
        orientation: object,
        random_state: np.random.RandomState | None = None,
        **kwargs: object,
    ) -> str:
        del word, font_size, position, orientation, kwargs
        rng = random_state if random_state is not None else np.random.RandomState(13)
        blend = float(rng.uniform(0.55, 1.0))
        ramped = (
            int(255 - ((255 - red) * blend)),
            int(255 - ((255 - green) * blend)),
            int(255 - ((255 - blue) * blend)),
        )
        return f"rgb({ramped[0]}, {ramped[1]}, {ramped[2]})"

    return color_func


__all__ = [
    "CURATED_FREQUENCY_STOPWORDS",
    "build_class_wordcloud",
    "class_wordcloud",
    "class_wordclouds",
    "write_wordcloud_pdf",
    "write_wordcloud_svg",
]
