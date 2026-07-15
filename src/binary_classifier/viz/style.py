"""Shared publication-style settings for visualization outputs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
from cycler import cycler

if TYPE_CHECKING:
    from matplotlib.axes import Axes

COLUMN_WIDTH = 3.5
PAGE_WIDTH = 7.0

OKABE_ITO_ORANGE = "#E69F00"
OKABE_ITO_SKY_BLUE = "#56B4E9"
OKABE_ITO_BLUISH_GREEN = "#009E73"
OKABE_ITO_YELLOW = "#F0E442"
OKABE_ITO_BLUE = "#0072B2"
OKABE_ITO_VERMILLION = "#D55E00"
OKABE_ITO_REDDISH_PURPLE = "#CC79A7"
OKABE_ITO_BLACK = "#000000"
MUTED_GREY = "#999999"
LIGHT_GREY = "#D9D9D9"

OKABE_ITO_PALETTE = {
    "orange": OKABE_ITO_ORANGE,
    "sky_blue": OKABE_ITO_SKY_BLUE,
    "bluish_green": OKABE_ITO_BLUISH_GREEN,
    "yellow": OKABE_ITO_YELLOW,
    "blue": OKABE_ITO_BLUE,
    "vermillion": OKABE_ITO_VERMILLION,
    "reddish_purple": OKABE_ITO_REDDISH_PURPLE,
    "black": OKABE_ITO_BLACK,
}
OKABE_ITO_CYCLE = tuple(OKABE_ITO_PALETTE.values())

PAPER_RCPARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "Arimo",
        "DejaVu Sans",
    ],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.prop_cycle": cycler(color=OKABE_ITO_CYCLE),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 8,
    "grid.color": LIGHT_GREY,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def figure_size(
    width: float = COLUMN_WIDTH,
    height: float | None = None,
    *,
    aspect: float = 0.75,
) -> tuple[float, float]:
    """Return a figure size in inches using paper-width conventions."""
    if height is None:
        height = width * aspect
    return (width, height)


def standardize_figsize(figsize: tuple[float, float]) -> tuple[float, float]:
    """Snap a requested size to the nearest standard paper width."""
    width, height = figsize
    midpoint = (COLUMN_WIDTH + PAGE_WIDTH) / 2.0
    standard_width = COLUMN_WIDTH if width <= midpoint else PAGE_WIDTH
    return figure_size(width=standard_width, height=height)


def pad_axes(ax: Axes, *, x: float = 0.02, y: float = 0.02) -> None:
    """Expand current axis limits by a fractional margin of their span."""
    if x:
        left, right = ax.get_xlim()
        x_pad = abs(right - left) * x
        ax.set_xlim(left - x_pad, right + x_pad)
    if y:
        bottom, top = ax.get_ylim()
        y_pad = abs(top - bottom) * y
        ax.set_ylim(bottom - y_pad, top + y_pad)


@contextmanager
def style_context() -> Iterator[None]:
    """Temporarily apply the shared paper figure style."""
    with mpl.rc_context(PAPER_RCPARAMS):
        yield
