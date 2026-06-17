"""Thin CLI wrapper for stage 10 visualization outputs."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from binary_classifier.config import BinaryClassifierConfig, load_config
from binary_classifier.data.load import load_missions
from binary_classifier.log_utils import setup_logging
from binary_classifier.paths import PathRegistry
from binary_classifier.viz import (
    documentation_curve,
    ngram_log_odds,
    pr_curve,
    prevalence_forest,
    reliability_diagram,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("config/religious_missions.yaml")
_PR_POINT_KEYS = (
    "pr_curve_points",
    "precision_recall_curve",
    "precision_recall_points",
    "pr_points",
    "points",
)
_RELIABILITY_POINT_KEYS = (
    "reliability_curve",
    "calibration_curve",
    "calibration_bins",
    "reliability_points",
    "points",
)
_ECE_KEYS = ("ece", "expected_calibration_error")
_TEXT_COLUMNS = ("mission_text", "text")


def run_visualization(cfg: BinaryClassifierConfig, registry: PathRegistry) -> None:
    """Render every available stage-10 visualization artifact.

    Missing or schema-incompatible inputs are logged as skips so the script can
    be run before later roadmap stages have produced every optional artifact.

    Args:
        cfg: Validated task configuration.
        registry: Path registry rooted at the selected config.

    Returns:
        None.

    """
    registry.figures_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for render_step in (
        _maybe_render_documentation_curve,
        _maybe_render_pr_curve,
        _maybe_render_reliability_diagram,
        _maybe_render_prevalence_forest,
        _maybe_render_ngram_log_odds,
    ):
        if render_step(cfg, registry):
            rendered += 1

    if rendered == 0:
        logger.warning("No figures rendered; all visualization inputs were skipped.")
        return
    logger.info("Rendered %d figure(s) to %s", rendered, registry.figures_dir)


def _maybe_render_documentation_curve(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render the learning-curve figure if the JSONL exists.

    Args:
        _cfg: Unused task configuration, accepted for a uniform renderer
            signature.
        registry: Path registry with the learning-curve results path.

    Returns:
        True when a figure is written, otherwise False.

    """
    path = registry.learning_curve_results
    if not path.exists():
        logger.warning("Skipping documentation curve; missing input: %s", path)
        return False
    try:
        rows = _load_jsonl_rows(path)
        _save_plot(
            registry,
            "documentation_curve",
            lambda ax: documentation_curve(rows, ax),
            figsize=(7.0, 4.5),
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Skipping documentation curve from %s: %s", path, exc)
        return False
    return True


def _maybe_render_pr_curve(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render a precision-recall curve from available evaluation JSON.

    Args:
        _cfg: Unused task configuration, accepted for a uniform renderer
            signature.
        registry: Path registry with evaluation artifact paths.

    Returns:
        True when a figure is written, otherwise False.

    """
    return _maybe_render_json_points(
        registry,
        figure_name="precision_recall_curve",
        title="precision-recall curve",
        input_paths=(registry.test_evaluation, registry.calibrator_path),
        extract_points=lambda payload: _find_nested(payload, _PR_POINT_KEYS),
        draw=pr_curve,
        figsize=(5.5, 4.5),
    )


def _maybe_render_reliability_diagram(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render a reliability diagram from available evaluation JSON.

    Args:
        _cfg: Unused task configuration, accepted for a uniform renderer
            signature.
        registry: Path registry with evaluation artifact paths.

    Returns:
        True when a figure is written, otherwise False.

    """
    return _maybe_render_json_points(
        registry,
        figure_name="reliability_diagram",
        title="reliability diagram",
        input_paths=(registry.test_evaluation, registry.calibrator_path),
        extract_points=_reliability_payload,
        draw=reliability_diagram,
        figsize=(5.5, 4.5),
    )


def _maybe_render_prevalence_forest(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render the NTEE prevalence forest plot if the CSV exists.

    Args:
        _cfg: Unused task configuration, accepted for a uniform renderer
            signature.
        registry: Path registry with the prevalence CSV path.

    Returns:
        True when a figure is written, otherwise False.

    """
    path = registry.prevalence_by_ntee
    if not path.exists():
        logger.warning("Skipping prevalence forest; missing input: %s", path)
        return False
    try:
        frame = pd.read_csv(path)
        _save_plot(
            registry,
            "prevalence_forest",
            lambda ax: prevalence_forest(frame, ax),
            figsize=(8.0, max(4.0, 0.35 * len(frame) + 1.5)),
        )
    except (OSError, ValueError) as exc:
        logger.warning("Skipping prevalence forest from %s: %s", path, exc)
        return False
    return True


def _maybe_render_ngram_log_odds(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render n-gram log-odds bars if silver labels and text are available.

    Args:
        cfg: Validated task configuration used for mission-text loading.
        registry: Path registry with the silver-label and figure paths.

    Returns:
        True when a figure is written, otherwise False.

    """
    path = registry.silver_labels
    if not path.exists():
        logger.warning("Skipping n-gram log odds; missing input: %s", path)
        return False
    try:
        silver = pd.read_csv(path)
        silver_with_text = _silver_with_text(cfg, silver)
        _save_plot(
            registry,
            "ngram_log_odds",
            lambda ax: ngram_log_odds(silver_with_text, ax, top_k=30),
            figsize=(8.0, 7.0),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning("Skipping n-gram log odds from %s: %s", path, exc)
        return False
    return True


def _maybe_render_json_points(
    registry: PathRegistry,
    *,
    figure_name: str,
    title: str,
    input_paths: Sequence[Path],
    extract_points: Callable[[object], object | None],
    draw: Callable[[object, Axes], None],
    figsize: tuple[float, float],
) -> bool:
    """Render the first usable point payload from JSON artifacts.

    Args:
        registry: Path registry with the output figure directory.
        figure_name: Base filename for the emitted PNG and SVG.
        title: Human-readable figure name used in log messages.
        input_paths: Candidate JSON artifacts, checked in priority order.
        extract_points: Function that extracts a point payload from a JSON
            object, or returns None when unavailable.
        draw: Plotting helper accepting ``(points, ax)``.
        figsize: Matplotlib figure size.

    Returns:
        True when a figure is written, otherwise False.

    """
    existing_paths = [path for path in input_paths if path.exists()]
    if not existing_paths:
        logger.warning(
            "Skipping %s; missing inputs: %s",
            title,
            ", ".join(str(path) for path in input_paths),
        )
        return False

    for path in existing_paths:
        try:
            payload = _load_json(path)
            points = extract_points(payload)
            if points is None:
                logger.warning("No %s points found in %s", title, path)
                continue
            _save_plot(
                registry,
                figure_name,
                lambda ax, points=points: draw(points, ax),
                figsize=figsize,
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Skipping %s from %s: %s", title, path, exc)
            continue
        return True

    logger.warning("Skipping %s; no usable point payloads found.", title)
    return False


def _save_plot(
    registry: PathRegistry,
    name: str,
    draw: Callable[[Axes], None],
    *,
    figsize: tuple[float, float],
) -> None:
    """Draw and save a plot as PNG and SVG.

    Args:
        registry: Path registry with the output figure directory.
        name: Base filename for both output formats.
        draw: Callable that draws on the provided axes.
        figsize: Matplotlib figure size.

    Returns:
        None.

    """
    fig, ax = plt.subplots(figsize=figsize)
    try:
        draw(ax)
        fig.tight_layout()
        png_path = registry.figures_dir / f"{name}.png"
        svg_path = registry.figures_dir / f"{name}.svg"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        logger.info("Rendered %s to %s and %s", name, png_path, svg_path)
    finally:
        plt.close(fig)


def _load_json(path: Path) -> object:
    """Load a JSON artifact.

    Args:
        path: JSON file path.

    Returns:
        Decoded JSON object.

    """
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_rows(path: Path) -> list[Mapping[str, Any]]:
    """Load mapping rows from a JSONL artifact.

    Args:
        path: JSONL file path.

    Returns:
        List of decoded mapping rows.

    """
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            logger.warning("Ignoring non-object JSONL row %d in %s", line_number, path)
            continue
        rows.append({str(key): value for key, value in payload.items()})
    return rows


def _find_nested(payload: object, keys: Sequence[str]) -> object | None:
    """Find the first non-null nested value for any key.

    Args:
        payload: Arbitrary decoded JSON-like object.
        keys: Candidate keys in priority order.

    Returns:
        The first matching value, or None when no key is present.

    """
    if isinstance(payload, Mapping):
        payload_mapping = cast(Mapping[Any, Any], payload)
        for key in keys:
            value = payload_mapping.get(key)
            if value is not None:
                return value
        for value in payload_mapping.values():
            nested = _find_nested(value, keys)
            if nested is not None:
                return nested
    if isinstance(payload, Sequence) and not isinstance(
        payload,
        (str, bytes, bytearray),
    ):
        for value in payload:
            nested = _find_nested(value, keys)
            if nested is not None:
                return nested
    return None


def _reliability_payload(payload: object) -> object | None:
    """Extract reliability points while preserving ECE when available.

    Args:
        payload: Arbitrary decoded JSON-like object.

    Returns:
        A payload accepted by ``reliability_diagram``, or None.

    """
    points = _find_nested(payload, _RELIABILITY_POINT_KEYS)
    if points is None:
        return None
    ece = _find_nested(payload, _ECE_KEYS)
    if ece is None:
        return points
    return {"reliability_curve": points, "ece": ece}


def _silver_with_text(
    cfg: BinaryClassifierConfig,
    silver: pd.DataFrame,
) -> pd.DataFrame:
    """Return silver labels with mission text attached.

    Args:
        cfg: Validated task configuration used for mission loading.
        silver: Silver-label artifact DataFrame.

    Returns:
        DataFrame containing silver labels and a text column.

    Raises:
        ValueError: If ``EIN2`` is missing or no mission text can be joined.

    """
    if any(column in silver.columns for column in _TEXT_COLUMNS):
        return silver
    if "EIN2" not in silver.columns:
        raise ValueError("silver label artifact is missing EIN2.")

    missions = load_missions(cfg)
    if "EIN2" not in missions.columns or "mission_text" not in missions.columns:
        raise ValueError("missions data is missing EIN2 or mission_text.")

    silver_norm = silver.copy()
    missions_norm = missions[["EIN2", "mission_text"]].copy()
    silver_norm["EIN2"] = silver_norm["EIN2"].astype(str).str.strip()
    missions_norm["EIN2"] = missions_norm["EIN2"].astype(str).str.strip()
    joined = silver_norm.merge(missions_norm, on="EIN2", how="inner")
    if joined.empty:
        raise ValueError("no silver-label EIN2 values matched mission text.")
    return joined


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.

    """
    parser = argparse.ArgumentParser(description="Run stage 10 visualizations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to the task configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and render available visualization figures."""
    setup_logging(stem="10_visualize")

    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)
    run_visualization(cfg, registry)


if __name__ == "__main__":
    main()
