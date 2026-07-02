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
    bakeoff_summary,
    canary_drift,
    documentation_curve,
    frozen_test_confusion_matrices,
    frozen_test_curves,
    ngram_log_odds,
    prevalence_decomposition,
    prevalence_forest,
    quantification_sensitivity,
    production_annotation_summary,
    reliability_diagram,
    rule_validation_intervals,
    score_distribution_by_tier_label,
    subgroup_performance,
)
from binary_classifier.viz.style import (
    PAGE_WIDTH,
    figure_size,
    standardize_figsize,
    style_context,
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
        _maybe_render_bakeoff_summary,
        _maybe_render_production_summary,
        _maybe_render_canary_drift,
        _maybe_render_documentation_curve,
        _maybe_render_pr_curve,
        _maybe_render_frozen_test_confusion_matrices,
        _maybe_render_reliability_diagram,
        _maybe_render_score_distribution,
        _maybe_render_prevalence_forest,
        _maybe_render_prevalence_decomposition,
        _maybe_render_rule_validation_intervals,
        _maybe_render_quantification_sensitivity,
        _maybe_render_subgroup_performance,
        _maybe_render_ngram_log_odds,
    ):
        if render_step(cfg, registry):
            rendered += 1

    if rendered == 0:
        logger.warning("No figures rendered; all visualization inputs were skipped.")
        return
    logger.info("Rendered %d figure(s) to %s", rendered, registry.figures_dir)


def _maybe_render_bakeoff_summary(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render the bake-off summary figure if stage-02 results exist.

    Args:
        _cfg: Unused task configuration, accepted for a uniform renderer
            signature.
        registry: Path registry with the bake-off results path.

    Returns:
        True when a figure is written, otherwise False.

    """
    path = registry.bakeoff_results
    if not path.exists():
        logger.warning("Skipping bake-off summary; missing input: %s", path)
        return False
    try:
        results = _load_json(path)
        _save_plot(
            registry,
            "bakeoff_summary",
            lambda ax: bakeoff_summary(results, ax),
            figsize=figure_size(width=PAGE_WIDTH, height=5.5),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Skipping bake-off summary from %s: %s", path, exc)
        return False
    return True


def _maybe_render_production_summary(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render production annotation diagnostics from available stage-03 CSVs.

    Args:
        _cfg: Unused task configuration, accepted for a uniform renderer
            signature.
        registry: Path registry with annotation-store and silver-label paths.

    Returns:
        True when a figure is written, otherwise False.

    """
    input_paths = (registry.annotation_store, registry.silver_labels)
    existing_paths = [path for path in input_paths if path.exists()]
    if not existing_paths:
        logger.warning(
            "Skipping production annotation summary; missing inputs: %s",
            ", ".join(str(path) for path in input_paths),
        )
        return False

    for path in existing_paths:
        try:
            frame = _production_summary_frame(path)
            sizing_frame = _production_summary_sizing_frame(frame)
            _save_plot(
                registry,
                "production_annotation_summary",
                lambda ax, frame=frame: production_annotation_summary(frame, ax),
                figsize=figure_size(
                    width=PAGE_WIDTH,
                    height=max(4.0, 0.35 * len(sizing_frame) + 1.5),
                ),
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "Skipping production annotation summary from %s: %s", path, exc
            )
            continue
        return True

    logger.warning("Skipping production annotation summary; no usable input found.")
    return False


def _maybe_render_canary_drift(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render canary drift diagnostics if the audit JSONL exists.

    Args:
        _cfg: Unused task configuration, accepted for a uniform renderer
            signature.
        registry: Path registry with the interim artifact directory.

    Returns:
        True when a figure is written, otherwise False.

    """
    path = registry.interim_dir / "canary_drift_audit.jsonl"
    if not path.exists():
        logger.warning("Skipping canary drift; missing input: %s", path)
        return False
    try:
        rows = _load_jsonl_rows(path)
        _save_plot(
            registry,
            "canary_drift",
            lambda ax: canary_drift(rows, ax),
            figsize=figure_size(width=PAGE_WIDTH, aspect=0.6),
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Skipping canary drift from %s: %s", path, exc)
        return False
    return True


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
    """Render frozen-test precision-recall and ROC curves when scores exist.

    Args:
        _cfg: Unused task configuration, accepted for a uniform renderer
            signature.
        registry: Path registry with evaluation artifact paths.

    Returns:
        True when a figure is written, otherwise False.

    """
    path = registry.test_evaluation
    if not path.exists():
        logger.warning("Skipping precision-recall curve; missing input: %s", path)
        return False
    try:
        payload = _load_json(path)
        if not isinstance(payload, Mapping) or not payload.get("test_scores"):
            logger.warning(
                "Skipping precision-recall curve; %s lacks frozen-test test_scores.",
                path,
            )
            return False
        payload_mapping = cast(Mapping[str, Any], payload)
        _save_plot(
            registry,
            "precision_recall_curve",
            lambda ax, payload=payload_mapping: frozen_test_curves(payload, ax),
            figsize=figure_size(width=PAGE_WIDTH, height=4.5),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Skipping precision-recall curve from %s: %s", path, exc)
        return False
    return True


def _maybe_render_frozen_test_confusion_matrices(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render frozen-test confusion matrices from persisted stage-07 JSON."""
    path = registry.test_evaluation
    if not path.exists():
        logger.warning(
            "Skipping frozen-test confusion matrices; missing input: %s", path
        )
        return False
    try:
        payload = _load_json(path)
        if not isinstance(payload, Mapping) or not payload.get("test_scores"):
            logger.warning(
                "Skipping frozen-test confusion matrices; %s lacks frozen-test test_scores.",
                path,
            )
            return False
        payload_mapping = cast(Mapping[str, Any], payload)
        _save_plot(
            registry,
            "frozen_test_confusion_matrices",
            lambda ax, payload=payload_mapping: frozen_test_confusion_matrices(
                payload, ax
            ),
            figsize=figure_size(width=PAGE_WIDTH, height=3.0),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Skipping frozen-test confusion matrices from %s: %s", path, exc)
        return False
    return True


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


def _maybe_render_score_distribution(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render calibrated-score distributions from stage-08 predictions."""
    path = registry.predictions_parquet
    if not path.exists():
        logger.warning("Skipping score distribution; missing input: %s", path)
        return False
    try:
        predictions = pd.read_parquet(path)
        thresholds = _score_distribution_thresholds(registry)
        _save_plot(
            registry,
            "score_distribution_by_tier_label",
            lambda ax: score_distribution_by_tier_label(predictions, thresholds, ax),
            figsize=figure_size(width=PAGE_WIDTH, height=5.0),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Skipping score distribution from %s: %s", path, exc)
        return False
    return True


def _maybe_render_prevalence_decomposition(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render prevalence component contributions from stage-09 report."""
    return _maybe_render_json_payload(
        registry,
        figure_name="prevalence_decomposition",
        title="prevalence decomposition",
        input_path=registry.prevalence_report,
        draw=prevalence_decomposition,
        figsize=figure_size(width=PAGE_WIDTH, height=4.0),
    )


def _maybe_render_rule_validation_intervals(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render Wilson intervals from rule-validation diagnostics."""
    return _maybe_render_json_payload(
        registry,
        figure_name="rule_validation_intervals",
        title="rule-validation intervals",
        input_path=registry.rule_validation,
        draw=rule_validation_intervals,
        figsize=figure_size(width=PAGE_WIDTH, height=2.8),
    )


def _maybe_render_quantification_sensitivity(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render PPI/EMQ weighting sensitivity from stage-09 report."""
    return _maybe_render_json_payload(
        registry,
        figure_name="quantification_sensitivity",
        title="quantification sensitivity",
        input_path=registry.prevalence_report,
        draw=quantification_sensitivity,
        figsize=figure_size(width=PAGE_WIDTH, height=3.8),
    )


def _maybe_render_subgroup_performance(
    _cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> bool:
    """Render frozen-test subgroup diagnostics from persisted test evaluation."""
    path = registry.test_evaluation
    if not path.exists():
        logger.warning("Skipping subgroup performance; missing input: %s", path)
        return False
    try:
        payload = _load_json(path)
        if not isinstance(payload, Mapping) or not payload.get("test_scores"):
            logger.warning(
                "Skipping subgroup performance; %s lacks frozen-test test_scores.",
                path,
            )
            return False
        payload_mapping = cast(Mapping[str, Any], payload)
        subgroups = payload_mapping.get("subgroups")
        _save_plot(
            registry,
            "subgroup_performance",
            lambda ax, subgroups=subgroups: subgroup_performance(subgroups, ax),
            figsize=figure_size(width=PAGE_WIDTH, height=5.2),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Skipping subgroup performance from %s: %s", path, exc)
        return False
    return True


def _maybe_render_json_payload(
    registry: PathRegistry,
    *,
    figure_name: str,
    title: str,
    input_path: Path,
    draw: Callable[[Mapping[str, Any], Axes], None],
    figsize: tuple[float, float],
) -> bool:
    """Render a plot directly from a JSON mapping payload."""
    if not input_path.exists():
        logger.warning("Skipping %s; missing input: %s", title, input_path)
        return False
    try:
        payload = _load_json(input_path)
        if not isinstance(payload, Mapping):
            raise ValueError("JSON payload must be an object.")
        payload_mapping = cast(Mapping[str, Any], payload)
        _save_plot(
            registry,
            figure_name,
            lambda ax, payload=payload_mapping: draw(payload, ax),
            figsize=figsize,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Skipping %s from %s: %s", title, input_path, exc)
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
    """Draw and save a plot as PDF, SVG, and PNG.

    Args:
        registry: Path registry with the output figure directory.
        name: Base filename for all output formats.
        draw: Callable that draws on the provided axes.
        figsize: Matplotlib figure size.

    Returns:
        None.

    """
    registry.figures_dir.mkdir(parents=True, exist_ok=True)
    with style_context():
        fig, ax = plt.subplots(figsize=standardize_figsize(figsize))
        try:
            draw(ax)
            fig.tight_layout()
            pdf_path = registry.figures_dir / f"{name}.pdf"
            svg_path = registry.figures_dir / f"{name}.svg"
            png_path = registry.figures_dir / f"{name}.png"
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(svg_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            logger.info(
                "Rendered %s to %s, %s, and %s",
                name,
                pdf_path,
                svg_path,
                png_path,
            )
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


def _production_summary_frame(path: Path) -> pd.DataFrame:
    """Read a production-summary CSV with minimal silver-label normalization."""
    frame = pd.read_csv(path)
    if "label" not in frame.columns and "silver_label" in frame.columns:
        frame = frame.rename(columns={"silver_label": "label"})
    if "source_id" not in frame.columns and path.name == "silver_labels.csv":
        frame = frame.assign(source_id="silver_label")
    return frame


def _production_summary_sizing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the diagnostic rows the production-summary plot will display."""
    if {"EIN2", "source_id", "label"} - set(frame.columns):
        return frame
    source_column = "model_id" if "model_id" in frame.columns else "source_id"
    n_sources = int(frame[source_column].nunique(dropna=True))
    labels = [f"source_{idx}" for idx in range(n_sources)]
    labels.extend(["Mean pairwise agreement", "Tie rate", "All-abstain rate"])
    return pd.DataFrame({"label": labels})


def _score_distribution_thresholds(registry: PathRegistry) -> dict[str, float]:
    """Load the operating, max-F1, and base-rate thresholds for score plots."""
    thresholds: dict[str, float] = {}
    if registry.calibrator_path.exists():
        payload = _load_json(registry.calibrator_path)
        if isinstance(payload, Mapping):
            calibrator = cast(Mapping[str, Any], payload)
            if calibrator.get("threshold") is not None:
                thresholds["operating"] = float(calibrator["threshold"])
            if calibrator.get("max_f1_threshold") is not None:
                thresholds["max_f1"] = float(calibrator["max_f1_threshold"])
    if registry.base_rate_precision.exists():
        payload = _load_json(registry.base_rate_precision)
        if isinstance(payload, Mapping):
            base_rate = cast(Mapping[str, Any], payload)
            if base_rate.get("threshold") is not None:
                thresholds["base_rate"] = float(base_rate["threshold"])
    if not thresholds:
        raise ValueError("No threshold artifacts available for score distribution.")
    return thresholds


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
