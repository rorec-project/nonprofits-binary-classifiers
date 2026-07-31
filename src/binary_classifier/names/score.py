"""Cross-field transfer scoring for cleaned organization names."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from binary_classifier.data.quality import apply_rule_label
from binary_classifier.inference import load_selected_model, score_texts
from binary_classifier.names.cleaner import normalize_name

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


_REQUIRED_COLUMNS = {"EIN2", "population", "name_raw", "name_cleaned"}
_OUTPUT_COLUMNS = [
    "EIN2",
    "population",
    "input_variant",
    "name_input",
    "prob_raw",
    "lexicon_rule_label",
    "calibration_status",
    "thresholds_transferable",
    "threshold",
    "threshold_maxf1",
    "threshold_baserate",
    "model_id",
    "checkpoint_sha256",
    "inference_date",
    "config_hash",
]


# ── Scoring orchestration ─────────────────────────────────────────────────────
def score_names(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    *,
    predictor: Any | None = None,
) -> None:
    """Score both name variants without applying mission routing or calibration.

    The primary suffix-stripped input and suffix-retaining ablation share the same
    encoding repair and truecasing. Mission-derived calibration and thresholds are
    intentionally not applied because they do not transfer across text fields. The
    mission cut points remain in the artifact as explicitly non-transferable
    provenance, not as name-classification decisions.
    """
    # Load both populations before scoring so every reachable EIN2 receives both variants.
    panel = _load_cleaned_frame(registry.names_panel_cleaned)
    bmf_only = _load_cleaned_frame(registry.names_bmf_only_cleaned)
    names = pd.concat([panel, bmf_only], ignore_index=True)
    _assert_unique_ein2(names)

    # Preserve mission operating points for auditability without applying them to names.
    thresholds = _load_mission_thresholds(registry)
    selected = load_selected_model(registry, require_checkpoint=predictor is None)

    records = _score_variants(
        cfg,
        names,
        selected,
        thresholds=thresholds,
        predictor=predictor,
        predictor_cache={},
    )

    registry.ensure_dirs()
    records.to_parquet(registry.names_scores, index=False)


# ── Input validation and mission-score provenance ─────────────────────────────
def _load_cleaned_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame.copy()


def _assert_unique_ein2(frame: pd.DataFrame) -> None:
    ein2 = frame["EIN2"].astype("string").str.strip()
    if ein2.isna().any() or ein2.eq("").any():
        raise ValueError("Cleaned name frames contain missing EIN2 values.")
    if ein2.duplicated().any():
        duplicated = ein2.loc[ein2.duplicated()].head().tolist()
        raise ValueError(
            f"Cleaned name frames contain duplicate EIN2 values: {duplicated}"
        )


def _load_mission_thresholds(registry: PathRegistry) -> dict[str, float]:
    """Load mission cut points retained as non-transferable score metadata."""
    calibrator_path = registry.calibrator_path
    if not calibrator_path.exists():
        raise RuntimeError(
            f"Calibrator artifact not found at {calibrator_path}. Run stage 07 first.",
        )
    calibrator = json.loads(calibrator_path.read_text())
    if not isinstance(calibrator, dict):
        raise ValueError(f"{calibrator_path} must be a JSON object.")

    base_rate_path = registry.base_rate_precision
    if not base_rate_path.exists():
        raise RuntimeError(
            f"Base-rate precision artifact not found at {base_rate_path}. "
            "Run stage 07 first.",
        )
    base_rate = json.loads(base_rate_path.read_text())
    if not isinstance(base_rate, dict):
        raise ValueError(f"{base_rate_path} must be a JSON object.")

    try:
        return {
            "threshold": float(calibrator["threshold"]),
            "threshold_maxf1": float(calibrator["max_f1_threshold"]),
            "threshold_baserate": float(base_rate["threshold"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Mission threshold artifacts must contain numeric threshold values.",
        ) from exc


# ── Input variants and raw-score persistence ──────────────────────────────────
def _score_variants(
    cfg: BinaryClassifierConfig,
    names: pd.DataFrame,
    selected: dict[str, Any],
    *,
    thresholds: dict[str, float],
    predictor: Any | None,
    predictor_cache: dict[str, Any],
) -> pd.DataFrame:
    variants = {
        "suffix_stripped": names["name_cleaned"].astype(str),
        "suffix_retaining": names["name_raw"].map(
            lambda value: normalize_name(value, strip_suffix=False),
        ),
    }
    timestamp = datetime.now(UTC).isoformat()
    config_hash = _config_hash(cfg)
    model_id, checkpoint_sha256 = _extract_model_provenance(selected, predictor)
    frames: list[pd.DataFrame] = []
    for variant, texts in variants.items():
        frame = names[["EIN2", "population"]].copy()
        frame["input_variant"] = variant
        frame["name_input"] = texts
        frame["prob_raw"] = score_texts(
            cfg,
            selected,
            texts.tolist(),
            predictor=predictor,
            predictor_cache=predictor_cache,
        )
        frame["lexicon_rule_label"] = texts.map(apply_rule_label)
        frame["calibration_status"] = "mission_calibration_invalid"
        frame["thresholds_transferable"] = False
        for column, value in thresholds.items():
            frame[column] = value
        frame["model_id"] = model_id
        frame["checkpoint_sha256"] = checkpoint_sha256
        frame["inference_date"] = timestamp
        frame["config_hash"] = config_hash
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[_OUTPUT_COLUMNS]


# ── Model and configuration provenance ────────────────────────────────────────
def _extract_model_provenance(
    selected: dict[str, Any],
    predictor: Any | None,
) -> tuple[str, str]:
    model_id = str(
        selected.get("encoder_id")
        or selected.get("tokenizer_id")
        or selected.get("model_id")
        or getattr(predictor, "model_id", ""),
    ).strip()
    checkpoint_sha256 = str(
        selected.get("checkpoint_sha256")
        or getattr(predictor, "checkpoint_sha256", ""),
    ).strip()
    if not model_id:
        raise ValueError("Selected model is missing an identity for names scoring.")
    if not checkpoint_sha256:
        raise ValueError(
            "Selected model is missing checkpoint_sha256 for names scoring."
        )
    return model_id, checkpoint_sha256


def _config_hash(cfg: BinaryClassifierConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
