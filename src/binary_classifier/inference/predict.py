"""Stage 08 batch inference and shard management."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from binary_classifier.data.load import load_missions
from binary_classifier.data.quality import assign_tier, compute_quality_score
from binary_classifier.evaluation.calibration import (
    CalibrationMethod,
    apply_calibration,
)
from binary_classifier.inference.router import route

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

Precision = Literal["bf16", "fp32"]
Device = Literal["cuda", "mps", "cpu"]

_PREDICTION_COLUMNS = [
    "EIN2",
    "pred_label",
    "prob_raw",
    "prob_calibrated",
    "decision_source",
    "tier",
    "Q",
    "ntee_major_group",
    "model_id",
    "checkpoint_sha256",
    "calibrator_method",
    "calibrator_params_hash",
    "threshold",
    "inference_date",
    "pipeline_version",
    "config_hash",
]

_ALLOWED_DECISION_SOURCES = {
    "classifier",
    "rule_strong_positive",
    "rule_short_negative",
    "rule_abstain",
    "low_via_classifier",
}

_RESUME_METADATA_COLUMNS = [
    "model_id",
    "checkpoint_sha256",
    "calibrator_method",
    "calibrator_params_hash",
    "threshold",
    "pipeline_version",
    "config_hash",
]


def run_inference(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    *,
    predictor: Any | None = None,
    limit: int | None = None,
) -> None:
    """Run full-corpus stage-08 inference with resumable parquet shards.

    Args:
        cfg: Validated binary-classifier configuration.
        registry: Path registry exposing model, calibration, and prediction paths.
        predictor: Optional injected predictor exposing ``predict_proba(texts)``.
            Defaults to loading the reviewed selected checkpoint.
        limit: Optional row limit for smoke runs after stable EIN2 sorting.

    Raises:
        RuntimeError: If required model/calibrator artifacts are missing.
        ValueError: If inputs, predictor outputs, or shard contents are malformed.

    """
    registry.ensure_dirs()
    shard_size = _positive_int(cfg.inference.shard_size, "inference.shard_size")
    batch_size = _positive_int(cfg.inference.batch_size, "inference.batch_size")

    logger.info("Loading full mission corpus for inference...")
    missions = _prepare_inference_frame(cfg, load_missions(cfg), limit=limit)
    logger.info("Prepared %d inference rows", len(missions))

    calibrator = _load_calibrator(registry.calibrator_path)
    selected = _load_selected_model(registry, require_checkpoint=predictor is None)
    device, precision = resolve_device_precision(cfg)
    scorer = (
        predictor
        if predictor is not None
        else _load_checkpoint_predictor(selected, device=device)
    )
    metadata = _prediction_metadata(
        cfg,
        selected=selected,
        predictor=scorer,
        calibrator=calibrator,
    )

    shard_paths = _process_shards(
        cfg,
        registry,
        missions,
        predictor=scorer,
        calibrator=calibrator,
        metadata=metadata,
        shard_size=shard_size,
        batch_size=batch_size,
        device=device,
        precision=precision,
    )
    merged = _merge_shards(shard_paths)
    _validate_ein2_completeness(missions["EIN2"], merged["EIN2"])
    merged.to_parquet(registry.predictions_parquet, index=False)
    logger.info("Wrote merged predictions to %s", registry.predictions_parquet)

    _write_monitor_scores(
        registry,
        merged,
        metadata=metadata,
        device=device,
        precision=precision,
        limit=limit,
    )


def resolve_device_precision(cfg: "BinaryClassifierConfig") -> tuple[Device, Precision]:
    """Resolve stage-08 device and precision from runtime capabilities.

    The policy is capability-based: ``auto`` chooses CUDA, then MPS, then CPU;
    bf16 is used only when the resolved device is CUDA and bf16 is supported.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - production environment issue
        if cfg.inference.device != "auto":
            raise RuntimeError(
                "Torch is required for explicit inference devices."
            ) from exc
        return "cpu", "fp32"

    requested = cfg.inference.device
    if requested == "auto":
        if torch.cuda.is_available():
            device: Device = "cuda"
        elif _mps_available(torch):
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cfg.inference.device='cuda' but CUDA is unavailable.")
        if device == "mps" and not _mps_available(torch):
            raise RuntimeError("cfg.inference.device='mps' but MPS is unavailable.")

    precision: Precision = (
        "bf16" if device == "cuda" and bool(torch.cuda.is_bf16_supported()) else "fp32"
    )
    logger.info("Resolved inference device=%s precision=%s", device, precision)
    return device, precision


def _prepare_inference_frame(
    cfg: "BinaryClassifierConfig",
    missions: pd.DataFrame,
    *,
    limit: int | None,
) -> pd.DataFrame:
    required = {"EIN2", "mission_text", "ntee_major_group"}
    missing = required - set(missions.columns)
    if missing:
        raise ValueError(f"Mission frame missing columns {sorted(missing)}.")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative when provided.")

    frame = missions.copy()
    frame["EIN2"] = _normalize_ein2(frame["EIN2"])
    if frame["EIN2"].duplicated().any():
        duplicated = frame.loc[frame["EIN2"].duplicated(), "EIN2"].head().tolist()
        raise ValueError(f"Mission frame contains duplicate EIN2 values: {duplicated}.")
    frame = frame.sort_values("EIN2", kind="mergesort").reset_index(drop=True)
    if limit is not None:
        frame = frame.head(limit).copy()

    frame["mission_text"] = frame["mission_text"].fillna("").astype(str)
    frame["Q"] = frame["mission_text"].map(compute_quality_score).astype(float)
    frame["tier"] = frame["Q"].map(lambda q: assign_tier(float(q), cfg.q_thresholds))
    routed = [
        route(text, tier, cfg)
        for text, tier in zip(frame["mission_text"], frame["tier"], strict=True)
    ]
    frame["decision_source"] = [item[0] for item in routed]
    frame["rule_label"] = [item[1] for item in routed]
    unknown = set(frame["decision_source"].unique()) - _ALLOWED_DECISION_SOURCES
    if unknown:
        raise ValueError(
            "Router produced decision source(s) unsupported by the stage-08 "
            f"prediction schema: {sorted(unknown)}."
        )
    return frame


def _load_calibrator(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Calibrator artifact not found at {path}. Run stage 07 before stage 08."
        )
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object.")
    method = raw.get("method")
    params = raw.get("params")
    if method not in {"platt", "temperature"}:
        raise ValueError(f"{path} method must be 'platt' or 'temperature'.")
    if not isinstance(params, dict):
        raise ValueError(f"{path} must contain a params object.")
    if "threshold" not in raw:
        raise ValueError(f"{path} missing threshold.")
    return dict(raw)


def _load_selected_model(
    registry: "PathRegistry",
    *,
    require_checkpoint: bool,
) -> dict[str, Any]:
    path = registry.selected_model
    if not path.exists():
        if require_checkpoint:
            raise RuntimeError(
                f"Selected model artifact not found at {path}. Run stage 06, copy "
                "the reviewed selected_model_skeleton to selected_model.json, "
                "then re-run stage 08."
            )
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object.")
    selected = dict(raw)
    relpath = str(selected.get("checkpoint_relpath", "")).strip()
    expected_sha = str(selected.get("checkpoint_sha256", "")).strip()
    if not require_checkpoint:
        return selected
    if not relpath or not expected_sha or expected_sha.startswith("TODO_"):
        raise RuntimeError(
            f"{path} must contain reviewed checkpoint_relpath and checkpoint_sha256."
        )
    checkpoint_path = _checkpoint_path(registry.models_dir, relpath)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Selected checkpoint file not found at {checkpoint_path}.")
    actual_sha = _sha256_file(checkpoint_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Selected checkpoint SHA-256 mismatch for {checkpoint_path}: expected "
            f"{expected_sha}, got {actual_sha}."
        )
    selected["checkpoint_path"] = str(checkpoint_path)
    return selected


def _load_checkpoint_predictor(selected: Mapping[str, Any], *, device: Device) -> Any:
    """Load a Hugging Face sequence-classification checkpoint lazily."""
    checkpoint_path = Path(str(selected["checkpoint_path"]))
    model_dir = checkpoint_path.parent
    tokenizer_id = str(selected.get("tokenizer_id") or model_dir)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised only in real runs
        raise RuntimeError(
            "Default stage-08 predictor loading requires torch and transformers. "
            "Install runtime dependencies or pass an injected predictor with "
            "predict_proba(texts)."
        ) from exc

    tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_id)
    model: Any = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    class _HFPredictor:
        model_id = str(selected.get("encoder_id") or tokenizer_id)
        checkpoint_sha256 = str(selected.get("checkpoint_sha256") or "unknown")

        def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
            encoded = cast(Any, tokenizer)(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            return np.asarray(probs, dtype=float)

    return _HFPredictor()


def _process_shards(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    frame: pd.DataFrame,
    *,
    predictor: Any,
    calibrator: Mapping[str, Any],
    metadata: Mapping[str, Any],
    shard_size: int,
    batch_size: int,
    device: Device,
    precision: Precision,
) -> list[Path]:
    paths: list[Path] = []
    shards_dir = registry.predictions_dir / "shards"
    for shard_index, start in enumerate(range(0, len(frame), shard_size)):
        shard_path = shards_dir / f"shard_{shard_index:05d}.parquet"
        shard = frame.iloc[start : start + shard_size].copy().reset_index(drop=True)
        paths.append(shard_path)
        if shard_path.exists():
            if _existing_shard_matches(shard_path, metadata, shard["EIN2"].tolist()):
                logger.warning("Skipping existing inference shard %s", shard_path)
                continue
            logger.warning(
                "Rewriting inference shard %s because it does not match current run",
                shard_path,
            )
        predictions = _predict_shard(
            cfg,
            shard,
            predictor=predictor,
            calibrator=calibrator,
            metadata=metadata,
            batch_size=batch_size,
            device=device,
            precision=precision,
        )
        predictions.to_parquet(shard_path, index=False)
        logger.info("Wrote inference shard %s (%d rows)", shard_path, len(predictions))
    return paths


def _existing_shard_matches(
    path: Path,
    metadata: Mapping[str, Any],
    expected_ein2: Sequence[str],
) -> bool:
    """Return whether a resumable shard belongs to the current run."""
    try:
        shard = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Cannot validate existing shard %s: %s", path, exc)
        return False
    if shard.columns.tolist() != _PREDICTION_COLUMNS:
        logger.warning(
            "Existing shard %s has incompatible prediction schema",
            path,
        )
        return False

    expected_ids = [str(ein2).strip() for ein2 in expected_ein2]
    observed_ids = _normalize_ein2(shard["EIN2"]).tolist()
    if len(observed_ids) != len(expected_ids) or observed_ids != expected_ids:
        return False

    for column in _RESUME_METADATA_COLUMNS:
        expected = metadata.get(column)
        if expected is None:
            return False
        values = shard[column].dropna().unique()
        if len(values) != 1:
            return False
        observed = values[0]
        if column == "threshold":
            try:
                if not np.isclose(float(observed), float(expected)):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(observed) != str(expected):
            return False
    return True


def _predict_shard(
    cfg: "BinaryClassifierConfig",
    shard: pd.DataFrame,
    *,
    predictor: Any,
    calibrator: Mapping[str, Any],
    metadata: Mapping[str, Any],
    batch_size: int,
    device: Device,
    precision: Precision,
) -> pd.DataFrame:
    del cfg
    output = shard[["EIN2", "decision_source", "tier", "Q", "ntee_major_group"]].copy()
    output["prob_raw"] = np.nan
    output["prob_calibrated"] = np.nan
    output["pred_label"] = np.nan

    rule_mask = shard["decision_source"].isin(
        {"rule_strong_positive", "rule_short_negative"}
    )
    output.loc[rule_mask, "pred_label"] = shard.loc[rule_mask, "rule_label"].astype(int)

    abstain_mask = shard["decision_source"] == "rule_abstain"
    output.loc[abstain_mask, "pred_label"] = 0

    classifier_mask = ~rule_mask & ~abstain_mask
    if classifier_mask.any():
        texts = shard.loc[classifier_mask, "mission_text"].astype(str).tolist()
        raw = _batched_positive_probabilities(
            predictor,
            texts,
            batch_size=batch_size,
            device=device,
            precision=precision,
        )
        method = cast(CalibrationMethod, calibrator["method"])
        params = cast(Mapping[str, float], calibrator["params"])
        calibrated = np.asarray(
            apply_calibration(raw.tolist(), method, params), dtype=float
        )
        threshold = float(calibrator["threshold"])
        classifier_index = output.index[classifier_mask]
        output.loc[classifier_index, "prob_raw"] = raw
        output.loc[classifier_index, "prob_calibrated"] = calibrated
        output.loc[classifier_index, "pred_label"] = (calibrated >= threshold).astype(
            int
        )

    if output["pred_label"].isna().any():
        raise ValueError("Inference shard contains unrouted rows with no pred_label.")
    for key, value in metadata.items():
        output[key] = value
    return output[_PREDICTION_COLUMNS].copy()


def _batched_positive_probabilities(
    predictor: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    device: Device,
    precision: Precision,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        with _torch_inference_context(device, precision):
            chunks.append(_predict_positive_probabilities(predictor, batch))
    if not chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(chunks).astype(float)


def _predict_positive_probabilities(predictor: Any, texts: Sequence[str]) -> np.ndarray:
    if not hasattr(predictor, "predict_proba"):
        raise ValueError("Predictor must expose predict_proba(texts).")
    raw = predictor.predict_proba(list(texts))
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        p1 = arr
    elif arr.ndim == 2 and arr.shape[1] == 2:
        p1 = arr[:, 1]
        if not np.allclose(arr[:, 0] + arr[:, 1], 1.0, atol=1e-6):
            raise ValueError("predict_proba columns must sum to 1.")
    else:
        raise ValueError("predict_proba must return shape (n,) or (n, 2).")
    if len(p1) != len(texts):
        raise ValueError(
            f"predict_proba returned {len(p1)} rows for {len(texts)} input texts."
        )
    if not np.isfinite(p1).all() or ((p1 < 0.0) | (p1 > 1.0)).any():
        raise ValueError("predict_proba probabilities must be finite in [0, 1].")
    return p1.astype(float)


@contextmanager
def _torch_inference_context(device: Device, precision: Precision):
    try:
        import torch
    except Exception:  # pragma: no cover - stub predictors can run without torch
        yield
        return
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda" and precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        yield


def _merge_shards(paths: Sequence[Path]) -> pd.DataFrame:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing inference shard(s): {missing}.")
    if not paths:
        return pd.DataFrame(columns=_PREDICTION_COLUMNS)
    frames = [pd.read_parquet(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    missing_columns = set(_PREDICTION_COLUMNS) - set(merged.columns)
    if missing_columns:
        raise ValueError(
            f"Merged predictions missing columns {sorted(missing_columns)}."
        )
    return merged[_PREDICTION_COLUMNS].copy()


def _write_monitor_scores(
    registry: "PathRegistry",
    predictions: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    device: Device,
    precision: Precision,
    limit: int | None,
) -> None:
    path = registry.monitor_manifest
    if not path.exists():
        raise RuntimeError(
            f"Monitor manifest not found at {path}. Run stage 01 before stage 08."
        )
    manifest = pd.read_csv(path)
    if "EIN2" not in manifest.columns:
        raise ValueError(f"{path} missing EIN2 column.")
    monitor_ids = _normalize_ein2(manifest["EIN2"])
    pred = predictions.copy()
    pred["EIN2"] = _normalize_ein2(pred["EIN2"])
    monitor = pd.DataFrame({"EIN2": monitor_ids}).merge(pred, on="EIN2", how="left")
    missing = monitor["pred_label"].isna()
    if missing.any():
        examples = monitor.loc[missing, "EIN2"].head().tolist()
        raise ValueError(f"Monitor EIN2(s) absent from predictions: {examples}.")
    rows = monitor[
        [
            "EIN2",
            "pred_label",
            "prob_raw",
            "prob_calibrated",
            "decision_source",
            "tier",
            "Q",
        ]
    ].to_dict(orient="records")
    payload = {
        "metadata": {
            **dict(metadata),
            "device": device,
            "precision": precision,
            "limit": limit,
            "n_monitor": int(len(monitor)),
            "monitor_manifest": str(path),
            "predictions_parquet": str(registry.predictions_parquet),
        },
        "rows": rows,
    }
    registry.monitor_scores.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    )
    logger.info("Wrote monitor scores to %s", registry.monitor_scores)


def _validate_ein2_completeness(expected: pd.Series, observed: pd.Series) -> None:
    expected_ids = _normalize_ein2(expected)
    observed_ids = _normalize_ein2(observed)
    if len(observed_ids) != len(expected_ids):
        raise ValueError(
            f"Merged predictions contain {len(observed_ids)} rows for "
            f"{len(expected_ids)} input EIN2s."
        )
    if observed_ids.duplicated().any():
        duplicated = observed_ids.loc[observed_ids.duplicated()].head().tolist()
        raise ValueError(f"Merged predictions contain duplicate EIN2s: {duplicated}.")
    if set(observed_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_ids))[:5]
        extra = sorted(set(observed_ids) - set(expected_ids))[:5]
        raise ValueError(
            "Merged predictions do not match input EIN2s: "
            f"missing={missing}, extra={extra}."
        )


def _prediction_metadata(
    cfg: "BinaryClassifierConfig",
    *,
    selected: Mapping[str, Any],
    predictor: Any,
    calibrator: Mapping[str, Any],
) -> dict[str, Any]:
    model_id = (
        selected.get("encoder_id")
        or selected.get("tokenizer_id")
        or selected.get("model_id")
        or getattr(predictor, "model_id", "unknown")
    )
    checkpoint_sha256 = selected.get("checkpoint_sha256") or getattr(
        predictor,
        "checkpoint_sha256",
        "unknown",
    )
    return {
        "model_id": str(model_id),
        "checkpoint_sha256": str(checkpoint_sha256),
        "calibrator_method": str(calibrator["method"]),
        "calibrator_params_hash": _calibrator_params_hash(calibrator),
        "threshold": float(calibrator["threshold"]),
        "inference_date": datetime.now(UTC).isoformat(),
        "pipeline_version": _pipeline_version(),
        "config_hash": _config_hash(cfg),
    }


def _calibrator_params_hash(calibrator: Mapping[str, Any]) -> str:
    """Return a stable digest of learned calibration parameters."""
    payload = json.dumps(calibrator.get("params", {}), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _normalize_ein2(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _mps_available(torch: Any) -> bool:
    return bool(
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _checkpoint_path(models_dir: Path, relpath: str) -> Path:
    candidate = Path(relpath)
    if candidate.is_absolute():
        return candidate
    return models_dir / candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(cfg: "BinaryClassifierConfig") -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _pipeline_version() -> str:
    try:
        return importlib.metadata.version("nonprofits-binary-classifiers")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
