"""Run manifest emission for local and canonical pipeline runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


def write_run_manifest(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
) -> dict[str, Any]:
    """Write a reproducibility manifest from existing local artifacts."""
    payload = {
        "schema_version": 1,
        "stage": "run_manifest",
        "generated_at": datetime.now(UTC).isoformat(),
        "config_hash": _config_hash(cfg),
        "git_sha": _git_sha(),
        "thresholds": _thresholds(registry),
        "input_row_counts": _input_row_counts(registry),
        "wave2_completeness": _wave2_completeness(registry),
        "environment_lock": _environment_lock(),
        "post_sprint_reproduce_assertion": None,
    }
    registry.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    registry.run_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _config_hash(cfg: "BinaryClassifierConfig") -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _thresholds(registry: "PathRegistry") -> dict[str, float | None]:
    calibrator = _read_json_if_exists(registry.calibrator_path)
    base_rate = _read_json_if_exists(registry.base_rate_precision)
    return {
        "operating": _float_or_none(calibrator.get("threshold")),
        "max_f1": _float_or_none(calibrator.get("max_f1_threshold")),
        "base_rate": _float_or_none(base_rate.get("threshold")),
    }


def _input_row_counts(registry: "PathRegistry") -> dict[str, int | None]:
    return {
        "predictions_parquet": _parquet_rows(registry.predictions_parquet),
        "predictions_full_parquet": _parquet_rows(registry.predictions_full_parquet),
        "anchor_oof_scores": _parquet_rows(registry.anchor_oof_scores),
        "silver_manifest": _csv_rows(registry.silver_manifest),
        "gold_manifest": _csv_rows(registry.gold_manifest),
        "anchor_manifest": _csv_rows(registry.anchor_manifest),
    }


def _wave2_completeness(registry: "PathRegistry") -> dict[str, Any]:
    if not registry.predictions_full_parquet.exists():
        return {"status": "missing", "path": str(registry.predictions_full_parquet)}
    frame = pd.read_parquet(
        registry.predictions_full_parquet, columns=["EIN2", "pred_label"]
    )
    return {
        "status": "ok" if frame["pred_label"].notna().all() else "failed",
        "path": str(registry.predictions_full_parquet),
        "rows": int(len(frame)),
        "unique_ein2": int(frame["EIN2"].nunique(dropna=False)),
        "null_pred_label": int(frame["pred_label"].isna().sum()),
    }


def _environment_lock() -> dict[str, Any]:
    uv_lock = Path("uv.lock")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uv_lock": {
            "path": str(uv_lock),
            "sha256": _file_sha256(uv_lock) if uv_lock.exists() else None,
        },
        "cuda": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,cuda_version",
                "--format=csv,noheader",
            ]
        ),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(pq.ParquetFile(path).metadata.num_rows)


def _csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(len(pd.read_csv(path, usecols=[0])))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _float_or_none(value: object) -> float | None:
    if value is None or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
