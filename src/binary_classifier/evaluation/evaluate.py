"""Stage 07 frozen-test evaluation entrypoint."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from binary_classifier import metrics
from binary_classifier.data.load import load_missions
from binary_classifier.data.quality import apply_rule_label
from binary_classifier.evaluation.calibration import (
    CalibrationMethod,
    apply_calibration,
    calibration_metrics,
    crossfit_calibrate,
)
from binary_classifier.evaluation.decision_curve import net_benefit
from binary_classifier.evaluation.subgroups import subgroup_report
from binary_classifier.evaluation.thresholds import pick_threshold
from binary_classifier.qc.preflight import (
    _validate_anchor_labels,
    _validate_test_unlock,
)

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

_ANCHOR_SCORE_COLUMNS = [
    "EIN2",
    "prob_raw",
    "prob_calibrated_oof",
    "human_label",
    "tier",
    "sample_prob",
]


def run_evaluation(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    *,
    predictor: Any | None = None,
) -> None:
    """Run stage 07 calibration, rule validation, and frozen-test evaluation.

    Args:
        cfg: Validated binary-classifier configuration.
        registry: Path registry exposing selected-model, anchor, and evaluation
            artifact locations.
        predictor: Optional injected predictor exposing ``predict_proba(texts)``.
            Defaults to loading the selected checkpoint.

    Raises:
        RuntimeError: If a human gate fails, the selected checkpoint hash does not
            match, a one-shot test report already exists, or the acceptance gate
            fails.
        ValueError: If required artifacts have malformed schemas.

    """
    selected = _load_and_verify_selected_model(registry)
    scorer = (
        predictor if predictor is not None else _load_checkpoint_predictor(selected)
    )

    _raise_gate_problems("G4", _validate_anchor_labels(cfg, registry))

    logger.info("Loading anchor rows and scoring raw probabilities...")
    missions = load_missions(cfg)
    anchor = _load_anchor_frame(registry, missions)
    raw_anchor = _predict_positive_probabilities(
        scorer, anchor["mission_text"].tolist()
    )
    labels_anchor = anchor["human_label"].astype(int).tolist()

    oof_calibrated, calibration_report = crossfit_calibrate(
        raw_anchor.tolist(),
        labels_anchor,
        folds=int(cfg.evaluation.crossfit_folds),
        methods=cfg.evaluation.calibration_methods,
        seed=int(cfg.SEED),
        ece_bins=int(cfg.evaluation.ece_bins),
    )
    threshold_report = pick_threshold(
        oof_calibrated,
        labels_anchor,
        cfg.evaluation.threshold_policy,
        float(cfg.evaluation.precision_floor),
    )

    registry.ensure_dirs()
    _write_anchor_oof_scores(
        registry,
        anchor=anchor,
        raw_probs=raw_anchor,
        calibrated_probs=oof_calibrated,
    )
    calibrator_payload = _calibrator_payload(
        registry,
        calibration_report=calibration_report,
        threshold_report=threshold_report,
        cfg=cfg,
    )
    _write_json(registry.calibrator_path, calibrator_payload)
    logger.info("Wrote calibrator to %s", registry.calibrator_path)

    rule_report = _rule_validation(anchor)
    _write_json(registry.rule_validation, rule_report)
    logger.info("Wrote rule validation to %s", registry.rule_validation)

    _raise_gate_problems("G3", _validate_test_unlock(cfg, registry))
    if registry.test_evaluation.exists():
        raise RuntimeError(
            f"Frozen-test evaluation already exists at {registry.test_evaluation}; "
            "delete it explicitly to re-run."
        )

    test = _read_frozen_test_labels(registry, missions)

    logger.info("Scoring frozen test split (%d rows)...", len(test))
    raw_test = _predict_positive_probabilities(scorer, test["mission_text"].tolist())
    method = cast(CalibrationMethod, calibrator_payload["method"])
    params = cast(Mapping[str, float], calibrator_payload["params"])
    calibrated_test = np.asarray(
        apply_calibration(raw_test.tolist(), method, params),
        dtype=float,
    )
    threshold = float(calibrator_payload["threshold"])
    y_true = test["human_label"].astype(int).to_numpy()
    y_pred = (calibrated_test >= threshold).astype(int)

    metric_bundle = metrics.compute_metric_bundle(
        y_true,
        y_pred,
        y_score=calibrated_test,
        minority_class=1,
        seed=int(cfg.SEED),
        n_resamples=int(cfg.evaluation.bootstrap_resamples),
    )
    anchor_calibration = calibration_metrics(
        labels_anchor,
        oof_calibrated,
        bins=int(cfg.evaluation.ece_bins),
    )
    report = _test_report(
        cfg,
        registry,
        selected=selected,
        metric_bundle=metric_bundle,
        test=test,
        y_pred=y_pred,
        y_prob=calibrated_test,
        anchor_calibration=anchor_calibration,
        calibrator_payload=calibrator_payload,
    )
    _write_json(registry.test_evaluation, report)
    logger.info("Wrote frozen-test evaluation to %s", registry.test_evaluation)

    verdict = cast(dict[str, Any], report["acceptance"])
    if not verdict["passed"]:
        raise RuntimeError(_acceptance_failure_message(verdict))


def _load_and_verify_selected_model(registry: "PathRegistry") -> dict[str, Any]:
    """Load ``selected_model.json`` and verify its checkpoint SHA-256."""
    path = registry.selected_model
    if not path.exists():
        raise RuntimeError(
            f"Selected model artifact not found at {path}. Run stage 06, copy the "
            "selected_model_skeleton to selected_model.json after human review, "
            "then re-run stage 07."
        )
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object.")
    selected = dict(raw)
    relpath = str(selected.get("checkpoint_relpath", "")).strip()
    expected_sha = str(selected.get("checkpoint_sha256", "")).strip()
    if not relpath or not expected_sha or expected_sha.startswith("TODO_"):
        raise RuntimeError(
            f"{path} must contain reviewed checkpoint_relpath and "
            "checkpoint_sha256 values from a completed final stage-06 run."
        )
    checkpoint_path = _checkpoint_path(registry.models_dir, relpath)
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Selected checkpoint file not found at {checkpoint_path}. Restore the "
            "checkpoint under models_dir or update selected_model.json after "
            "human review."
        )
    actual_sha = _sha256_file(checkpoint_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Selected checkpoint SHA-256 mismatch for {checkpoint_path}: expected "
            f"{expected_sha}, got {actual_sha}. Recreate selected_model.json from "
            "the reviewed stage-06 artifact before unlocking test evaluation."
        )
    selected["checkpoint_path"] = str(checkpoint_path)
    return selected


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


def _load_checkpoint_predictor(selected: Mapping[str, Any]) -> Any:
    """Load a Hugging Face sequence-classification checkpoint lazily."""
    checkpoint_path = Path(str(selected["checkpoint_path"]))
    model_dir = checkpoint_path.parent
    tokenizer_id = str(selected.get("tokenizer_id") or model_dir)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised only in real runs
        raise RuntimeError(
            "Default stage-07 predictor loading requires torch and transformers. "
            "Install runtime dependencies or pass an injected predictor with "
            "predict_proba(texts)."
        ) from exc

    tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_id)
    model: Any = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    class _HFPredictor:
        def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
            encoded = cast(Any, tokenizer)(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            return np.asarray(probs, dtype=float)

    return _HFPredictor()


def _raise_gate_problems(name: str, problems: Sequence[str]) -> None:
    if problems:
        joined = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(f"{name} gate failed before stage 07 work:\n{joined}")


def _load_anchor_frame(
    registry: "PathRegistry", missions: pd.DataFrame
) -> pd.DataFrame:
    anchor = pd.read_csv(registry.anchor_coding_template)
    required = {"EIN2", "tier", "human_label"}
    missing = required - set(anchor.columns)
    if missing:
        raise ValueError(
            f"{registry.anchor_coding_template} missing columns {sorted(missing)}."
        )
    anchor = anchor[["EIN2", "tier", "human_label"]].copy()
    anchor["EIN2"] = _normalize_ein2(anchor["EIN2"])
    labels = pd.to_numeric(anchor["human_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Anchor human_label values must be strict 0/1.")
    anchor["human_label"] = labels.astype(int)

    mission_cols = ["EIN2", "mission_text", "ntee_major_group", "data_source"]
    mission_frame = missions[mission_cols].copy()
    mission_frame["EIN2"] = _normalize_ein2(mission_frame["EIN2"])
    anchor = anchor.merge(mission_frame, on="EIN2", how="left")

    manifest_cols = ["EIN2", "sample_prob"]
    if registry.anchor_manifest.exists():
        manifest = pd.read_csv(registry.anchor_manifest)
        if set(manifest_cols).issubset(manifest.columns):
            manifest = manifest[manifest_cols].copy()
            manifest["EIN2"] = _normalize_ein2(manifest["EIN2"])
            anchor = anchor.merge(manifest, on="EIN2", how="left")
    if "sample_prob" not in anchor.columns:
        anchor["sample_prob"] = np.nan

    missing_text = anchor["mission_text"].isna()
    if missing_text.any():
        raise ValueError(
            "Could not join source mission text for "
            f"{int(missing_text.sum())} anchor EIN2(s)."
        )
    return anchor.reset_index(drop=True)


def _normalize_ein2(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


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


def _write_anchor_oof_scores(
    registry: "PathRegistry",
    *,
    anchor: pd.DataFrame,
    raw_probs: np.ndarray,
    calibrated_probs: Sequence[float],
) -> None:
    scores = pd.DataFrame(
        {
            "EIN2": anchor["EIN2"].astype(str),
            "prob_raw": raw_probs.astype(float),
            "prob_calibrated_oof": np.asarray(calibrated_probs, dtype=float),
            "human_label": anchor["human_label"].astype(int),
            "tier": anchor["tier"].astype(str),
            "sample_prob": anchor["sample_prob"].astype(float),
        },
        columns=_ANCHOR_SCORE_COLUMNS,
    )
    registry.anchor_oof_scores.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(registry.anchor_oof_scores, index=False)
    logger.info("Wrote anchor OOF scores to %s", registry.anchor_oof_scores)


def _calibrator_payload(
    registry: "PathRegistry",
    *,
    calibration_report: Mapping[str, Any],
    threshold_report: Mapping[str, Any],
    cfg: "BinaryClassifierConfig",
) -> dict[str, Any]:
    deployed = cast(Mapping[str, Any], calibration_report["deployed"])
    return {
        "method": deployed["method"],
        "params": deployed["params"],
        "threshold": threshold_report["threshold"],
        "threshold_policy": threshold_report["policy"],
        "precision_floor": float(cfg.evaluation.precision_floor),
        "max_f1_threshold": threshold_report["max_f1_threshold"],
        "fitted_on": "anchor",
        "crossfit_folds": int(calibration_report["crossfit_folds"]),
        "anchor_oof_scores_path": str(registry.anchor_oof_scores),
    }


def _rule_validation(anchor: pd.DataFrame) -> dict[str, Any]:
    low = anchor.loc[anchor["tier"].astype(str).str.upper() == "LOW"].copy()
    rule_labels = [apply_rule_label(text) for text in low["mission_text"].tolist()]
    low["rule_label"] = rule_labels
    covered = low.dropna(subset=["rule_label"]).copy()
    if covered.empty:
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    else:
        y_true = covered["human_label"].astype(int).to_numpy()
        y_pred = covered["rule_label"].astype(int).to_numpy()
        counts = {
            "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
            "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
            "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
            "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        }
    return {
        "scope": "anchor_low_cells",
        "counts": {
            "n_low": int(len(low)),
            "n_rule_applied": int(len(covered)),
            "n_abstain": int(len(low) - len(covered)),
            **counts,
        },
        "metrics": {
            "sensitivity": _rate_with_ci(counts["tp"], counts["tp"] + counts["fn"]),
            "specificity": _rate_with_ci(counts["tn"], counts["tn"] + counts["fp"]),
            "precision": _rate_with_ci(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _rate_with_ci(counts["tp"], counts["tp"] + counts["fn"]),
        },
    }


def _rate_with_ci(successes: int, denominator: int) -> dict[str, Any]:
    value = None if denominator == 0 else successes / denominator
    ci = (
        _wilson_ci(successes, denominator)
        if denominator
        else {"lower": None, "upper": None}
    )
    return {
        "value": value,
        "ci": ci,
        "numerator": successes,
        "denominator": denominator,
    }


def _wilson_ci(
    successes: int, denominator: int, z: float = 1.959963984540054
) -> dict[str, float]:
    p_hat = successes / denominator
    denom = 1.0 + z**2 / denominator
    centre = p_hat + z**2 / (2.0 * denominator)
    margin = z * np.sqrt(
        (p_hat * (1.0 - p_hat) + z**2 / (4.0 * denominator)) / denominator
    )
    return {
        "lower": float(max(0.0, (centre - margin) / denom)),
        "upper": float(min(1.0, (centre + margin) / denom)),
    }


def _read_frozen_test_labels(
    registry: "PathRegistry",
    missions: pd.DataFrame,
) -> pd.DataFrame:
    """Read the frozen test split; this is the only stage-07 test reader."""
    path = registry.gold_coding_template
    df = pd.read_csv(path)
    required = {"EIN2", "split", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}.")
    split_values = df["split"].astype(str).str.strip()
    test = df.loc[split_values == "test", ["EIN2", "human_label"]].copy()
    if test.empty:
        raise ValueError(f"No rows for split 'test' in {path}.")
    test["EIN2"] = _normalize_ein2(test["EIN2"])
    labels = pd.to_numeric(test["human_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Frozen test human_label values must be strict 0/1.")
    test["human_label"] = labels.astype(int)

    mission_frame = missions.copy()
    mission_frame["EIN2"] = _normalize_ein2(mission_frame["EIN2"])
    joined = test.merge(mission_frame, on="EIN2", how="left")
    missing_text = joined["mission_text"].isna()
    if missing_text.any():
        raise ValueError(
            "Could not join source mission text for "
            f"{int(missing_text.sum())} frozen-test EIN2(s)."
        )
    return joined.reset_index(drop=True)


def _test_report(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    *,
    selected: Mapping[str, Any],
    metric_bundle: Mapping[str, Any],
    test: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    anchor_calibration: Mapping[str, Any],
    calibrator_payload: Mapping[str, Any],
) -> dict[str, Any]:
    report_df = test.rename(columns={"mission_text": "text"}).copy()
    thresholds = [float(x) for x in np.linspace(0.05, 0.95, 19)]
    return {
        "metric_bundle": dict(metric_bundle),
        "subgroups": subgroup_report(
            report_df,
            test["human_label"].astype(int).tolist(),
            y_pred.astype(int).tolist(),
            y_prob.astype(float).tolist(),
            by=["ntee_major_group", "data_source"],
            length_bins=cfg.evaluation.length_bins,
            min_n=1,
        ),
        "decision_curve": net_benefit(
            test["human_label"].astype(int).tolist(),
            y_prob.astype(float).tolist(),
            thresholds,
        ),
        "calibration_on_anchor_oof": dict(anchor_calibration),
        "acceptance": _acceptance_verdict(cfg, metric_bundle, anchor_calibration),
        "metadata": {
            "model_id": selected.get("encoder_id") or selected.get("tokenizer_id"),
            "checkpoint_relpath": selected.get("checkpoint_relpath"),
            "checkpoint_sha256": selected.get("checkpoint_sha256"),
            "calibrator_path": str(registry.calibrator_path),
            "calibrator": dict(calibrator_payload),
            "config_hash": _config_hash(cfg),
            "git_sha": _git_sha(),
            "date": datetime.now(UTC).isoformat(),
        },
    }


def _acceptance_verdict(
    cfg: "BinaryClassifierConfig",
    metric_bundle: Mapping[str, Any],
    anchor_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = cfg.evaluation.acceptance
    pr_auc = metric_bundle.get("pr_auc")
    ci = cast(Mapping[str, Mapping[str, float]], metric_bundle["bootstrap_ci"])
    minority_f1_lower = ci["minority_f1"]["lower"]
    ece = float(anchor_calibration["ece"])
    checks = {
        "min_pr_auc": {
            "observed": pr_auc,
            "threshold": criteria.min_pr_auc,
            "passed": pr_auc is not None and float(pr_auc) >= criteria.min_pr_auc,
        },
        "min_minority_f1_ci_lower": {
            "observed": minority_f1_lower,
            "threshold": criteria.min_minority_f1_ci_lower,
            "passed": minority_f1_lower >= criteria.min_minority_f1_ci_lower,
        },
        "max_ece": {
            "observed": ece,
            "threshold": criteria.max_ece,
            "passed": ece <= criteria.max_ece,
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def _acceptance_failure_message(verdict: Mapping[str, Any]) -> str:
    checks = cast(Mapping[str, Mapping[str, Any]], verdict["checks"])
    failed = [
        f"{name}: observed={check['observed']} threshold={check['threshold']}"
        for name, check in checks.items()
        if not check["passed"]
    ]
    return (
        "EVALUATION GATE FAILED: "
        + "; ".join(failed)
        + ". Frozen-test report was written for audit; do not re-run without "
        "explicitly deleting it."
    )


def _config_hash(cfg: "BinaryClassifierConfig") -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
