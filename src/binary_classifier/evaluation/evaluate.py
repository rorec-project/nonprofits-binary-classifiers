"""Stage 07 frozen-test evaluation entrypoint.

This module implements the final evaluation gate for the binary classifier
pipeline. It loads a human-selected checkpoint, calibrates probabilities on
the anchor set via cross-fitted Platt + temperature scaling, validates the
LOW-tier rule layer, and produces a single one-shot frozen-test report that
includes minority-class metrics, subgroup disparities, and calibration
diagnostics. The acceptance gate is ``max_ece``-only for this pass; Brier and
log-loss are reserved for future work.

Data provenance
    - Anchor labels: ``anchor_coding_template`` (human-coded 0/1, G4 gate).
    - Frozen test: ``gold_coding_template`` split == ``test`` (human-coded 0/1,
      G3 gate).
    - Model checkpoint: ``selected_model.json`` with SHA-256 verification.

Methodology and citations
    - Precision-Recall AUC and minority-class F1 are the primary headline
      metrics (Davis & Goadrich, 2006, DOI: 10.1145/1143844.1143874; Saito &
      Rehmsmeier, 2015, DOI: 10.1371/journal.pone.0118432).
    - The Matthews Correlation Coefficient (MCC) is reported as a balanced
      summary statistic (Chicco & Jurman, 2020, DOI:
      10.1186/s12864-019-6413-7).
    - Threshold selection and expected-loss framing follow Hernández-Orallo,
      Flach & Ferri (2012, https://www.jmlr.org/papers/v13/hernandez-orallo12a.html).
    - Calibration uses Platt scaling + temperature scaling (see
      ``calibration.py`` for full citation list).

Note: Vickers-Elkin decision-curve analysis was intentionally removed because
it is orthogonal to a prevalence-estimation study; the downstream deliverable is
a calibrated, uncertainty-quantified population share (PPI++), not a clinical
treat-vs-abstain decision.
"""

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
from sklearn.metrics import precision_recall_curve, roc_curve

from binary_classifier import metrics
from binary_classifier.data.load import load_missions
from binary_classifier.data.quality import apply_rule_label
from binary_classifier.evaluation.calibration import (
    CalibrationMethod,
    apply_calibration,
    calibration_metrics,
    crossfit_calibrate,
)
from binary_classifier.evaluation.base_rate import base_rate_report
from binary_classifier.evaluation.subgroups import subgroup_report
from binary_classifier.evaluation.thresholds import pick_threshold
from binary_classifier.inference.router import route
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
    "decision_source",
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
    # Verify the selected checkpoint via SHA-256 to prevent accidental drift
    # between the human-reviewed stage-06 choice and the model loaded here.
    selected = _load_and_verify_selected_model(registry)

    # G4 anchor-labels gate: prevalence estimates on LOW-quality rows cannot be
    # validated without fully coded anchor labels, so we fail early.
    _raise_gate_problems("G4", _validate_anchor_labels(cfg, registry))

    # G3 test-unlock gate: the frozen test must only be scored once per
    # checkpoint to prevent leakage during model iteration. One-shot refusal
    # must happen before any calibration artifact is refreshed.
    _raise_gate_problems("G3", _validate_test_unlock(cfg, registry))
    if registry.test_evaluation.exists():
        raise RuntimeError(
            f"Frozen-test evaluation already exists at {registry.test_evaluation}; "
            "delete it explicitly to re-run.",
        )

    scorer = (
        predictor if predictor is not None else _load_checkpoint_predictor(selected)
    )

    # Load the anchor set, score raw probabilities, and calibrate OOF.
    # Cross-fitted calibration avoids overfitting the calibration mapping to the
    # same data used to train it, which is critical when the anchor is small.
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
        cfg,
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

    base_rate_payload = base_rate_report(
        pd.read_parquet(registry.anchor_oof_scores),
        operating_threshold=float(calibrator_payload["threshold"]),
        max_f1_threshold=float(calibrator_payload["max_f1_threshold"]),
        target=float(cfg.evaluation.base_rate_precision_target),
        population_base_rate=cfg.evaluation.population_base_rate,
        seed=int(cfg.SEED),
        n_resamples=int(cfg.evaluation.bootstrap_resamples),
    )
    _write_json(registry.base_rate_precision, base_rate_payload)
    logger.info("Wrote base-rate precision report to %s", registry.base_rate_precision)

    # Validate the LOW-tier rule layer on the anchor set. Rules are the only
    # source of labels for LOW-quality rows, so their sensitivity/specificity
    # must be quantified before prevalence estimation can use them.
    rule_report = _rule_validation(anchor)
    _write_json(registry.rule_validation, rule_report)
    logger.info("Wrote rule validation to %s", registry.rule_validation)

    test = _read_frozen_test_labels(registry, missions)

    # Score the frozen test, apply the fitted calibrator, threshold, and
    # compute the full metric bundle. All metrics are minority-class-oriented
    # because the positive class (religious) is the rare outcome of interest.
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

    # Primary metrics follow the imbalanced-text evaluation literature: PR-AUC
    # and minority F1 as headline numbers, MCC as a balanced summary, and
    # bootstrap CIs for uncertainty quantification (Davis & Goadrich 2006;
    # Saito & Rehmsmeier 2015; Chicco & Jurman 2020).
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
        base_rate_payload=base_rate_payload,
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
    cfg: "BinaryClassifierConfig",
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
            "decision_source": [
                route(text, tier, cfg)[0]
                for text, tier in zip(
                    anchor["mission_text"],
                    anchor["tier"],
                    strict=True,
                )
            ],
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
        "achieved_precision": threshold_report["achieved_precision"],
        "achieved_recall": threshold_report["achieved_recall"],
        "max_f1_threshold": threshold_report["max_f1_threshold"],
        "pr_curve_points": threshold_report["pr_curve_points"],
        "fitted_on": "anchor",
        "crossfit_folds": int(calibration_report["crossfit_folds"]),
        "anchor_oof_scores_path": str(registry.anchor_oof_scores),
    }


# The LOW-tier rule layer is evaluated separately because rules are the only
# labels available for LOW-quality rows. Reporting sensitivity, specificity,
# and precision on the anchor lets the prevalence stage decide whether the
# rule layer is accurate enough to include LOW rows in the population estimate.
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
    successes: int,
    denominator: int,
    z: float = 1.959963984540054,
) -> dict[str, float]:
    p_hat = successes / denominator
    denom = 1.0 + z**2 / denominator
    centre = p_hat + z**2 / (2.0 * denominator)
    margin = z * np.sqrt(
        (p_hat * (1.0 - p_hat) + z**2 / (4.0 * denominator)) / denominator,
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
            f"{int(missing_text.sum())} frozen-test EIN2(s).",
        )
    return joined.reset_index(drop=True)


# Assemble the one-shot frozen-test report. Decision-curve analysis was
# intentionally omitted because the downstream deliverable is a calibrated
# population-prevalence estimate (PPI++), not a clinical treat-vs-abstain
# decision.
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
    base_rate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the frozen-test report (metrics, subgroups, acceptance verdict).

    Caveat: these metrics are computed on the boundary- and positive-enriched
    gold ``test`` split, a deliberately non-representative slice (~50 % positive
    vs. the low population base rate). Because precision, PR-AUC, and F1 are not
    prevalence-invariant, they characterise performance on hard/enriched cases,
    not the deployment distribution; population-level performance is recovered
    from the design-weighted anchor sample via PPI, not from this report.
    """
    report_df = test.rename(columns={"mission_text": "text"}).copy()
    y_true = test["human_label"].astype(int).to_numpy()
    thresholds = _report_thresholds(calibrator_payload, base_rate_payload)
    return {
        "metric_bundle": dict(metric_bundle),
        "test_scores": _test_scores(test, y_true, y_prob),
        "confusion_matrices": _threshold_confusion_matrices(
            y_true,
            y_prob,
            thresholds,
        ),
        "pr_curve_points": _sklearn_pr_curve_points(y_true, y_prob),
        "roc_curve_points": _sklearn_roc_curve_points(y_true, y_prob),
        "subgroups": subgroup_report(
            report_df,
            test["human_label"].astype(int).tolist(),
            y_pred.astype(int).tolist(),
            y_prob.astype(float).tolist(),
            by=["ntee_major_group", "data_source"],
            length_bins=cfg.evaluation.length_bins,
            min_n=1,
        ),
        "calibration_on_anchor_oof": dict(anchor_calibration),
        "acceptance": _acceptance_verdict(cfg, metric_bundle, anchor_calibration),
        "metadata": {
            "model_id": selected.get("encoder_id") or selected.get("tokenizer_id"),
            "checkpoint_relpath": selected.get("checkpoint_relpath"),
            "checkpoint_sha256": selected.get("checkpoint_sha256"),
            "calibrator_path": str(registry.calibrator_path),
            "calibrator": dict(calibrator_payload),
            "base_rate_precision_path": str(registry.base_rate_precision),
            "base_rate_precision": dict(base_rate_payload),
            "config_hash": _config_hash(cfg),
            "git_sha": _git_sha(),
            "date": datetime.now(UTC).isoformat(),
        },
    }


def _report_thresholds(
    calibrator_payload: Mapping[str, Any],
    base_rate_payload: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "operating": float(calibrator_payload["threshold"]),
        "max_f1": float(calibrator_payload["max_f1_threshold"]),
        "base_rate": float(base_rate_payload["threshold"]),
    }


def _test_scores(
    test: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    return {
        str(ein2): {"y_true": int(label), "prob_calibrated": float(prob)}
        for ein2, label, prob in zip(test["EIN2"], y_true, y_prob, strict=True)
    }


def _threshold_confusion_matrices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    matrices: dict[str, dict[str, Any]] = {}
    for name, threshold in thresholds.items():
        y_pred = (y_prob >= threshold).astype(int)
        matrices[name] = {
            "threshold": float(threshold),
            "confusion_matrix": _confusion_matrix_counts(y_true, y_pred),
        }
    return matrices


def _confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def _sklearn_pr_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> list[dict[str, float | None]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    points: list[dict[str, float | None]] = []
    for idx, (p, r) in enumerate(zip(precision, recall, strict=True)):
        threshold = None if idx >= len(thresholds) else float(thresholds[idx])
        points.append(
            {"threshold": threshold, "precision": float(p), "recall": float(r)}
        )
    return points


def _sklearn_roc_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> list[dict[str, float]]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return [
        {"threshold": float(threshold), "fpr": float(fp), "tpr": float(tp)}
        for fp, tp, threshold in zip(fpr, tpr, thresholds, strict=True)
    ]


# Acceptance gate is max_ece-only for this pass. Brier and log-loss are
# reserved for future calibration work; the research deliverable is a
# calibrated prevalence estimate, so ECE is the primary diagnostic.
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
