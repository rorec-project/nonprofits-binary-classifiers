"""Quality control and agreement metrics for the annotation pipeline.

Provides LLM-vs-human agreement scoring and the ≥85% validation gate,
plus a full sklearn metric bundle (confusion matrix, minority-class
precision/recall/F1, MCC, balanced accuracy, Cohen's κ, PR-AUC when
confidence scores exist, and bootstrap CIs). Also handles versioning and
freezing of the final label artifact.

.. note::
    Cohen's κ is sensitive to class imbalance and prevalence (the
    distribution of classes in the data). It should be interpreted
    alongside the other metrics in the bundle, not as a standalone
    gate.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from binary_classifier.annotate.aggregate import aggregate_labels
from binary_classifier.annotate.schema import AnnotationStore

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def run_quality_check(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    store_path: Path | None = None,
    human_validation_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Aggregate labels and run the blocking QC gate.

    Steps:
        1. Load the long/tidy annotation store.
        2. Aggregate per-EIN2 labels by majority vote (default).
        3. Compute LLM-vs-human agreement on the coded validation split.
        4. Compute the full sklearn metric bundle (confusion matrix,
           minority-class precision/recall/F1, MCC, balanced accuracy,
           Cohen's κ, PR-AUC when confidence scores exist, and bootstrap
           95% CIs for accuracy and minority F1).
        5. Freeze the silver labels **only if** agreement meets the configured
           threshold; otherwise raise (write nothing).

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved manifest paths.
        store_path: Path to the long/tidy store. Defaults to
            ``registry.annotation_store``.
        human_validation_path: Coded human labels. Defaults to the gold coding
            template; the ``validation`` split (``EIN2, human_label``) is used.
        output_path: Where to write the frozen silver labels.
            Defaults to ``train_test_datasets/silver_labels.csv``.

    Returns:
        Dict with ``agreement``, ``n_total``, ``n_abstain``, ``n_valid``,
        ``confusion_matrix``, ``minority_class``, ``precision``, ``recall``,
        ``f1``, ``mcc``, ``balanced_accuracy``, ``cohens_kappa``,
        ``pr_auc`` (or ``None``), ``bootstrap_ci``, and ``frozen_path``.

    Raises:
        ValueError: If the store is empty, no validation labels overlap, or the
            agreement gate fails (below the configured threshold).
        FileNotFoundError: If no coded validation labels are available.
    """
    if store_path is None:
        store_path = registry.annotation_store
    if output_path is None:
        output_path = registry.train_test_dir / "silver_labels.csv"
    if human_validation_path is None:
        human_validation_path = registry.gold_coding_template

    store = AnnotationStore(store_path)
    df = store.to_frame()

    if df.empty:
        raise ValueError(
            f"Annotation store at {store_path} is empty. Run stage 03 first."
        )

    # Aggregate
    aggregated = aggregate_labels(df, method="majority")
    n_total = len(aggregated)
    n_abstain = int(aggregated["silver_label"].isna().sum())
    logger.info(
        "Aggregated %d EIN2s; %d abstain/tie (%.1f%%)",
        n_total,
        n_abstain,
        100 * n_abstain / n_total,
    )

    # Validation agreement (blocking gate).
    human_df = _load_validation_labels(human_validation_path)
    merged = aggregated.merge(
        human_df[["EIN2", "human_label"]],
        on="EIN2",
        how="inner",
    )
    valid = merged.dropna(subset=["silver_label", "human_label"])
    if valid.empty:
        raise ValueError(
            "No overlapping validation labels to compare; cannot run the "
            "agreement gate. Check that the silver run and the coded "
            "validation split share EIN2s."
        )

    agree = int((valid["silver_label"] == valid["human_label"]).sum())
    agreement = agree / len(valid)
    threshold = cfg.qc.agreement_threshold

    # Compute full metric bundle
    metrics = _compute_metrics(valid)

    logger.info(
        "Validation agreement: %.1f%% (%d/%d); threshold %.0f%%",
        agreement * 100,
        agree,
        len(valid),
        threshold * 100,
    )
    logger.info(
        "Metrics — minority class %s: P=%.3f R=%.3f F1=%.3f | "
        "MCC=%.3f | Balanced Acc=%.3f | Cohen's κ=%.3f%s",
        metrics["minority_class"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["mcc"],
        metrics["balanced_accuracy"],
        metrics["cohens_kappa"],
        (
            f" | PR-AUC={metrics['pr_auc']:.3f}"
            if metrics["pr_auc"] is not None
            else ""
        ),
    )
    logger.info(
        "Bootstrap 95%% CI — accuracy: [%.3f, %.3f]; minority F1: [%.3f, %.3f]",
        metrics["bootstrap_ci"]["accuracy"]["lower"],
        metrics["bootstrap_ci"]["accuracy"]["upper"],
        metrics["bootstrap_ci"]["minority_f1"]["lower"],
        metrics["bootstrap_ci"]["minority_f1"]["upper"],
    )

    if agreement < threshold:
        msg = (
            f"AGREEMENT GATE FAILED: {agreement:.1%} < {threshold:.0%}. "
            f"Revise prompts and re-label; nothing was frozen.\n"
            f"Metrics: minority_class={metrics['minority_class']}, "
            f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, MCC={metrics['mcc']:.3f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.3f}, "
            f"cohens_kappa={metrics['cohens_kappa']:.3f}"
        )
        if metrics["pr_auc"] is not None:
            msg += f", pr_auc={metrics['pr_auc']:.3f}"
        raise ValueError(msg)

    # Freeze (gate passed).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)
    logger.info("Frozen silver labels written to %s", output_path)

    return {
        "agreement": agreement,
        "n_total": n_total,
        "n_abstain": n_abstain,
        "n_valid": len(valid),
        "confusion_matrix": metrics["confusion_matrix"],
        "minority_class": metrics["minority_class"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mcc": metrics["mcc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "cohens_kappa": metrics["cohens_kappa"],
        "pr_auc": metrics["pr_auc"],
        "bootstrap_ci": metrics["bootstrap_ci"],
        "frozen_path": str(output_path),
    }


def _compute_metrics(valid: pd.DataFrame) -> dict:
    """Compute the full sklearn metric bundle on the validation overlap.

    Args:
        valid: DataFrame with ``silver_label`` and ``human_label`` columns.

    Returns:
        Dict with all computed metrics.
    """
    y_true = valid["human_label"].astype(int).values
    y_pred = valid["silver_label"].astype(int).values

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Determine minority class
    counts = np.bincount(y_true)
    minority_class = int(np.argmin(counts))

    # Precision, recall, F1 for both classes; report minority explicitly
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    precision = float(precisions[minority_class])
    recall = float(recalls[minority_class])
    f1 = float(f1s[minority_class])

    mcc = float(matthews_corrcoef(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # PR-AUC when confidence scores exist
    pr_auc = None
    if "silver_confidence" in valid.columns and valid["silver_confidence"].notna().any():
        scores = valid["silver_confidence"].astype(float).values
        mask = ~np.isnan(scores)
        if mask.any():
            pr_auc = float(average_precision_score(y_true[mask], scores[mask]))

    # Bootstrap CI
    bootstrap_ci = _bootstrap_ci(y_true, y_pred, minority_class)

    return {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "minority_class": minority_class,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": balanced_acc,
        "cohens_kappa": kappa,
        "pr_auc": pr_auc,
        "bootstrap_ci": bootstrap_ci,
    }


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minority_class: int,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for accuracy and minority-class F1.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        minority_class: The class to report F1 for.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g. 0.95).

    Returns:
        Dict with ``accuracy`` and ``minority_f1`` each containing
        ``lower`` and ``upper``.
    """
    rng = np.random.default_rng(seed=42)
    n = len(y_true)
    accs = np.empty(n_resamples, dtype=float)
    f1s = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        accs[i] = float(np.mean(y_true[idx] == y_pred[idx]))
        f1s[i] = f1_score(
            y_true[idx], y_pred[idx], pos_label=minority_class, zero_division=0
        )

    alpha = 1 - confidence_level
    lower_p = alpha / 2 * 100
    upper_p = (1 - alpha / 2) * 100

    return {
        "accuracy": {
            "lower": float(np.percentile(accs, lower_p)),
            "upper": float(np.percentile(accs, upper_p)),
        },
        "minority_f1": {
            "lower": float(np.percentile(f1s, lower_p)),
            "upper": float(np.percentile(f1s, upper_p)),
        },
    }


def _load_validation_labels(human_validation_path: Path) -> pd.DataFrame:
    """Load coded validation labels from the gold coding template.

    Args:
        human_validation_path: Path to ``gold_to_code.csv``.

    Returns:
        DataFrame with ``EIN2`` and ``human_label`` for the validation split.

    Raises:
        FileNotFoundError: If the template is missing.
        ValueError: If columns are missing or no validation labels are coded.

    """
    if not human_validation_path.exists():
        raise FileNotFoundError(
            f"No human coding template at {human_validation_path}. Code "
            f"human_label (0/1) for the validation split before stage 04."
        )
    df = pd.read_csv(human_validation_path)
    required = {"EIN2", "split", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{human_validation_path} missing columns: {sorted(missing)}.")
    sub = df[df["split"] == "validation"].dropna(subset=["human_label"])
    if sub.empty:
        raise ValueError(
            f"No coded validation labels in {human_validation_path}. Fill "
            f"human_label (0/1) for the validation split before stage 04."
        )
    return sub
