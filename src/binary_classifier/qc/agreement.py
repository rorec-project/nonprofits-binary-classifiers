"""Quality control and agreement metrics for the annotation pipeline.

Provides LLM-vs-human agreement scoring and the validation freeze gate, plus a
full sklearn metric bundle (confusion matrix, minority-class precision/recall/F1,
MCC, balanced accuracy, Cohen's κ, Krippendorff alpha, PR-AUC when confidence scores
exist, and bootstrap CIs). Also handles versioning and freezing of the final
label artifact.

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
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from binary_classifier.annotate.aggregate import aggregate_labels
from binary_classifier.annotate.schema import AnnotationStore
from binary_classifier.qc.evidence import (
    abstain_fabricated_positives,
    verify_evidence_spans,
)

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
        2. Optionally verify evidence spans and abstain fabricated positive
           labels before voting. This is config-gated so default runs keep the
           current data dependency surface.
        3. Aggregate per-EIN2 labels by majority vote (default).
        4. Compute LLM-vs-human agreement on the coded validation split.
        5. Compute the full sklearn metric bundle (confusion matrix,
            minority-class precision/recall/F1, MCC, balanced accuracy,
            Cohen's κ, Krippendorff alpha, PR-AUC when confidence scores exist,
            and bootstrap 95% CIs for accuracy and minority F1).
        6. Freeze the silver labels **only if** Cohen's κ and the lower bound
            of the minority-F1 bootstrap CI meet their configured thresholds.
            Raw agreement is retained as a reported diagnostic, not the sole
            gate. Gold-manifest rows are excluded before writing so the
            training artifact cannot leak human-held-out examples.

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved manifest paths.
        store_path: Path to the long/tidy store. Defaults to
            ``registry.annotation_store``.
        human_validation_path: Coded human labels. Defaults to the gold coding
            template; the ``validation`` split (``EIN2, human_label``) is used.
        output_path: Where to write the frozen silver labels.
            Defaults to ``processed_dir / "silver_labels.csv"``.

    Returns:
        Dict with ``agreement``, ``n_total``, ``n_abstain``, ``n_valid``,
        ``confusion_matrix``, ``minority_class``, ``precision``, ``recall``,
        ``f1``, ``mcc``, ``balanced_accuracy``, ``cohens_kappa``,
        ``krippendorff_alpha``, ``pr_auc`` (or ``None``), ``bootstrap_ci``,
        gate thresholds, and ``frozen_path``.

    Raises:
        ValueError: If the store is empty, no validation labels overlap, or the
            agreement gate fails the configured chance-corrected κ threshold
            or minority-F1 CI lower-bound floor.
        FileNotFoundError: If no coded validation labels are available.

    """
    if store_path is None:
        store_path = registry.annotation_store
    if output_path is None:
        output_path = registry.processed_dir / "silver_labels.csv"
    if human_validation_path is None:
        human_validation_path = registry.gold_coding_template

    store = AnnotationStore(store_path)
    df = store.to_frame()

    if df.empty:
        raise ValueError(
            f"Annotation store at {store_path} is empty. Run stage 03 first.",
        )

    if cfg.qc.abstain_on_fabricated_positive:
        df = _abstain_fabricated_positive_labels(df, registry)

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
            "validation split share EIN2s.",
        )

    agree = int((valid["silver_label"] == valid["human_label"]).sum())
    agreement = agree / len(valid)
    raw_agreement_threshold = cfg.qc.agreement_threshold
    kappa_threshold = cfg.qc.kappa_threshold
    f1_ci_floor = cfg.qc.f1_ci_floor

    # Compute full metric bundle
    metrics = _compute_metrics(valid, seed=cfg.SEED)
    minority_f1_ci = metrics["bootstrap_ci"]["minority_f1"]
    minority_f1_ci_lower = minority_f1_ci["lower"]

    logger.info(
        "Validation raw agreement (reported only): %.1f%% (%d/%d); "
        "legacy benchmark %.0f%%",
        agreement * 100,
        agree,
        len(valid),
        raw_agreement_threshold * 100,
    )
    logger.info(
        "Metrics — minority class %s: P=%.3f R=%.3f F1=%.3f | "
        "MCC=%.3f | Balanced Acc=%.3f | Cohen's κ=%.3f | "
        "Krippendorff alpha=%.3f%s",
        metrics["minority_class"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["mcc"],
        metrics["balanced_accuracy"],
        metrics["cohens_kappa"],
        metrics["krippendorff_alpha"],
        (f" | PR-AUC={metrics['pr_auc']:.3f}" if metrics["pr_auc"] is not None else ""),
    )
    logger.info(
        "Bootstrap 95%% CI — accuracy: [%.3f, %.3f]; minority F1: [%.3f, %.3f]",
        metrics["bootstrap_ci"]["accuracy"]["lower"],
        metrics["bootstrap_ci"]["accuracy"]["upper"],
        metrics["bootstrap_ci"]["minority_f1"]["lower"],
        metrics["bootstrap_ci"]["minority_f1"]["upper"],
    )

    # κ ≥ 0.70 preserves the old ≈85% balanced-validation operating point as
    # chance-corrected agreement (Landis & Koch 1977). The minority-F1 CI floor
    # is the deliberate variance-aware lever for modelXprompt LLM annotation
    # risk (SILICON (arXiv:2412.14461); Variance-Aware protocol
    # (arXiv:2601.02370); "LLM Hacking" modelXprompt variance
    # (arXiv:2509.08825)).
    kappa_pass = bool(
        np.isfinite(metrics["cohens_kappa"])
        and metrics["cohens_kappa"] >= kappa_threshold,
    )
    f1_ci_pass = bool(
        np.isfinite(minority_f1_ci_lower) and minority_f1_ci_lower >= f1_ci_floor,
    )
    if not (kappa_pass and f1_ci_pass):
        msg = (
            "AGREEMENT GATE FAILED: "
            f"cohens_kappa={metrics['cohens_kappa']:.3f} "
            f"(threshold={kappa_threshold:.3f}), "
            f"krippendorff_alpha={metrics['krippendorff_alpha']:.3f}, "
            f"minority_f1={metrics['f1']:.3f}, "
            f"minority_f1_ci_lower={minority_f1_ci_lower:.3f} "
            f"(floor={f1_ci_floor:.3f}); "
            f"raw_agreement={agreement:.1%} "
            f"(reported only; legacy benchmark={raw_agreement_threshold:.0%}). "
            f"Revise prompts and re-label; nothing was frozen.\n"
            f"Metrics: minority_class={metrics['minority_class']}, "
            f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, MCC={metrics['mcc']:.3f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.3f}, "
            f"cohens_kappa={metrics['cohens_kappa']:.3f}, "
            f"krippendorff_alpha={metrics['krippendorff_alpha']:.3f}, "
            f"minority_f1_ci=[{minority_f1_ci['lower']:.3f}, "
            f"{minority_f1_ci['upper']:.3f}]"
        )
        if metrics["pr_auc"] is not None:
            msg += f", pr_auc={metrics['pr_auc']:.3f}"
        raise ValueError(msg)

    # Freeze (gate passed).
    # The validation gate above needs gold rows in ``aggregated`` after T2's
    # silver union gold annotation. The frozen stage-05 training artifact must
    # remove every gold-manifest EIN2 only after that gate has passed.
    frozen = _exclude_gold_manifest_ein2s(aggregated, registry.gold_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_csv(output_path, index=False)
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
        "krippendorff_alpha": metrics["krippendorff_alpha"],
        "kappa_threshold": kappa_threshold,
        "f1_ci_floor": f1_ci_floor,
        "pr_auc": metrics["pr_auc"],
        "bootstrap_ci": metrics["bootstrap_ci"],
        "frozen_path": str(output_path),
    }


def _abstain_fabricated_positive_labels(
    store_df: pd.DataFrame,
    registry: "PathRegistry",
) -> pd.DataFrame:
    """Verify evidence spans and abstain fabricated positive votes.

    The LLM evidence spans are a hallucination surface: a positive label that
    cites text absent from the source record should not contribute a positive
    vote to the weak-supervision aggregate. This guard is deliberately called
    before aggregation so fabricated positives become abstentions rather than
    silver labels.

    Args:
        store_df: Long/tidy annotation-store frame.
        registry: Path registry used to locate the upstream text parquet.

    Returns:
        A copy of ``store_df`` with fabricated positive rows abstained, or the
        original frame when verification cannot run because the source parquet
        is absent.

    """
    try:
        ev = verify_evidence_spans(registry, store_df)
    except FileNotFoundError as exc:
        logger.warning("Skipping evidence verification: %s", exc)
        return store_df

    logger.info(
        "Evidence verification: %d/%d spans verified, %d fabricated (%.1f%%)",
        ev["verified_spans"],
        ev["total_spans"],
        ev["fabricated_spans"],
        ev["fabrication_rate"] * 100,
    )
    if not ev["fabricated_records"]:
        return store_df

    logger.warning(
        "Fabricated spans found for %d record(s)",
        len({(e, s) for e, s, _ in ev["fabricated_records"]}),
    )
    # Hallucination guard: unsupported positive evidence can create false
    # religious labels, so those positive votes abstain before majority voting.
    return abstain_fabricated_positives(store_df, ev["fabricated_records"])


def _exclude_gold_manifest_ein2s(
    aggregated: pd.DataFrame,
    gold_manifest_path: Path,
) -> pd.DataFrame:
    """Remove all gold-manifest EIN2s from the frozen training artifact.

    Stage 03 intentionally annotates silver plus gold so the validation gate can
    compare LLM labels with human labels. Stage 05 consumes ``silver_labels.csv``
    for training, so prompt-dev, validation, test, and monitor rows must be
    excluded from the written artifact. This is an exclusion guard rather than a
    silver keep-list because silver and gold are independent draws: a held-out
    test row must be dropped even when it also appears in the silver manifest.

    Args:
        aggregated: Per-EIN2 labels produced from the full annotation store.
        gold_manifest_path: Path to the stage-01 gold manifest.

    Returns:
        A copy of ``aggregated`` with all gold-manifest ``EIN2`` values removed.

    Raises:
        FileNotFoundError: If the manifest is missing, because the leak guard
            cannot prove held-out rows were excluded.
        ValueError: If the manifest lacks the required ``EIN2`` column.

    """
    if not gold_manifest_path.exists():
        raise FileNotFoundError(
            f"No gold manifest at {gold_manifest_path}. Run stage 01 first; "
            "cannot freeze silver labels without the gold-row leak guard.",
        )

    gold_df = pd.read_csv(gold_manifest_path)
    if "EIN2" not in gold_df.columns:
        raise ValueError(f"{gold_manifest_path} missing required EIN2 column.")

    gold_ein2s = set(gold_df["EIN2"].dropna().astype(str))
    if not gold_ein2s:
        return aggregated.copy()

    # CSV round-trips can infer identifier dtypes differently across files, so
    # compare normalized strings to avoid retaining a gold row due to dtype drift.
    keep_mask = ~aggregated["EIN2"].astype(str).isin(gold_ein2s)
    dropped = len(aggregated) - int(keep_mask.sum())
    logger.info("Excluded %d gold-manifest EIN2s before freezing", dropped)
    return aggregated.loc[keep_mask].copy()


def _compute_metrics(valid: pd.DataFrame, seed: int) -> dict:
    """Compute the full sklearn metric bundle on the validation overlap.

    The QC gate now uses chance-corrected agreement plus the lower bound of the
    minority-class F1 bootstrap interval, so this bundle keeps the raw agreement
    diagnostics and the gate-driving quantities together for logging and tests.

    Args:
        valid: DataFrame with ``silver_label`` and ``human_label`` columns.

    Returns:
        Dict with all computed metrics.

    """
    y_true = valid["human_label"].astype(int).to_numpy()
    y_pred = valid["silver_label"].astype(int).to_numpy()

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Determine minority class
    counts = np.bincount(y_true)
    minority_class = int(np.argmin(counts))

    # Precision, recall, F1 for both classes; report minority explicitly
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    precision = float(precisions[minority_class])
    recall = float(recalls[minority_class])
    f1 = float(f1s[minority_class])

    mcc = float(matthews_corrcoef(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    alpha = _krippendorff_alpha_nominal_two_raters(y_true, y_pred)

    # PR-AUC when confidence scores exist
    pr_auc = None
    if (
        "silver_confidence" in valid.columns
        and valid["silver_confidence"].notna().any()
    ):
        scores = valid["silver_confidence"].astype(float).to_numpy()
        mask = ~np.isnan(scores)
        if mask.any():
            pr_auc = float(average_precision_score(y_true[mask], scores[mask]))

    # Bootstrap CI
    bootstrap_ci = _bootstrap_ci(y_true, y_pred, minority_class, seed=seed)

    return {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "minority_class": minority_class,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": balanced_acc,
        "cohens_kappa": kappa,
        "krippendorff_alpha": alpha,
        "pr_auc": pr_auc,
        "bootstrap_ci": bootstrap_ci,
    }


def _krippendorff_alpha_nominal_two_raters(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute nominal Krippendorff's alpha for two validation label vectors.

    The validation gate currently compares pairwise-complete human and silver
    labels, but alpha is reported alongside κ because it extends naturally to
    missing/abstaining annotators if the gate later admits those cells
    (Krippendorff's alpha).

    Args:
        y_true: Human validation labels.
        y_pred: Silver labels aligned to ``y_true``.

    Returns:
        Krippendorff's alpha for nominal labels, or ``nan`` when no labels are
        available.

    """
    if len(y_true) == 0:
        return float("nan")

    observed_disagreement = float(np.mean(y_true != y_pred))
    pooled = np.concatenate([y_true, y_pred])
    labels = np.unique(pooled)

    # For two complete raters, nominal alpha is 1 minus observed disagreement over
    # chance disagreement computed from the pooled label distribution.
    expected_agreement = sum(
        float((np.sum(pooled == label) / len(pooled)) ** 2) for label in labels
    )
    expected_disagreement = 1 - expected_agreement
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else 0.0
    return float(1 - observed_disagreement / expected_disagreement)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minority_class: int,
    seed: int,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for accuracy and minority-class F1.

    The minority-F1 lower bound is the variance-aware half of the freeze gate,
    preventing a high point estimate from freezing labels when validation
    uncertainty remains too wide.

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
    rng = np.random.default_rng(seed=seed)
    n = len(y_true)
    accs = np.empty(n_resamples, dtype=float)
    f1s = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        accs[i] = float(np.mean(y_true[idx] == y_pred[idx]))
        f1s[i] = f1_score(
            y_true[idx],
            y_pred[idx],
            pos_label=minority_class,
            zero_division=0,
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
            f"human_label (0/1) for the validation split before stage 04.",
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
            f"human_label (0/1) for the validation split before stage 04.",
        )
    return sub
