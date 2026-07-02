"""Stage 09 prevalence estimation entrypoint.

This module implements the **population-prevalence estimation** stage of the
binary-classifier pipeline.  It consumes three upstream artifacts:

* ``predictions_parquet`` (stage 07) -- calibrated probabilities and rule labels
  for the full scored corpus, split into HIGH/MEDIUM and LOW quality tiers.
* ``anchor_oof_scores`` + ``anchor_manifest`` (stage 06/08) -- human labels and
  out-of-fold calibrated probabilities for the audit/anchor sample, joined by
  ``EIN2``.
* ``rule_validation`` (stage 07) -- sensitivity and specificity estimates for the
  deterministic rule layer applied to LOW-quality rows.

Methodology
-----------

The prevalence estimand is the population share of the positive class (religious
missions) over the full scored corpus.  Because the pipeline splits rows into
HIGH+MEDIUM (HM) and LOW tiers, the final estimate is a **composite** of two
stratum-specific estimates weighted by fixed corpus shares.

HIGH+MEDIUM tier
    We use **Prediction-Powered Inference** (PPI; Angelopoulos et al., 2023,
    https://doi.org/10.1126/science.adi6001) with a small gold-label audit
    sample and calibrated classifier probabilities on the unlabeled population.
    PPI treats the classifier as a proxy and corrects the population mean using
    the labeled residual, yielding valid confidence intervals without assuming the
    model is perfectly calibrated.  Design weights (inverse sampling probabilities)
    can be incorporated to account for the non-uniform sampling of the anchor
    (Horvitz & Thompson, 1952, https://doi.org/10.1080/01621459.1952.10483446).
    The PPI mean estimator is algebraically equivalent to the classical
    survey-sampling difference estimator (Cassel et al., 1976), and PPI++
    corresponds to the generalized regression (GREG) estimator (Mozer, 2026,
    arXiv:2603.19160).

LOW tier
    LOW-quality rows lack reliable classifier scores, so they are labeled by a
    deterministic rule layer.  The observed rule-label prevalence is corrected
    for misclassification using the **Rogan--Gladen** adjustment
    (Rogan & Gladen, 1978, https://doi.org/10.1093/oxfordjournals.aje.a112510)
    with sensitivity and specificity estimated on the anchor subset.
    When the rule validation is degenerate (e.g. zero-denominator), the module
    falls back to the uncorrected observed rate.

Composite
    The stratum estimates are recombined with fixed corpus shares and their
    variances are propagated (Forman, 2008,
    https://doi.org/10.1007/s10618-008-0097-y).

Cross-checks
    EMQ (Saerens et al., 2002, https://doi.org/10.1162/089976602753284446) and
    KDEy (Moreo et al., 2021 via QuaPy) are run as sensitivity checks on the HM
    tier only.  They are reported but do not override the primary PPI estimate.

NTEE breakdown
    When enabled, the same two-stage estimator is applied independently to each
    NTEE major group.  Small groups are suppressed and fall back to an EMQ
    estimate without confidence intervals.

Prevalence estimation from text classifiers is a well-studied problem in
computational social science.  Hopkins & King (2010,
https://doi.org/10.1111/j.1540-5907.2009.00428.x) showed that estimating
aggregate proportions (prevalence) is a distinct task from document-level
classification, and that direct estimation can outperform classify-and-count.
Keith & O'Connor (2018, https://aclanthology.org/D18-1487/) introduced
uncertainty-aware generative models for document-level prevalence inference.
Gentzkow, Kelly & Taddy (2019, https://doi.org/10.1257/jel.20181020) provide the
applied-economics text-as-data framework within which this pipeline sits.
Meyer & Mittag (2017, https://doi.org/10.1016/j.jeconom.2017.06.012) formalize
the econometric consequences of misclassification in binary choice models.

References
----------
* Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction-Powered Inference. *Science*.
  https://doi.org/10.1126/science.adi6001
* Hopkins, D. J., & King, G. (2010). A Method of Automated Nonparametric
  Content Analysis for Social Science. *American Journal of Political Science*.
  https://doi.org/10.1111/j.1540-5907.2009.00428.x
* Keith, K., & O'Connor, B. (2018). Uncertainty-Aware Generative Models for
  Inferring Document Class Prevalence. *EMNLP*.
  https://aclanthology.org/D18-1487/
* Gentzkow, M., Kelly, B., & Taddy, M. (2019). Text as Data. *Journal of
  Economic Literature*. https://doi.org/10.1257/jel.20181020
* Meyer, B. D., & Mittag, N. (2017). Misclassification in Binary Choice Models.
  *Journal of Econometrics*. https://doi.org/10.1016/j.jeconom.2017.06.012
* Horvitz, D. G., & Thompson, D. J. (1952). A Generalization of Sampling
  Without Replacement from a Finite Universe. *Journal of the American
  Statistical Association*, 47(260), 663--685.
  https://doi.org/10.1080/01621459.1952.10483446
* Mozer, R. (2026). PPI is the Difference Estimator: Recognizing the Survey
  Sampling Roots of Prediction-Powered Inference. arXiv:2603.19160.
  https://doi.org/10.48550/arXiv.2603.19160
* Rogan, W. J., & Gladen, B. (1978). Estimating prevalence from the results of
  a screening test. *American Journal of Epidemiology*, 107(1), 71--76.
  https://doi.org/10.1093/oxfordjournals.aje.a112510
* Forman, G. (2008). Quantifying Counts and Costs via Classification.
  *Data Mining and Knowledge Discovery*, 17(2), 221--252.
  https://doi.org/10.1007/s10618-008-0097-y
* Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the Outputs of
  a Classifier to New a Priori Probabilities: A Simple Procedure.
  *Neural Computation*, 14(1), 21--41.
  https://doi.org/10.1162/089976602753284446
* Moreo, A., Esuli, A., & Sebastiani, F. (2021). QuaPy: A Python-based open
  source framework for quantification. *ACM SIGIR*.

"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

from binary_classifier.data.load import load_missions
from binary_classifier.prevalence.composite import (
    composite,
    rogan_gladen,
    rogan_gladen_variance,
)
from binary_classifier.prevalence.ppi import ppi_prevalence
from binary_classifier.prevalence.quantify import emq_prevalence, kdey_prevalence
from binary_classifier.prevalence.weights import design_weights

if TYPE_CHECKING:
    from pathlib import Path

    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

# Tier definitions and artifact schema
# ------------------------------------
# These constants define the columns expected in upstream stage-07/08 artifacts
# and the quality tiers used to split the corpus.  HIGH and MEDIUM rows are
# scored by the classifier; LOW rows are labeled by a deterministic rule layer.
# The split is pre-determined in stage 01 (Q >= 3.0 for HM, Q < 3.0 for LOW).
_HM_TIERS = {"HIGH", "MEDIUM"}
_LOW_TIER = "LOW"
_PREDICTION_COLUMNS = {
    "EIN2",
    "pred_label",
    "prob_calibrated",
    "decision_source",
    "tier",
    "ntee_major_group",
}
_ANCHOR_OOF_COLUMNS = {
    "EIN2",
    "prob_calibrated_oof",
    "human_label",
    "tier",
    "sample_prob",
}
_ANCHOR_MANIFEST_COLUMNS = {"EIN2", "tier", "ntee_major_group", "sample_prob"}
_MISSING = object()
_RAW_MULTIPLICITY_COLUMN = "_raw_ein2_multiplicity"


# ---------------------------------------------------------------------------
# Prevalence result containers
# ---------------------------------------------------------------------------
# Small frozen dataclasses hold point estimates, variances, and confidence
# bounds so that downstream report generation is type-safe and immutable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RuleMetric:
    """A rule-metric point estimate with its variance and confidence bounds.

    Attributes:
        value: Point estimate (e.g. sensitivity or specificity).
        variance: Delta-method or Wald variance.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.

    """

    value: float
    variance: float
    ci_lower: float
    ci_upper: float


T = TypeVar("T")


def _assert_not_none(x: T | None, name: str = "") -> T:
    """Narrow a nullable value to non-None for the type checker."""
    if x is None:
        raise TypeError(f"Expected {name} to be non-None")
    return x


@dataclass(frozen=True)
class _PrevalenceEstimate:
    """A prevalence point estimate with its variance and confidence bounds.

    Attributes:
        estimate: Point estimate clipped to [0, 1].
        variance: Non-negative variance.
        ci_lower: Lower bound of the confidence interval, clipped to [0, 1].
        ci_upper: Upper bound of the confidence interval, clipped to [0, 1].

    """

    estimate: float
    variance: float
    ci_lower: float
    ci_upper: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable prevalence estimate bundle."""
        return {
            "estimate": self.estimate,
            "variance": self.variance,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


# ---------------------------------------------------------------------------
# Stage 09 entrypoint
# ---------------------------------------------------------------------------
# run_prevalence orchestrates the full prevalence estimation pipeline:
# 1. Load and validate upstream artifacts.
# 2. Compute HM prevalence via PPI (primary) and design-weighted PPI.
# 3. Compute LOW prevalence via Rogan--Gladen or uncorrected observed rate.
# 4. Recombine stratum estimates with fixed corpus shares.
# 5. Run cross-checks (EMQ, KDEy) on the HM tier.
# 6. Break down by NTEE major group.
# 7. Write the JSON report and the per-NTEE CSV.
# ---------------------------------------------------------------------------


def run_prevalence(cfg: BinaryClassifierConfig, registry: PathRegistry) -> None:
    """Run stage 09 prevalence estimation from stage-07/08 artifacts.

    Args:
        cfg: Validated binary-classifier configuration.
        registry: Path registry exposing prediction, evaluation, anchor, and
            prevalence artifact paths.

    Raises:
        RuntimeError: If a required upstream artifact is missing.
        ValueError: If an input artifact has a malformed schema or no usable rows
            remain for a required estimator.

    """
    registry.prevalence_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(cfg.prevalence.alpha)
    z_value = _z_value(alpha)

    predictions, anchor, rule_validation = _load_inputs(cfg, registry)
    total_n = _raw_count(predictions)
    hm_predictions = _high_medium(predictions)
    low_predictions = _low(predictions)
    anchor_hm = _high_medium(anchor)
    anchor_low = _low(anchor)
    shares = {
        "HIGH_MEDIUM": _raw_count(hm_predictions) / total_n,
        "LOW": _raw_count(low_predictions) / total_n,
    }

    logger.info(
        "Estimating prevalence with %d HM rows, %d LOW rows, and %d anchor rows",
        len(hm_predictions),
        len(low_predictions),
        len(anchor),
    )

    # HM prevalence is estimated with PPI (Angelopoulos et al., 2023).
    # Both unweighted and weighted (inverse-probability) variants are produced;
    # the config selects which one is primary.
    hm = _estimate_hm(anchor_hm, hm_predictions, alpha=alpha, z_value=z_value)
    hm_primary_key = (
        "weighted_ppi" if cfg.prevalence.use_design_weights else "unweighted_ppi"
    )
    hm_primary = hm[hm_primary_key]

    # LOW prevalence uses the Rogan--Gladen correction when sensitivity and
    # specificity are available from stage-07 rule validation; otherwise it
    # falls back to the uncorrected observed rule-label rate.
    low = _estimate_low(
        anchor_low,
        low_predictions,
        rule_validation,
        alpha=alpha,
        z_value=z_value,
        include_sensitivity=bool(cfg.prevalence.low_tier_sensitivity),
    )
    composite_estimate, composite_variance = composite(
        {
            "HIGH_MEDIUM": (
                hm_primary.estimate,
                hm_primary.variance,
                shares["HIGH_MEDIUM"],
            ),
            "LOW": (low.estimate.estimate, low.estimate.variance, shares["LOW"]),
        },
    )
    composite_bundle = _estimate_from_variance(
        composite_estimate,
        composite_variance,
        z_value,
    )

    # Quantification cross-checks (EMQ, KDEy) are run on the HM tier only.
    # They are reported for diagnostic purposes but never override the PPI primary.
    cross_checks = _run_cross_checks(
        cfg.prevalence.cross_checks,
        anchor_hm,
        hm_predictions,
    )
    by_ntee = _prevalence_by_ntee(
        cfg,
        anchor,
        predictions,
        rule_validation,
        alpha=alpha,
        z_value=z_value,
    )
    by_ntee.to_csv(registry.prevalence_by_ntee, index=False)
    logger.info("Wrote NTEE prevalence estimates to %s", registry.prevalence_by_ntee)

    report = _report(
        cfg,
        registry,
        alpha=alpha,
        predictions=predictions,
        anchor=anchor,
        anchor_hm=anchor_hm,
        anchor_low=anchor_low,
        hm_predictions=hm_predictions,
        low_predictions=low_predictions,
        shares=shares,
        hm=hm,
        hm_primary_key=hm_primary_key,
        low=low,
        composite_bundle=composite_bundle,
        cross_checks=cross_checks,
        by_ntee=by_ntee,
    )
    _write_json(registry.prevalence_report, report)
    logger.info("Wrote prevalence report to %s", registry.prevalence_report)


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------
# _load_inputs reads the four upstream artifacts, validates schemas, joins
# anchor_oof_scores to anchor_manifest by EIN2, and normalizes tiers.
# Any schema mismatch or duplicate EIN2 raises ValueError so that estimation
# never runs on malformed data.
# ---------------------------------------------------------------------------


def _load_inputs(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions_path = (
        registry.predictions_full_parquet
        if registry.predictions_full_parquet.exists()
        else registry.predictions_parquet
    )
    _require_file(predictions_path, "predictions_parquet")
    _require_file(registry.anchor_oof_scores, "anchor_oof_scores")
    _require_file(registry.rule_validation, "rule_validation")
    _require_file(registry.anchor_manifest, "anchor_manifest")

    predictions = pd.read_parquet(predictions_path)
    anchor_oof = pd.read_parquet(registry.anchor_oof_scores)
    anchor_manifest = pd.read_csv(registry.anchor_manifest)
    rule_validation = json.loads(registry.rule_validation.read_text())
    if not isinstance(rule_validation, dict):
        raise ValueError(f"{registry.rule_validation} must contain a JSON object")

    _require_columns(predictions, _PREDICTION_COLUMNS, "predictions_parquet")
    _require_columns(anchor_oof, _ANCHOR_OOF_COLUMNS, "anchor_oof_scores")
    _require_columns(anchor_manifest, _ANCHOR_MANIFEST_COLUMNS, "anchor_manifest")
    predictions = _normalize_frame(predictions, "predictions_parquet")
    anchor_oof = _normalize_frame(anchor_oof, "anchor_oof_scores")
    anchor_manifest = _normalize_frame(anchor_manifest, "anchor_manifest")
    _raise_on_duplicate_ein2(predictions, "predictions_parquet")
    _raise_on_duplicate_ein2(anchor_oof, "anchor_oof_scores")
    _raise_on_duplicate_ein2(anchor_manifest, "anchor_manifest")

    manifest_ntee = anchor_manifest[["EIN2", "ntee_major_group"]]
    anchor = anchor_oof.merge(
        manifest_ntee,
        on="EIN2",
        how="left",
        validate="one_to_one",
    )
    if anchor["ntee_major_group"].isna().any():
        missing = int(anchor["ntee_major_group"].isna().sum())
        raise ValueError(
            f"anchor_oof_scores has {missing} EIN2 values missing from anchor_manifest",
        )

    predictions = _finalize_common_columns(predictions)
    anchor = _finalize_common_columns(anchor)
    predictions = _attach_raw_multiplicity(cfg, predictions, registry, predictions_path)
    anchor = _attach_anchor_multiplicity(anchor, predictions)
    _validate_tiers(predictions, "predictions_parquet")
    _validate_tiers(anchor, "anchor_oof_scores")
    if predictions.empty:
        raise ValueError("predictions_parquet must contain at least one row")
    return predictions, anchor, rule_validation


# ---------------------------------------------------------------------------
# HIGH+MEDIUM prevalence estimation (PPI)
# ---------------------------------------------------------------------------
# The HM tier contains rows with reliable classifier probabilities.  We apply
# Prediction-Powered Inference (Angelopoulos et al., 2023) to the labeled
# anchor (human labels + out-of-fold calibrated probabilities) and the unlabeled
# corpus (calibrated probabilities only).  Both unweighted and design-weighted
# PPI estimates are produced; the config selects the primary one.
# ---------------------------------------------------------------------------


def _estimate_hm(
    anchor_hm: pd.DataFrame,
    hm_predictions: pd.DataFrame,
    *,
    alpha: float,
    z_value: float,
) -> dict[str, _PrevalenceEstimate]:
    anchor = anchor_hm.dropna(
        subset=["human_label", "prob_calibrated_oof", "sample_prob"],
    ).copy()
    dropped_anchor = len(anchor_hm) - len(anchor)
    if dropped_anchor:
        logger.warning(
            "Dropped %d HM anchor rows with missing labels/scores",
            dropped_anchor,
        )
    corpus = hm_predictions.dropna(subset=["prob_calibrated"]).copy()
    dropped_corpus = len(hm_predictions) - len(corpus)
    if dropped_corpus:
        logger.warning(
            "Dropped %d HM prediction rows with missing calibrated probabilities",
            dropped_corpus,
        )

    if corpus.empty:
        if hm_predictions.empty:
            empty = _estimate_from_variance(0.0, 0.0, z_value)
            return {"unweighted_ppi": empty, "weighted_ppi": empty}
        raise ValueError("No usable HM calibrated probabilities remain for PPI")
    if anchor.empty:
        raise ValueError("No usable HM anchor rows remain for PPI")

    y_labeled = pd.to_numeric(anchor["human_label"], errors="coerce")
    yhat_labeled = pd.to_numeric(anchor["prob_calibrated_oof"], errors="coerce")
    yhat_unlabeled = _expand_series_by_multiplicity(corpus, "prob_calibrated")
    if (
        y_labeled.isna().any()
        or yhat_labeled.isna().any()
        or yhat_unlabeled.isna().any()
    ):
        raise ValueError(
            "HM prevalence inputs must be numeric after missing rows are dropped",
        )

    unweighted_raw = ppi_prevalence(
        y_labeled.tolist(),
        yhat_labeled.tolist(),
        yhat_unlabeled.tolist(),
        alpha=alpha,
    )
    weights = design_weights(anchor, normalize=True)
    weighted_raw = ppi_prevalence(
        y_labeled.tolist(),
        yhat_labeled.tolist(),
        yhat_unlabeled.tolist(),
        alpha=alpha,
        w=weights.tolist(),
    )
    return {
        "unweighted_ppi": _ppi_to_estimate(unweighted_raw, z_value),
        "weighted_ppi": _ppi_to_estimate(weighted_raw, z_value),
    }


def _anchor_residual_multiplicity_sensitivity(
    anchor_hm: pd.DataFrame,
    hm_predictions: pd.DataFrame,
    baseline: _PrevalenceEstimate,
    *,
    alpha: float,
    z_value: float,
) -> dict[str, float | str]:
    anchor = anchor_hm.dropna(subset=["human_label", "prob_calibrated_oof"]).copy()
    corpus = hm_predictions.dropna(subset=["prob_calibrated"]).copy()
    if anchor.empty or corpus.empty:
        return {"status": "skipped", "reason": "no usable HM anchor/corpus rows"}
    try:
        weighted_raw = ppi_prevalence(
            pd.to_numeric(anchor["human_label"], errors="coerce").tolist(),
            pd.to_numeric(anchor["prob_calibrated_oof"], errors="coerce").tolist(),
            _expand_series_by_multiplicity(corpus, "prob_calibrated").tolist(),
            alpha=alpha,
            w=_multiplicity(anchor).tolist(),
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
    estimate = _ppi_to_estimate(weighted_raw, z_value)
    return {
        "status": "ok",
        "estimate": estimate.estimate,
        "delta_vs_unweighted_anchor_residual": estimate.estimate - baseline.estimate,
    }


# ---------------------------------------------------------------------------
# LOW-tier prevalence estimation (Rogan--Gladen)
# ---------------------------------------------------------------------------
# LOW-quality rows lack reliable classifier scores, so they are labeled by a
# deterministic rule layer.  The observed rule-label prevalence is corrected
# for misclassification using the Rogan--Gladen formula
# (Rogan & Gladen, 1978, https://doi.org/10.1093/oxfordjournals.aje.a112510)
# with sensitivity and specificity from stage-07 validation.  When the rule
# metrics are missing or degenerate (sens + spec - 1 <= 0), the module falls
# back to the uncorrected observed rate to avoid a division-by-zero or
# unstable correction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LowEstimate:
    """A LOW-tier prevalence estimate with diagnostic metadata.

    Attributes:
        estimate: Prevalence point estimate and variance/CI bundle.
        observed_prevalence: Raw rule-label prevalence before correction.
        observed_variance: Variance of the observed rule-label prevalence.
        n_rule_labeled: Number of LOW rows used.
        sensitivity: Sensitivity metric, or None if unvalidated.
        specificity: Specificity metric, or None if unvalidated.
        sensitivity_band: Min/max RG estimate over sensitivity CI bounds,
            or None if not computed.
        corrected: Whether the Rogan--Gladen correction was applied.

    """

    estimate: _PrevalenceEstimate
    observed_prevalence: float
    observed_variance: float
    n_rule_labeled: int
    sensitivity: _RuleMetric | None
    specificity: _RuleMetric | None
    sensitivity_band: dict[str, float] | None
    corrected: bool = True
    sub_strata: dict[str, Any] | None = None


def _estimate_low(
    anchor_low: pd.DataFrame,
    low_predictions: pd.DataFrame,
    rule_validation: Mapping[str, Any],
    *,
    alpha: float,
    z_value: float,
    include_sensitivity: bool,
) -> _LowEstimate:
    if low_predictions.empty:
        return _LowEstimate(
            estimate=_estimate_from_variance(0.0, 0.0, z_value),
            observed_prevalence=0.0,
            observed_variance=0.0,
            n_rule_labeled=0,
            sensitivity=None,
            specificity=None,
            sensitivity_band=None,
            corrected=False,
            sub_strata={},
        )

    classifier_low = _low_classifier(low_predictions)
    rule_low = _low_rules(low_predictions)
    total_raw = _raw_count(low_predictions)
    components: dict[str, tuple[float, float, float]] = {}
    sub_strata: dict[str, Any] = {
        "total_raw_n": total_raw,
        "classifier_raw_n": _raw_count(classifier_low),
        "rule_raw_n": _raw_count(rule_low),
    }

    if not classifier_low.empty:
        if "decision_source" in anchor_low.columns:
            anchor_classifier_low = _low_classifier(anchor_low)
        else:
            anchor_classifier_low = anchor_low
        low_ppi = _estimate_hm(
            anchor_classifier_low,
            classifier_low,
            alpha=alpha,
            z_value=z_value,
        )["weighted_ppi"]
        classifier_share = _raw_count(classifier_low) / total_raw
        components["low_via_classifier"] = (
            low_ppi.estimate,
            low_ppi.variance,
            classifier_share,
        )
        sub_strata["low_via_classifier"] = {
            "estimator": "ppi",
            "share": classifier_share,
            "raw_n": _raw_count(classifier_low),
            "estimate": low_ppi.as_dict(),
        }

    rule_estimate = _estimate_low_rules(
        rule_low,
        rule_validation,
        alpha=alpha,
        z_value=z_value,
        include_sensitivity=include_sensitivity,
    )
    if _raw_count(rule_low) > 0:
        rule_share = _raw_count(rule_low) / total_raw
        components["low_rule"] = (
            rule_estimate.estimate.estimate,
            rule_estimate.estimate.variance,
            rule_share,
        )
        sub_strata["rule"] = {
            "estimator": "rogan_gladen"
            if rule_estimate.corrected
            else "uncorrected_observed",
            "share": rule_share,
            "raw_n": _raw_count(rule_low),
            "estimate": rule_estimate.estimate.as_dict(),
            "observed_prevalence": rule_estimate.observed_prevalence,
        }

    if not components:
        raise ValueError("No usable LOW rows remain after sub-stratification")
    point, variance = composite(components)
    estimate = _estimate_from_variance(point, variance, z_value)
    return _LowEstimate(
        estimate=estimate,
        observed_prevalence=rule_estimate.observed_prevalence,
        observed_variance=rule_estimate.observed_variance,
        n_rule_labeled=rule_estimate.n_rule_labeled,
        sensitivity=rule_estimate.sensitivity,
        specificity=rule_estimate.specificity,
        sensitivity_band=rule_estimate.sensitivity_band,
        corrected=rule_estimate.corrected,
        sub_strata=sub_strata,
    )


def _estimate_low_rules(
    low_predictions: pd.DataFrame,
    rule_validation: Mapping[str, Any],
    *,
    alpha: float,
    z_value: float,
    include_sensitivity: bool,
) -> _LowEstimate:
    if low_predictions.empty:
        return _LowEstimate(
            estimate=_estimate_from_variance(0.0, 0.0, z_value),
            observed_prevalence=0.0,
            observed_variance=0.0,
            n_rule_labeled=0,
            sensitivity=_maybe_rule_metric(rule_validation, "sensitivity", alpha=alpha),
            specificity=_maybe_rule_metric(rule_validation, "specificity", alpha=alpha),
            sensitivity_band=None,
            corrected=False,
            sub_strata={},
        )

    low = low_predictions.dropna(subset=["pred_label"]).copy()
    dropped = len(low_predictions) - len(low)
    if dropped:
        logger.warning("Dropped %d LOW rows with missing rule labels", dropped)
    if low.empty:
        raise ValueError("No usable LOW rule labels remain for Rogan-Gladen")
    labels = pd.to_numeric(low["pred_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("LOW pred_label values must be numeric 0/1")

    weights = _multiplicity(low)
    p_obs = float(np.average(labels.astype(float), weights=weights))
    n_low = int(weights.sum())
    p_obs_var = p_obs * (1.0 - p_obs) / n_low
    sensitivity = _maybe_rule_metric(rule_validation, "sensitivity", alpha=alpha)
    specificity = _maybe_rule_metric(rule_validation, "specificity", alpha=alpha)
    corrected = (
        sensitivity is not None
        and specificity is not None
        and (sensitivity.value + specificity.value - 1.0) > 0.0
    )
    if corrected:
        # Type narrowing: `corrected` implies both metrics are present.
        assert sensitivity is not None and specificity is not None
        estimate = rogan_gladen(p_obs, sensitivity.value, specificity.value)
        variance = rogan_gladen_variance(
            p_obs,
            sensitivity.value,
            specificity.value,
            p_obs_var,
            sensitivity.variance,
            specificity.variance,
        )
    else:
        logger.warning(
            "LOW-tier rule layer is unvalidated (missing or degenerate "
            "sensitivity/specificity); using the uncorrected observed LOW rate.",
        )
        estimate = p_obs
        variance = p_obs_var
    return _LowEstimate(
        estimate=_estimate_from_variance(estimate, variance, z_value),
        observed_prevalence=p_obs,
        observed_variance=p_obs_var,
        n_rule_labeled=n_low,
        sensitivity=sensitivity,
        specificity=specificity,
        corrected=corrected,
        sensitivity_band=(
            _low_sensitivity_band(
                p_obs,
                _assert_not_none(sensitivity, "sensitivity"),
                _assert_not_none(specificity, "specificity"),
            )
            if include_sensitivity and corrected
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Quantification cross-checks
# ---------------------------------------------------------------------------
# EMQ and KDEy are run on the HM tier as diagnostic checks.  They are not
# used to produce the primary prevalence estimate but are recorded in the
# JSON report for downstream review.  EMQ implements the Saerens-Latinne-
# Decaestecker prior-shift EM loop (Saerens et al., 2002,
# https://doi.org/10.1162/089976602753284446).  KDEy uses QuaPy's
# KDEyML quantifier (Moreo et al., 2021).
# ---------------------------------------------------------------------------


def _run_cross_checks(
    requested: Iterable[str],
    anchor_hm: pd.DataFrame,
    hm_predictions: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    anchor = anchor_hm.dropna(subset=["human_label", "prob_calibrated_oof"])
    corpus = hm_predictions.dropna(subset=["prob_calibrated"])
    if anchor.empty or corpus.empty:
        return {
            name: {"status": "skipped", "reason": "no usable HM anchor/corpus rows"}
            for name in requested
        }

    val_posteriors = _posterior_matrix(anchor["prob_calibrated_oof"])
    val_labels = pd.to_numeric(anchor["human_label"], errors="coerce").tolist()
    corpus_posteriors = _posterior_matrix(corpus["prob_calibrated"])
    for name in requested:
        if name == "emq":
            try:
                results[name] = {
                    "status": "ok",
                    "estimate": emq_prevalence(
                        val_posteriors,
                        val_labels,
                        corpus_posteriors,
                    ),
                    "scope": "HIGH_MEDIUM",
                }
            except Exception as exc:
                results[name] = {
                    "status": "failed",
                    "error": str(exc),
                    "scope": "HIGH_MEDIUM",
                }
                logger.warning("EMQ cross-check failed: %s", exc)
        elif name == "kdey":
            try:
                results[name] = {
                    "status": "ok",
                    "estimate": kdey_prevalence(
                        val_posteriors,
                        val_labels,
                        corpus_posteriors,
                    ),
                    "scope": "HIGH_MEDIUM",
                }
            except Exception as exc:
                results[name] = {
                    "status": "failed",
                    "error": str(exc),
                    "scope": "HIGH_MEDIUM",
                }
                logger.warning("KDEy cross-check failed: %s", exc)
        else:
            results[name] = {"status": "skipped", "reason": "unknown cross-check"}
    return results


# ---------------------------------------------------------------------------
# Subgroup prevalence by NTEE
# ---------------------------------------------------------------------------
# When enabled, the composite estimator is applied independently to each NTEE
# major group.  Groups with fewer anchor rows than ntee_min_n are suppressed
# and fall back to an EMQ estimate without a confidence interval.  This
# prevents unstable PPI intervals from being reported as reliable.
# ---------------------------------------------------------------------------


def _prevalence_by_ntee(
    cfg: BinaryClassifierConfig,
    anchor: pd.DataFrame,
    predictions: pd.DataFrame,
    rule_validation: Mapping[str, Any],
    *,
    alpha: float,
    z_value: float,
) -> pd.DataFrame:
    columns = [
        "ntee_major_group",
        "n_anchor",
        "estimator",
        "estimate",
        "ci_lower",
        "ci_upper",
        "suppressed",
    ]
    if not cfg.prevalence.per_ntee:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    ntee_values = sorted(predictions["ntee_major_group"].astype(str).unique().tolist())
    for ntee in ntee_values:
        group_predictions = predictions.loc[
            predictions["ntee_major_group"] == ntee
        ].copy()
        group_anchor = anchor.loc[anchor["ntee_major_group"] == ntee].copy()
        n_anchor = int(len(group_anchor))
        suppressed = n_anchor < int(cfg.prevalence.ntee_min_n)
        if suppressed:
            rows.append(
                {
                    "ntee_major_group": ntee,
                    "n_anchor": n_anchor,
                    "estimator": "not estimated",
                    "estimate": math.nan,
                    "ci_lower": math.nan,
                    "ci_upper": math.nan,
                    "suppressed": True,
                },
            )
            continue

        try:
            group_estimate = _group_composite_estimate(
                cfg,
                group_anchor,
                group_predictions,
                rule_validation,
                alpha=alpha,
                z_value=z_value,
            )
            rows.append(
                {
                    "ntee_major_group": ntee,
                    "n_anchor": n_anchor,
                    "estimator": "ppi_rg_composite",
                    "estimate": group_estimate.estimate,
                    "ci_lower": group_estimate.ci_lower,
                    "ci_upper": group_estimate.ci_upper,
                    "suppressed": False,
                },
            )
        except Exception as exc:
            logger.warning(
                "NTEE %s prevalence failed; writing suppressed fallback: %s",
                ntee,
                exc,
            )
            rows.append(
                {
                    "ntee_major_group": ntee,
                    "n_anchor": n_anchor,
                    "estimator": "not estimated",
                    "estimate": math.nan,
                    "ci_lower": math.nan,
                    "ci_upper": math.nan,
                    "suppressed": True,
                },
            )
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Per-group composite estimation
# ---------------------------------------------------------------------------
# Re-applies the same two-stage (PPI + Rogan--Gladen) composite logic used
# in the overall estimate to a single NTEE group.
# ---------------------------------------------------------------------------


def _group_composite_estimate(
    cfg: BinaryClassifierConfig,
    group_anchor: pd.DataFrame,
    group_predictions: pd.DataFrame,
    rule_validation: Mapping[str, Any],
    *,
    alpha: float,
    z_value: float,
) -> _PrevalenceEstimate:
    total = _raw_count(group_predictions)
    if total == 0:
        return _estimate_from_variance(0.0, 0.0, z_value)
    hm_predictions = _high_medium(group_predictions)
    low_predictions = _low(group_predictions)
    hm_share = _raw_count(hm_predictions) / total
    low_share = _raw_count(low_predictions) / total
    if hm_share > 0.0:
        hm_estimates = _estimate_hm(
            _high_medium(group_anchor),
            hm_predictions,
            alpha=alpha,
            z_value=z_value,
        )
        hm_key = (
            "weighted_ppi" if cfg.prevalence.use_design_weights else "unweighted_ppi"
        )
        hm_estimate = hm_estimates[hm_key]
    else:
        hm_estimate = _estimate_from_variance(0.0, 0.0, z_value)
    low_estimate = _estimate_low(
        _low(group_anchor),
        low_predictions,
        rule_validation,
        alpha=alpha,
        z_value=z_value,
        include_sensitivity=False,
    ).estimate
    point, variance = composite(
        {
            "HIGH_MEDIUM": (hm_estimate.estimate, hm_estimate.variance, hm_share),
            "LOW": (low_estimate.estimate, low_estimate.variance, low_share),
        },
    )
    return _estimate_from_variance(point, variance, z_value)


# ---------------------------------------------------------------------------
# Suppressed NTEE fallback
# ---------------------------------------------------------------------------
# When an NTEE group is too small for PPI, we fall back to EMQ on the HM
# tier and an uncorrected or Rogan--Gladen estimate on the LOW tier.  The
# fallback is explicitly flagged as suppressed in the output CSV.
# ---------------------------------------------------------------------------


def _ntee_emq_fallback(
    group_anchor: pd.DataFrame,
    group_predictions: pd.DataFrame,
    rule_validation: Mapping[str, Any],
    z_value: float,
) -> float:
    total = _raw_count(group_predictions)
    if total == 0:
        return math.nan
    hm_predictions = _high_medium(group_predictions)
    low_predictions = _low(group_predictions)
    hm_share = _raw_count(hm_predictions) / total
    low_share = _raw_count(low_predictions) / total
    estimates: dict[str, tuple[float, float, float]] = {}
    if hm_share > 0.0:
        anchor_hm = _high_medium(group_anchor).dropna(
            subset=["human_label", "prob_calibrated_oof"],
        )
        hm_corpus = hm_predictions.dropna(subset=["prob_calibrated"])
        if anchor_hm.empty or hm_corpus.empty:
            return math.nan
        try:
            hm_estimate = emq_prevalence(
                _posterior_matrix(anchor_hm["prob_calibrated_oof"]),
                pd.to_numeric(anchor_hm["human_label"], errors="coerce").tolist(),
                _posterior_matrix(hm_corpus["prob_calibrated"]),
            )
        except Exception as exc:
            logger.warning("NTEE EMQ fallback failed: %s", exc)
            return math.nan
        estimates["HIGH_MEDIUM"] = (hm_estimate, 0.0, hm_share)
    if low_share > 0.0:
        try:
            low_estimate = _estimate_low(
                _low(group_anchor),
                low_predictions,
                rule_validation,
                alpha=0.05,
                z_value=z_value,
                include_sensitivity=False,
            ).estimate
        except Exception as exc:
            logger.warning("NTEE LOW fallback failed: %s", exc)
            return math.nan
        estimates["LOW"] = (low_estimate.estimate, 0.0, low_share)
    if not estimates:
        return math.nan
    try:
        return composite(estimates)[0]
    except ValueError:
        return math.nan


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
# Assembles the final JSON report from all sub-estimates, cross-checks, and
# metadata.  The report schema is versioned so that downstream consumers can
# detect structural changes.
# ---------------------------------------------------------------------------


def _report(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    *,
    alpha: float,
    predictions: pd.DataFrame,
    anchor: pd.DataFrame,
    anchor_hm: pd.DataFrame,
    anchor_low: pd.DataFrame,
    hm_predictions: pd.DataFrame,
    low_predictions: pd.DataFrame,
    shares: Mapping[str, float],
    hm: Mapping[str, _PrevalenceEstimate],
    hm_primary_key: str,
    low: _LowEstimate,
    composite_bundle: _PrevalenceEstimate,
    cross_checks: Mapping[str, Mapping[str, Any]],
    by_ntee: pd.DataFrame,
) -> dict[str, Any]:
    suppressed = int(by_ntee["suppressed"].sum()) if "suppressed" in by_ntee else 0
    return {
        "schema_version": 1,
        "stage": "09_prevalence",
        "alpha": alpha,
        "estimand_statements": {
            "HIGH_MEDIUM": (
                "Per-organization prevalence of the positive label among raw EIN2 "
                "organizations in HIGH and MEDIUM quality mission records. Duplicate "
                "mission texts are counted once per raw EIN2 in the unlabeled corpus mean."
            ),
            "LOW": (
                "Per-organization prevalence among raw EIN2 LOW-quality records. "
                "LOW classifier-routed rows use PPI; pure rule rows use Rogan-Gladen."
            ),
            "composite": (
                "Per-organization prevalence over the full raw EIN2 prediction corpus, "
                "recombining strata with raw-EIN2 tier shares."
            ),
        },
        "inputs": {
            "predictions_parquet": str(registry.predictions_parquet),
            "anchor_oof_scores": str(registry.anchor_oof_scores),
            "rule_validation": str(registry.rule_validation),
            "anchor_manifest": str(registry.anchor_manifest),
            "predictions_full_parquet": str(registry.predictions_full_parquet),
            "raw_multiplicity_source": predictions.attrs.get(
                "raw_multiplicity_source",
                "unknown",
            ),
        },
        "n": {
            "total": int(len(predictions)),
            "total_raw_ein2": _raw_count(predictions),
            "anchor": int(len(anchor)),
            "anchor_high_medium": int(len(anchor_hm)),
            "anchor_low": int(len(anchor_low)),
            "HIGH_MEDIUM": int(len(hm_predictions)),
            "HIGH_MEDIUM_raw_ein2": _raw_count(hm_predictions),
            "LOW": int(len(low_predictions)),
            "LOW_raw_ein2": _raw_count(low_predictions),
        },
        "tier_shares": dict(shares),
        "hm": {
            "primary": hm_primary_key,
            "weighted_ppi": hm["weighted_ppi"].as_dict(),
            "unweighted_ppi": hm["unweighted_ppi"].as_dict(),
            "use_design_weights": bool(cfg.prevalence.use_design_weights),
            "sensitivity_anchor_residual_multiplicity_weighted": (
                _anchor_residual_multiplicity_sensitivity(
                    anchor_hm,
                    hm_predictions,
                    hm["unweighted_ppi"],
                    alpha=alpha,
                    z_value=_z_value(alpha),
                )
            ),
        },
        "low": {
            "observed_prevalence": low.observed_prevalence,
            "observed_variance": low.observed_variance,
            "n_rule_labeled": low.n_rule_labeled,
            "correction_applied": low.corrected,
            "estimator": _low_estimator_name(low),
            "estimate": low.estimate.as_dict(),
            "rogan_gladen": low.estimate.as_dict(),
            "sensitivity": _rule_metric_dict(low.sensitivity),
            "specificity": _rule_metric_dict(low.specificity),
            "sensitivity_band": low.sensitivity_band,
            "sub_strata": low.sub_strata,
        },
        "composite": composite_bundle.as_dict(),
        "cross_checks": cross_checks,
        "per_ntee": {
            "enabled": bool(cfg.prevalence.per_ntee),
            "path": str(registry.prevalence_by_ntee),
            "n_groups": int(len(by_ntee)),
            "n_suppressed": suppressed,
            "ntee_min_n": int(cfg.prevalence.ntee_min_n),
        },
    }


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------
# Helpers that convert between variance, confidence intervals, and point
# estimates, always clipping to the unit interval.
# ---------------------------------------------------------------------------


def _estimate_from_variance(
    estimate: float,
    variance: float,
    z_value: float,
) -> _PrevalenceEstimate:
    point = _clip01(estimate)
    var = max(0.0, float(variance))
    margin = z_value * math.sqrt(var)
    return _PrevalenceEstimate(
        estimate=point,
        variance=var,
        ci_lower=_clip01(point - margin),
        ci_upper=_clip01(point + margin),
    )


def _ppi_to_estimate(raw: Mapping[str, Any], z_value: float) -> _PrevalenceEstimate:
    estimate = _finite_float(raw["estimate"], "PPI estimate")
    ci_lower = _finite_float(raw["ci_lower"], "PPI ci_lower")
    ci_upper = _finite_float(raw["ci_upper"], "PPI ci_upper")
    variance = _variance_from_ci(ci_lower, ci_upper, z_value)
    return _PrevalenceEstimate(
        estimate=_clip01(estimate),
        variance=variance,
        ci_lower=_clip01(ci_lower),
        ci_upper=_clip01(ci_upper),
    )


def _variance_from_ci(ci_lower: float, ci_upper: float, z_value: float) -> float:
    lower = _finite_float(ci_lower, "ci_lower")
    upper = _finite_float(ci_upper, "ci_upper")
    width = max(0.0, upper - lower)
    return (width / (2.0 * z_value)) ** 2


# ---------------------------------------------------------------------------
# Rule-metric extraction
# ---------------------------------------------------------------------------
# _maybe_rule_metric parses the stage-07 rule-validation JSON and extracts
# sensitivity or specificity as a _RuleMetric.  If the metric is missing,
# null, or has a zero denominator, it returns None so that the LOW tier
# falls back to the uncorrected observed rate rather than crashing on
# float(None).
# ---------------------------------------------------------------------------


def _maybe_rule_metric(
    rule_validation: Mapping[str, Any],
    name: str,
    *,
    alpha: float,
) -> _RuleMetric | None:
    """Extract a rule metric, or ``None`` when it is absent or unvalidated.

    Stage 07 writes ``value: null`` (with null CI bounds) for a rule metric
    whose denominator is zero -- i.e. the anchor sample contains LOW rows but the
    rule layer covered none of them. Returning ``None`` here lets the LOW
    prevalence path fall back to the uncorrected observed rate instead of
    crashing on ``float(None)``.

    Args:
        rule_validation: Parsed stage-07 rule-validation report.
        name: Metric name (``"sensitivity"`` or ``"specificity"``).
        alpha: Significance level used to derive the metric variance from its CI.

    Returns:
        A populated :class:`_RuleMetric`, or ``None`` when the metric is missing
        or carries a null point value.

    """
    try:
        metric = _find_rule_metric(rule_validation, name)
    except ValueError:
        return None
    if isinstance(metric, int | float):
        value = _unit_interval(metric, name)
        return _RuleMetric(value=value, variance=0.0, ci_lower=value, ci_upper=value)
    if not isinstance(metric, Mapping):
        return None

    value_obj = _first_present(
        metric,
        ("value", "estimate", "point", "mean"),
        default=None,
    )
    if value_obj is None:
        return None
    value = _unit_interval(value_obj, name)
    lower, upper = _ci_bounds(metric, value)
    variance = _variance_from_ci(lower, upper, _z_value(alpha))
    return _RuleMetric(value=value, variance=variance, ci_lower=lower, ci_upper=upper)


def _find_rule_metric(rule_validation: Mapping[str, Any], name: str) -> Any:
    metrics = rule_validation.get("metrics")
    if isinstance(metrics, Mapping) and name in metrics:
        return metrics[name]
    if name in rule_validation:
        return rule_validation[name]
    rule_name = f"rule_{name}"
    if rule_name in rule_validation:
        return rule_validation[rule_name]
    raise ValueError(f"rule_validation missing {name!r} metric")


def _ci_bounds(metric: Mapping[str, Any], value: float) -> tuple[float, float]:
    ci = _first_present(
        metric,
        ("ci", "wilson_ci", "confidence_interval"),
        default=None,
    )
    if isinstance(ci, Mapping):
        lower = _first_present(ci, ("lower", "lo", "lcl"), default=value)
        upper = _first_present(ci, ("upper", "hi", "ucl"), default=value)
    elif isinstance(ci, list | tuple) and len(ci) == 2:
        lower = ci[0]
        upper = ci[1]
    else:
        lower = _first_present(
            metric,
            ("lower", "ci_lower", "wilson_lower"),
            default=value,
        )
        upper = _first_present(
            metric,
            ("upper", "ci_upper", "wilson_upper"),
            default=value,
        )
    lower_value = _unit_interval(lower, "ci_lower")
    upper_value = _unit_interval(upper, "ci_upper")
    if lower_value > upper_value:
        raise ValueError("rule_validation CI lower bound exceeds upper bound")
    return lower_value, upper_value


# ---------------------------------------------------------------------------
# Sensitivity band computation
# ---------------------------------------------------------------------------
# Propagates the Wilson confidence intervals for sensitivity and specificity
# through the Rogan--Gladen formula to produce a plausible range for the
# corrected LOW prevalence.  This is reported as a diagnostic sensitivity band.
# ---------------------------------------------------------------------------


def _low_sensitivity_band(
    p_obs: float,
    sensitivity: _RuleMetric,
    specificity: _RuleMetric,
) -> dict[str, float]:
    estimates = []
    for sens in (sensitivity.ci_lower, sensitivity.ci_upper):
        for spec in (specificity.ci_lower, specificity.ci_upper):
            if sens + spec - 1.0 > 0.0:
                estimates.append(rogan_gladen(p_obs, sens, spec))
    if not estimates:
        raise ValueError(
            "rule metric Wilson intervals do not contain any valid RG denominator",
        )
    return {"lower": min(estimates), "upper": max(estimates)}


# ---------------------------------------------------------------------------
# Posterior matrix construction
# ---------------------------------------------------------------------------
# Converts a Series of calibrated probabilities into a (n, 2) posterior matrix
# required by the EMQ and KDEy quantifiers: column 0 is P(negative), column 1
# is P(positive).  Each row sums to 1.
# ---------------------------------------------------------------------------


def _posterior_matrix(probabilities: Iterable[Any]) -> list[list[float]]:
    probs = pd.to_numeric(pd.Series(list(probabilities)), errors="coerce")
    if probs.isna().any():
        raise ValueError("posterior probabilities must be numeric")
    values = probs.astype(float).tolist()
    return [[1.0 - p, p] for p in values]


# ---------------------------------------------------------------------------
# Tier filtering
# ---------------------------------------------------------------------------


def _high_medium(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["tier"].astype(str).str.upper().isin(_HM_TIERS)].copy()


def _low(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["tier"].astype(str).str.upper() == _LOW_TIER].copy()


def _low_classifier(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["decision_source"].astype(str).eq("low_via_classifier")
    ].copy()


def _low_rules(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame["decision_source"].astype(str)
    return frame.loc[source.str.startswith("rule_")].copy()


def _raw_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(_multiplicity(frame).sum())


def _multiplicity(frame: pd.DataFrame) -> pd.Series:
    if _RAW_MULTIPLICITY_COLUMN not in frame.columns:
        return pd.Series(1.0, index=frame.index)
    values = pd.to_numeric(frame[_RAW_MULTIPLICITY_COLUMN], errors="coerce")
    if values.isna().any() or (values <= 0).any():
        raise ValueError("raw EIN2 multiplicities must be positive numeric values")
    return values.astype(float)


def _expand_series_by_multiplicity(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    repeats = _multiplicity(frame).round().astype(int)
    return values.loc[values.index.repeat(repeats)].reset_index(drop=True)


def _attach_raw_multiplicity(
    cfg: BinaryClassifierConfig,
    predictions: pd.DataFrame,
    registry: PathRegistry,
    predictions_path: Path,
) -> pd.DataFrame:
    frame = predictions.copy()
    if _RAW_MULTIPLICITY_COLUMN in frame.columns:
        frame.attrs["raw_multiplicity_source"] = "existing_column"
        return frame
    if predictions_path == registry.predictions_full_parquet:
        frame[_RAW_MULTIPLICITY_COLUMN] = 1
        frame.attrs["raw_multiplicity_source"] = "predictions_full_parquet"
        return frame
    try:
        raw = load_missions(cfg, deduplicate=False)
        deduped = load_missions(cfg, deduplicate=True)
    except FileNotFoundError as exc:
        logger.warning(
            "Raw mission frame unavailable for multiplicity weighting; using "
            "one raw EIN2 per prediction row: %s",
            exc,
        )
        frame[_RAW_MULTIPLICITY_COLUMN] = 1
        frame.attrs["raw_multiplicity_source"] = "unit_fallback_missing_raw_frame"
        return frame

    raw_key = _mission_key_column(cfg, raw)
    deduped_key = _mission_key_column(cfg, deduped)
    counts = (
        raw.assign(_mission_key=_normalize_mission_key(raw[raw_key]))
        .groupby("_mission_key", dropna=False)
        .size()
        .to_frame(_RAW_MULTIPLICITY_COLUMN)
        .reset_index()
    )
    keyed_deduped = deduped[["EIN2", deduped_key]].copy()
    keyed_deduped["_mission_key"] = _normalize_mission_key(keyed_deduped[deduped_key])
    multiplicities = keyed_deduped[["EIN2", "_mission_key"]].merge(
        counts,
        on="_mission_key",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        multiplicities[["EIN2", _RAW_MULTIPLICITY_COLUMN]],
        on="EIN2",
        how="left",
        validate="one_to_one",
    )
    if frame[_RAW_MULTIPLICITY_COLUMN].isna().any():
        missing = int(frame[_RAW_MULTIPLICITY_COLUMN].isna().sum())
        if missing == len(frame):
            logger.warning(
                "Raw mission frame did not match prediction EIN2 values; using one "
                "raw EIN2 per prediction row. Provide predictions_full.parquet or "
                "the matching raw frame for per-organization prevalence.",
            )
            frame[_RAW_MULTIPLICITY_COLUMN] = 1
            frame.attrs["raw_multiplicity_source"] = "unit_fallback_no_raw_match"
            return frame
        raise ValueError(f"Missing raw multiplicity for {missing} prediction rows")
    frame.attrs["raw_multiplicity_source"] = "raw_frame_text_counts"
    return frame


def _attach_anchor_multiplicity(
    anchor: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if _RAW_MULTIPLICITY_COLUMN not in predictions.columns:
        anchored = anchor.copy()
        anchored[_RAW_MULTIPLICITY_COLUMN] = 1
        return anchored
    multiplicities = predictions[["EIN2", _RAW_MULTIPLICITY_COLUMN]].drop_duplicates(
        subset=["EIN2"],
    )
    anchored = anchor.merge(
        multiplicities,
        on="EIN2",
        how="left",
        validate="one_to_one",
    )
    anchored[_RAW_MULTIPLICITY_COLUMN] = anchored[_RAW_MULTIPLICITY_COLUMN].fillna(1)
    return anchored


def _mission_key_column(cfg: BinaryClassifierConfig, frame: pd.DataFrame) -> str:
    if cfg.field in frame.columns:
        return cfg.field
    if "mission_text" in frame.columns:
        return "mission_text"
    raise ValueError(f"Mission frame missing configured field {cfg.field!r}")


def _normalize_mission_key(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__BINARY_CLASSIFIER_MISSING_MISSION__")


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Required {name} artifact not found at {path}")


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _normalize_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    normalized = frame.copy()
    original_missing = normalized["EIN2"].isna()
    normalized["EIN2"] = normalized["EIN2"].astype(str).str.strip()
    missing = normalized["EIN2"].eq("") | original_missing
    if missing.any():
        raise ValueError(f"{name} contains blank EIN2 values")
    return normalized


def _finalize_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    finalized = frame.copy()
    finalized["tier"] = finalized["tier"].astype(str).str.strip().str.upper()
    finalized["ntee_major_group"] = (
        finalized["ntee_major_group"].fillna("UNKNOWN").astype(str).str.strip()
    )
    finalized.loc[finalized["ntee_major_group"].eq(""), "ntee_major_group"] = "UNKNOWN"
    return finalized


def _raise_on_duplicate_ein2(frame: pd.DataFrame, name: str) -> None:
    duplicates = int(frame["EIN2"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{name} contains {duplicates} duplicate EIN2 values")


def _validate_tiers(frame: pd.DataFrame, name: str) -> None:
    unknown = set(frame["tier"].astype(str).str.upper().unique()) - (
        _HM_TIERS | {_LOW_TIER}
    )
    if unknown:
        raise ValueError(f"{name} contains unsupported tier values: {sorted(unknown)}")


def _rule_metric_dict(metric: _RuleMetric | None) -> dict[str, float] | None:
    if metric is None:
        return None
    return {
        "value": metric.value,
        "variance": metric.variance,
        "ci_lower": metric.ci_lower,
        "ci_upper": metric.ci_upper,
    }


def _low_estimator_name(low: _LowEstimate) -> str:
    sub_strata = low.sub_strata or {}
    has_classifier = int(sub_strata.get("classifier_raw_n", 0)) > 0
    has_rule = int(sub_strata.get("rule_raw_n", 0)) > 0
    if has_classifier and has_rule:
        return "ppi_rg_composite"
    if has_classifier:
        return "ppi"
    return "rogan_gladen" if low.corrected else "uncorrected_observed"


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def _first_present(
    mapping: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    default: Any = _MISSING,
) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    if default is not _MISSING:
        return default
    raise ValueError(f"missing one of required keys: {keys}")


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _unit_interval(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _z_value(alpha: float) -> float:
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    return NormalDist().inv_cdf(1.0 - alpha_value / 2.0)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
