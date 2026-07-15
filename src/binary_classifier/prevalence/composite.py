"""Composite prevalence estimators and Rogan--Gladen correction.

This module provides two building blocks for the stage-09 prevalence estimator:

1. **Rogan--Gladen correction** for observed prevalence when the labeling
   mechanism (here, the deterministic rule layer applied to LOW-quality rows)
   has imperfect sensitivity and specificity.
2. **Composite stratum recombination** that merges HIGH+MEDIUM and LOW
   sub-estimates with fixed corpus shares.

The Rogan--Gladen formula (Rogan & Gladen, 1978,
https://doi.org/10.1093/oxfordjournals.aje.a112510) is the standard
epidemiological correction for prevalence under an imperfect diagnostic test.
Forman (2008, https://doi.org/10.1007/s10618-008-0097-y) situates the same
correction in the machine-learning quantification literature as Adjusted
Classify-and-Count (ACC).

The delta-method variance treats the three inputs as independent and propagates
their uncertainties into the corrected prevalence.  The composite estimator
applies fixed stratum shares rather than estimated ones, so variance comes only
from the within-stratum estimates.

References
----------
* Rogan, W. J., & Gladen, B. (1978). Estimating prevalence from the results of
  a screening test. *American Journal of Epidemiology*, 107(1), 71--76.
  https://doi.org/10.1093/oxfordjournals.aje.a112510
* Forman, G. (2008). Quantifying Counts and Costs via Classification.
  *Data Mining and Knowledge Discovery*, 17(2), 221--252.
  https://doi.org/10.1007/s10618-008-0097-y

"""

from __future__ import annotations

import math
from collections.abc import Mapping


def rogan_gladen(p_obs: float, sens: float, spec: float) -> float:
    """Correct observed prevalence for imperfect sensitivity and specificity.

    Implements the Rogan--Gladen correction
    (Rogan & Gladen, 1978, https://doi.org/10.1093/oxfordjournals.aje.a112510)
    clipped to the unit interval.  The formula is standard in epidemiology
    for de-biasing an observed prevalence when the test (here, the rule layer)
    has known sensitivity and specificity.

    Args:
        p_obs: Observed positive share in the rule-labeled population.
        sens: Rule sensitivity, ``P(rule=1 | human=1)``.
        spec: Rule specificity, ``P(rule=0 | human=0)``.

    Returns:
        The Rogan--Gladen estimate clipped to the unit interval.

    Raises:
        ValueError: If inputs are non-finite, outside ``[0, 1]``, or if
            ``sens + spec - 1`` is not positive.

    """
    p_obs_value = _unit_interval(p_obs, "p_obs")
    sens_value = _unit_interval(sens, "sens")
    spec_value = _unit_interval(spec, "spec")
    denominator = sens_value + spec_value - 1.0
    if denominator <= 0.0:
        raise ValueError("sens + spec - 1 must be positive for Rogan-Gladen")
    estimate = (p_obs_value + spec_value - 1.0) / denominator
    return _clip01(estimate)


def rogan_gladen_variance(
    p_obs: float,
    sens: float,
    spec: float,
    p_obs_var: float,
    sens_var: float,
    spec_var: float,
) -> float:
    """Delta-method variance for the Rogan--Gladen prevalence estimate.

    Variance terms are treated as independent. The derivatives are evaluated on
    the unclipped Rogan--Gladen expression; the returned variance is clipped only
    at zero to avoid negative roundoff.

    Args:
        p_obs: Observed positive share in the rule-labeled population.
        sens: Rule sensitivity.
        spec: Rule specificity.
        p_obs_var: Variance of ``p_obs``.
        sens_var: Variance of ``sens``.
        spec_var: Variance of ``spec``.

    Returns:
        Non-negative delta-method variance.

    Raises:
        ValueError: If inputs are invalid.

    """
    p_obs_value = _unit_interval(p_obs, "p_obs")
    sens_value = _unit_interval(sens, "sens")
    spec_value = _unit_interval(spec, "spec")
    obs_var = _nonnegative_finite(p_obs_var, "p_obs_var")
    se_var = _nonnegative_finite(sens_var, "sens_var")
    sp_var = _nonnegative_finite(spec_var, "spec_var")

    denominator = sens_value + spec_value - 1.0
    if denominator <= 0.0:
        raise ValueError("sens + spec - 1 must be positive for Rogan-Gladen")
    numerator = p_obs_value + spec_value - 1.0

    d_obs = 1.0 / denominator
    d_sens = -numerator / denominator**2
    d_spec = (sens_value - p_obs_value) / denominator**2
    variance = d_obs**2 * obs_var + d_sens**2 * se_var + d_spec**2 * sp_var
    return max(0.0, float(variance))


def composite(
    prev_by_stratum: Mapping[str, tuple[float, float, float]],
) -> tuple[float, float]:
    """Recombine stratum estimates with fixed corpus shares.

    Implements ``p_hat = sum_k share_k * p_hat_k`` and
    ``Var(p_hat) = sum_k share_k^2 * Var_k``.

    Args:
        prev_by_stratum: Mapping from stratum name to
            ``(estimate, variance, corpus_share)``.

    Returns:
        Composite estimate and variance.

    Raises:
        ValueError: If any component is non-finite, outside its admissible range,
            or if shares do not sum to one within a small tolerance.

    """
    if not prev_by_stratum:
        raise ValueError("prev_by_stratum must be non-empty")

    estimate = 0.0
    variance = 0.0
    share_sum = 0.0
    for stratum, (stratum_estimate, stratum_variance, share) in prev_by_stratum.items():
        est_value = _unit_interval(stratum_estimate, f"{stratum}.estimate")
        var_value = _nonnegative_finite(stratum_variance, f"{stratum}.variance")
        share_value = _unit_interval(share, f"{stratum}.share")
        estimate += share_value * est_value
        variance += share_value**2 * var_value
        share_sum += share_value

    if not math.isclose(share_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("stratum shares must sum to 1")
    return _clip01(estimate), max(0.0, float(variance))


def _unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
