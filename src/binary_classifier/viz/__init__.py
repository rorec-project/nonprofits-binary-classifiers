"""Pure plotting helpers for binary-classifier stage artifacts."""

from binary_classifier.viz.curves import (
    documentation_curve,
    frozen_test_confusion_matrices,
    frozen_test_curves,
    pr_curve,
    reliability_diagram,
    score_distribution_by_tier_label,
    subgroup_performance,
)
from binary_classifier.viz.bakeoff import (
    bakeoff_summary,
    canary_drift,
    production_annotation_summary,
)
from binary_classifier.viz.ngrams import ngram_log_odds
from binary_classifier.viz.prevalence_plots import (
    prevalence_decomposition,
    prevalence_forest,
    quantification_sensitivity,
    rule_validation_intervals,
)

__all__ = [
    "bakeoff_summary",
    "canary_drift",
    "documentation_curve",
    "frozen_test_confusion_matrices",
    "frozen_test_curves",
    "ngram_log_odds",
    "prevalence_decomposition",
    "pr_curve",
    "prevalence_forest",
    "production_annotation_summary",
    "quantification_sensitivity",
    "reliability_diagram",
    "rule_validation_intervals",
    "score_distribution_by_tier_label",
    "subgroup_performance",
]
