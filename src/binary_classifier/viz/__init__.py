"""Pure plotting helpers for binary-classifier stage artifacts."""

from binary_classifier.viz.curves import (
    documentation_curve,
    pr_curve,
    reliability_diagram,
)
from binary_classifier.viz.bakeoff import (
    bakeoff_summary,
    canary_drift,
    production_annotation_summary,
)
from binary_classifier.viz.ngrams import ngram_log_odds
from binary_classifier.viz.prevalence_plots import prevalence_forest

__all__ = [
    "bakeoff_summary",
    "canary_drift",
    "documentation_curve",
    "ngram_log_odds",
    "pr_curve",
    "prevalence_forest",
    "production_annotation_summary",
    "reliability_diagram",
]
