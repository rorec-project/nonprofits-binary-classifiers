"""Pure plotting helpers for binary-classifier stage artifacts."""

from binary_classifier.viz.curves import (
    documentation_curve,
    draw_single_confusion_matrix,
    frozen_test_confusion_matrices,
    frozen_test_curves,
    threshold_sweep_plot,
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
from binary_classifier.viz.ngrams import (
    compute_keyness_frame,
    keyness_sensitivity_heatmap,
    keyness_volcano_plot,
    ngram_log_odds,
    ngram_weighted_log_odds,
    term_scatter_plot,
    top_terms_lollipop_plot,
)
from binary_classifier.viz.prevalence_plots import (
    ntee_classified_count_by_group,
    ntee_classified_share_by_group,
    ntee_classified_share_vs_corrected_estimate,
    ntee_mean_score_by_group,
    prevalence_decomposition,
    prevalence_forest,
    quantification_sensitivity,
    rule_validation_intervals,
)
from binary_classifier.viz.wordclouds import (
    build_class_wordcloud,
    class_wordcloud,
    class_wordclouds,
    write_wordcloud_pdf,
    write_wordcloud_svg,
)

__all__ = [
    "bakeoff_summary",
    "build_class_wordcloud",
    "canary_drift",
    "class_wordcloud",
    "class_wordclouds",
    "compute_keyness_frame",
    "documentation_curve",
    "draw_single_confusion_matrix",
    "frozen_test_confusion_matrices",
    "frozen_test_curves",
    "threshold_sweep_plot",
    "ngram_log_odds",
    "ngram_weighted_log_odds",
    "keyness_sensitivity_heatmap",
    "keyness_volcano_plot",
    "ntee_classified_count_by_group",
    "ntee_classified_share_by_group",
    "ntee_classified_share_vs_corrected_estimate",
    "ntee_mean_score_by_group",
    "prevalence_decomposition",
    "pr_curve",
    "prevalence_forest",
    "production_annotation_summary",
    "quantification_sensitivity",
    "reliability_diagram",
    "rule_validation_intervals",
    "score_distribution_by_tier_label",
    "subgroup_performance",
    "term_scatter_plot",
    "top_terms_lollipop_plot",
    "write_wordcloud_pdf",
    "write_wordcloud_svg",
]
