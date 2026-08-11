"""Data layer for the binary classifier pipeline.

Provides modules for loading upstream parquet files, computing the quality
rubric ``Q``, and constructing stratified, positive-enriched samples.
"""

from binary_classifier.data.load import load_missions
from binary_classifier.data.ntee_labels import load_ntee_labels, ntee_label_map
from binary_classifier.data.quality import assign_tier, compute_quality_score
from binary_classifier.data.sample import (
    build_gold_set,
    build_silver_pool,
    split_human_sets,
)

__all__ = [
    "load_missions",
    "compute_quality_score",
    "assign_tier",
    "build_silver_pool",
    "build_gold_set",
    "split_human_sets",
    "load_ntee_labels",
    "ntee_label_map",
]
