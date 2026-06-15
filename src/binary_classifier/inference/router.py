"""Routing policy for classifier inference and LOW-tier rules."""

import logging
from typing import Protocol

from binary_classifier.config import InferenceConfig
from binary_classifier.data.quality import apply_rule_label

logger = logging.getLogger(__name__)


class HasInferenceConfig(Protocol):
    """Protocol for config objects exposing inference settings."""

    inference: InferenceConfig


def route(text: str, tier: str, cfg: HasInferenceConfig) -> tuple[str, int | None]:
    """Route one text row to classifier scoring or a deterministic rule label.

    HIGH and MEDIUM quality rows always go to the classifier. LOW-quality rows
    optionally use the rule layer first: strong religious lexicon hits become
    positive rule labels, very short non-religious fragments become negative
    rule labels, and ambiguous rows either fall through to the classifier or
    abstain depending on configuration.

    Args:
        text: Input text to classify.
        tier: Quality tier (``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``), with
            surrounding whitespace and casing normalized.
        cfg: Full pipeline config or test double exposing an ``inference``
            config object.

    Returns:
        A ``(route_name, label)`` pair. ``label`` is only set for deterministic
        rule outcomes.

    """
    normalized_tier = tier.strip().upper()

    if normalized_tier in {"HIGH", "MEDIUM"}:
        return "classifier", None

    if normalized_tier != "LOW" or not cfg.inference.route_low_to_rules:
        return "classifier", None

    rule_label = apply_rule_label(text)
    if rule_label == 1:
        return "rule_strong_positive", 1
    if rule_label == 0:
        return "rule_short_negative", 0
    if cfg.inference.rule_ambiguous_to_classifier:
        return "low_via_classifier", None
    return "rule_abstain", None
