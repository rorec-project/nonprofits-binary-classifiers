"""Truth-table tests for inference routing."""

from binary_classifier.config import BinaryClassifierConfig
from binary_classifier.inference.router import route


def test_high_tier_routes_to_classifier(tiny_config: BinaryClassifierConfig) -> None:
    """HIGH-quality rows bypass the LOW-tier rule layer."""
    assert route("church", " high ", tiny_config) == ("classifier", None)


def test_medium_tier_routes_to_classifier(tiny_config: BinaryClassifierConfig) -> None:
    """MEDIUM-quality rows bypass the LOW-tier rule layer."""
    assert route("church", "medium", tiny_config) == ("classifier", None)


def test_low_tier_routes_to_classifier_when_rules_disabled(
    tiny_config: BinaryClassifierConfig,
) -> None:
    """LOW-quality rows go straight to the classifier when rules are disabled."""
    tiny_config.inference.route_low_to_rules = False

    assert route("church", "LOW", tiny_config) == ("classifier", None)


def test_low_tier_strong_positive_rule(tiny_config: BinaryClassifierConfig) -> None:
    """LOW-quality rows with strong religious terms get a positive rule label."""
    assert route("church", "LOW", tiny_config) == ("rule_strong_positive", 1)


def test_low_tier_short_negative_rule(tiny_config: BinaryClassifierConfig) -> None:
    """LOW-quality short non-religious fragments get a negative rule label."""
    assert route("animal rescue", "LOW", tiny_config) == (
        "rule_short_negative",
        0,
    )


def test_low_tier_ambiguous_rule_falls_through_to_classifier(
    tiny_config: BinaryClassifierConfig,
) -> None:
    """Ambiguous LOW-quality rows fall through when configured to do so."""
    assert route("faith", "LOW", tiny_config) == ("low_via_classifier", None)


def test_low_tier_ambiguous_rule_abstains(
    tiny_config: BinaryClassifierConfig,
) -> None:
    """Ambiguous LOW-quality rows abstain when classifier fallthrough is disabled."""
    tiny_config.inference.rule_ambiguous_to_classifier = False

    assert route("faith", "LOW", tiny_config) == ("rule_abstain", None)
