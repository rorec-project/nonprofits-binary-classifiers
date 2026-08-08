"""Computable quality rubric ``Q`` and religious-lexicon priors.

The rubric scores short nonprofit mission texts on structural richness
(purpose verb, named beneficiary, named activity, length, clause count)
while penalizing boilerplate, vague templates, and extreme brevity.

The module also provides a high-precision **rule label** for the LOW /
bare-label inference layer (point 5 in the roadmap) and the **positive-
protective rescue rule** for enriching the positive pool.
"""

import re

from binary_classifier.config import QThresholdsConfig

# ruff: noqa: PLR2004


# ── Lexicons ───────────────────────────────────────────────────────────────────

RELIGIOUS_LEXICON = [
    "christ",
    "christian",
    "jesus",
    "gospel",
    "bible",
    "biblical",
    "scripture",
    "church",
    "congregation",
    "parish",
    "ministry",
    "ministries",
    "missionary",
    "evangel",
    "catholic",
    "baptist",
    "methodist",
    "lutheran",
    "presbyterian",
    "episcopal",
    "pentecostal",
    "jewish",
    "synagogue",
    "torah",
    "rabbi",
    "islam",
    "muslim",
    "mosque",
    "quran",
    "buddhist",
    "hindu",
    "temple",
    "salvation army",
    "worship",
    "prayer",
    "holy",
    "sacred",
    "seminary",
    "chapel",
    "diocese",
    "orthodox",
    "spiritual",
    "faith",
    "faith-based",
]

STRONG_TRADITION_LEXICON = [
    "christ",
    "christian",
    "jesus",
    "gospel",
    "bible",
    "church",
    "catholic",
    "baptist",
    "methodist",
    "lutheran",
    "presbyterian",
    "episcopal",
    "pentecostal",
    "jewish",
    "synagogue",
    "torah",
    "rabbi",
    "islam",
    "muslim",
    "mosque",
    "quran",
    "buddhist",
    "hindu",
    "temple",
    "salvation army",
    "seminary",
    "chapel",
    "diocese",
    "orthodox",
]

_PURPOSE_VERBS = [
    "provide",
    "support",
    "promote",
    "serve",
    "educate",
    "deliver",
    "fund",
    "preserve",
    "assist",
    "help",
    "aid",
    "offer",
    "give",
    "facilitate",
    "advance",
    "strengthen",
    "develop",
    "build",
    "create",
    "maintain",
    "manage",
    "operate",
    "improve",
    "enhance",
    "protect",
    "conserve",
    "advocate",
    "defend",
    "empower",
    "enrich",
    "inspire",
    "encourage",
    "prepare",
    "train",
    "teach",
    "guide",
    "lead",
    "organize",
    "coordinate",
    "conduct",
    "perform",
    "establish",
    "foster",
    "cultivate",
    "nurture",
    "sustain",
    "expand",
    "extend",
    "increase",
    "raise",
    "collect",
    "distribute",
    "share",
    "bring",
    "keep",
    "ensure",
    "work",
    "act",
    "dedicate",
    "commit",
    "devote",
    "strive",
    "aim",
    "seek",
    "endeavor",
    "undertake",
    "carry",
    "engage",
    "participate",
    "collaborate",
    "partner",
    "unite",
    "join",
    "link",
    "connect",
    "integrate",
    "combine",
    "merge",
    "present",
    "glorify",
    "worship",
    "pray",
    "minister",
]

_BENEFICIARIES = [
    "children",
    "child",
    "kids",
    "youth",
    "families",
    "family",
    "individuals",
    "individual",
    "community",
    "communities",
    "students",
    "student",
    "seniors",
    "senior",
    "elderly",
    "aged",
    "disabled",
    "disability",
    "veterans",
    "veteran",
    "patients",
    "patient",
    "homeless",
    "animals",
    "animal",
    "refugees",
    "refugee",
    "immigrants",
    "immigrant",
    "women",
    "woman",
    "men",
    "man",
    "people",
    "persons",
    "person",
    "residents",
    "resident",
    "public",
    "society",
    "world",
    "nation",
    "country",
    "state",
    "city",
    "town",
    "county",
    "region",
    "area",
    "neighborhood",
    "district",
    "poor",
    "needy",
    "disadvantaged",
    "underserved",
    "low-income",
    "low income",
    "at-risk",
    "at risk",
    "high-risk",
    "high risk",
    "vulnerable",
    "marginalized",
    "minority",
    "majority",
    "group",
    "population",
    "demographic",
    "class",
    "category",
]

_ACTIVITIES = [
    "scholarships",
    "scholarship",
    "food bank",
    "housing",
    "research",
    "counseling",
    "camps",
    "camp",
    "education",
    "training",
    "support",
    "services",
    "service",
    "programs",
    "program",
    "activities",
    "activity",
    "events",
    "event",
    "outreach",
    "advocacy",
    "assistance",
    "care",
    "treatment",
    "therapy",
    "rehabilitation",
    "recovery",
    "shelter",
    "meals",
    "meal",
    "clothing",
    "clothes",
    "medicine",
    "healthcare",
    "health",
    "wellness",
    "fitness",
    "sports",
    "sport",
    "recreation",
    "arts",
    "art",
    "culture",
    "music",
    "dance",
    "theater",
    "theatre",
    "film",
    "media",
    "publishing",
    "writing",
    "reading",
    "literacy",
    "library",
    "museum",
    "conservation",
    "preservation",
    "restoration",
    "protection",
    "cleanup",
    "recycling",
    "gardening",
    "farming",
    "agriculture",
    "nutrition",
    "cooking",
    "food",
    "water",
    "sanitation",
    "hygiene",
    "safety",
    "security",
    "emergency",
    "rescue",
    "relief",
    "aid",
    "development",
    "construction",
    "building",
    "renovation",
    "repair",
    "maintenance",
    "transportation",
    "travel",
    "tourism",
    "hospitality",
    "accommodation",
    "lodging",
    "camping",
    "scouting",
    "mentoring",
    "tutoring",
    "coaching",
    "teaching",
    "instruction",
    "guidance",
    "leadership",
    "fellowship",
    "networking",
    "partnership",
    "collaboration",
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def _word_boundary_pattern(term: str) -> str:
    """Return a regex pattern with word boundaries for ``term``."""
    escaped = re.escape(term)
    return rf"(?<!\w){escaped}(?!\w)"


# ── Compiled patterns ──────────────────────────────────────────────────────────

_RE_RELIGIOUS = re.compile(
    "|".join(_word_boundary_pattern(w) for w in RELIGIOUS_LEXICON),
    re.IGNORECASE,
)

_RE_STRONG_TRADITION = re.compile(
    "|".join(_word_boundary_pattern(w) for w in STRONG_TRADITION_LEXICON),
    re.IGNORECASE,
)

_RE_PURPOSE_VERBS = re.compile(
    "|".join(_word_boundary_pattern(w) for w in _PURPOSE_VERBS),
    re.IGNORECASE,
)

_RE_BENEFICIARIES = re.compile(
    "|".join(_word_boundary_pattern(w) for w in _BENEFICIARIES),
    re.IGNORECASE,
)

_RE_ACTIVITIES = re.compile(
    "|".join(_word_boundary_pattern(w) for w in _ACTIVITIES),
    re.IGNORECASE,
)

_BOILERPLATE_PATTERNS = [
    re.compile(r"\b501\s*\(c\)", re.IGNORECASE),
    re.compile(r"\binternal\s+revenue\s+code\b", re.IGNORECASE),
    re.compile(r"\birc\b", re.IGNORECASE),
    re.compile(r"\btax\s+exempt\b", re.IGNORECASE),
    re.compile(r"\bexempt\s+from\s+taxation\b", re.IGNORECASE),
]

_VAGUE_PATTERNS = [
    re.compile(r"\bto\s+promote\s+the\s+general\s+welfare\b", re.IGNORECASE),
    re.compile(r"\bgeneral\s+welfare\b", re.IGNORECASE),
    re.compile(r"\bcharitable\s+purposes\s+only\b", re.IGNORECASE),
    re.compile(r"\bexclusively\s+charitable\b", re.IGNORECASE),
    re.compile(r"\bexclusively\s+for\s+charitable\b", re.IGNORECASE),
    re.compile(r"\bgeneral\s+charitable\s+purposes\b", re.IGNORECASE),
    re.compile(r"\bbroad\s+charitable\s+purposes\b", re.IGNORECASE),
    re.compile(r"\ball\s+lawful\s+purposes\b", re.IGNORECASE),
    re.compile(r"\bany\s+lawful\s+purpose\b", re.IGNORECASE),
    re.compile(r"\ball\s+legal\s+purposes\b", re.IGNORECASE),
    re.compile(r"\bany\s+legal\s+purpose\b", re.IGNORECASE),
]


# ── Public API ─────────────────────────────────────────────────────────────────


def compute_quality_score(text: str) -> float:
    """Compute the quality rubric score ``Q`` for a single mission text.

    ``Q`` is a weighted sum of structural features (length, purpose verb,
    beneficiary, activity, specificity, clause count) minus penalties for
    boilerplate, vague templates, or extreme brevity. Max theoretical
    score is 6.0; default tiers are HIGH ≥5.0, MEDIUM [3.0, 5.0),
    LOW <3.0. The separate 4.5-<5.0 band is only the positive-protective
    rescue enrichment sub-rule, not the MEDIUM tier upper bound.

    Args:
        text: Raw mission text (will be lower-cased internally).

    Returns:
        The total quality score ``Q`` (float).

    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    t = text.lower()
    words = t.split()
    word_count = len(words)

    # Length score (capped sub-linear)
    if word_count < 4:
        s_len = 0.0
    elif word_count <= 7:
        s_len = 0.5
    elif word_count <= 14:
        s_len = 1.0
    else:
        s_len = 1.5

    # Structural features
    f_purpose = 1.0 if _RE_PURPOSE_VERBS.search(t) else 0.0
    f_benef = 1.0 if _RE_BENEFICIARIES.search(t) else 0.0
    f_acts = 1.0 if _RE_ACTIVITIES.search(t) else 0.0
    f_specific = 1.0 if (f_benef > 0 and f_acts > 0) else 0.0

    # Multi-clause indicator
    clause_marks = len(re.findall(r"[.;,]", t))
    s_clause = 0.5 if clause_marks >= 2 else 0.0

    # Penalties
    penalties = 0.0

    if any(p.search(t) for p in _BOILERPLATE_PATTERNS):
        penalties -= 1.5

    # Vague-only template: apply only when the text is short enough that
    # the vague phrase dominates the mission statement.
    if any(p.search(t) for p in _VAGUE_PATTERNS) and word_count < 15:
        penalties -= 2.0

    if word_count < 4:
        penalties -= 1.0

    q = s_len + f_purpose + f_benef + f_acts + f_specific + s_clause + penalties
    return q


def compute_quality_components(text: str) -> dict[str, float]:
    """Return a dictionary of every ``Q`` component for inspection.

    Keys: ``Q``, ``s_len``, ``f_purpose``, ``f_benef``, ``f_acts``,
    ``f_specific``, ``s_clause``, ``penalties``.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    t = text.lower()
    words = t.split()
    word_count = len(words)

    if word_count < 4:
        s_len = 0.0
    elif word_count <= 7:
        s_len = 0.5
    elif word_count <= 14:
        s_len = 1.0
    else:
        s_len = 1.5

    f_purpose = 1.0 if _RE_PURPOSE_VERBS.search(t) else 0.0
    f_benef = 1.0 if _RE_BENEFICIARIES.search(t) else 0.0
    f_acts = 1.0 if _RE_ACTIVITIES.search(t) else 0.0
    f_specific = 1.0 if (f_benef > 0 and f_acts > 0) else 0.0

    clause_marks = len(re.findall(r"[.;,]", t))
    s_clause = 0.5 if clause_marks >= 2 else 0.0

    penalties = 0.0
    if any(p.search(t) for p in _BOILERPLATE_PATTERNS):
        penalties -= 1.5
    if any(p.search(t) for p in _VAGUE_PATTERNS) and word_count < 15:
        penalties -= 2.0
    if word_count < 4:
        penalties -= 1.0

    q = s_len + f_purpose + f_benef + f_acts + f_specific + s_clause + penalties

    return {
        "Q": q,
        "s_len": s_len,
        "f_purpose": f_purpose,
        "f_benef": f_benef,
        "f_acts": f_acts,
        "f_specific": f_specific,
        "s_clause": s_clause,
        "penalties": penalties,
    }


def has_religious_lexicon(text: str) -> bool:
    """Return True if ``text`` contains any religious-lexicon word."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return bool(_RE_RELIGIOUS.search(text.lower()))


def has_strong_tradition_lexicon(text: str) -> bool:
    """Return True if ``text`` contains a strong-tradition (denominational) term."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return bool(_RE_STRONG_TRADITION.search(text.lower()))


def has_concrete_purpose_verb(text: str) -> bool:
    """Return True if ``text`` contains a concrete purpose/action verb."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return bool(_RE_PURPOSE_VERBS.search(text.lower()))


def is_positive_rescue(text: str, q_score: float) -> bool:
    """Apply the positive-protective rescue rule.

    MEDIUM rows (top of MEDIUM band, ``Q`` ≥4.5 and <5.0) that carry a
    strong-tradition lexicon hit **and** a concrete purpose verb are treated
    as eligible for the positive sampling pool.

    Args:
        text: Mission text.
        q_score: Pre-computed ``Q`` score.

    Returns:
        True if the row qualifies for rescue.

    """
    return (
        4.5 <= q_score < 5.0
        and has_strong_tradition_lexicon(text)
        and has_concrete_purpose_verb(text)
    )


def assign_tier(q_score: float, thresholds: QThresholdsConfig | None = None) -> str:
    """Assign a quality tier based on ``Q`` score.

    Args:
        q_score: Computed quality score.
        thresholds: Tier thresholds. If None, defaults from
            ``QThresholdsConfig`` are used.

    Returns:
        One of ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.

    """
    if thresholds is None:
        thresholds = QThresholdsConfig()

    if q_score >= thresholds.HIGH:
        return "HIGH"
    elif q_score >= thresholds.MEDIUM:
        return "MEDIUM"
    return "LOW"


def apply_rule_label(text: str) -> int | None:
    """High-precision rule label for the LOW / bare-label inference layer.

    Returns ``1`` when the text contains a strong-tradition lexicon hit
    (high-confidence religious signal), ``0`` when the text is very short
    and contains no religious lexicon at all (high-confidence secular),
    and ``None`` when the signal is ambiguous (routed to the classifier
    rather than the rule layer at inference time).

    Args:
        text: Mission text.

    Returns:
        ``1``, ``0``, or ``None``.

    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    t = text.lower()
    words = t.split()

    # Strong positive signal: denominational or core tradition terms
    if has_strong_tradition_lexicon(t):
        return 1

    # Strong negative signal: very short, no religious lexicon whatsoever
    if len(words) < 6 and not has_religious_lexicon(t):
        return 0

    # Ambiguous: let the classifier handle it
    return None
