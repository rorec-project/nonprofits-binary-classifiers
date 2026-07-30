"""Fixed synthetic probes for names-arm diagnostic runs.

These constructed strings diagnose model response patterns; they are not a
representative evaluation set and must never be used for accuracy claims.
"""

PROBE_SET_VERSION = "v2"

PROBES = (
    ("tradition_baptist", "tradition_token", "First Baptist Church", "baptist_pair"),
    ("control_community", "matched_control", "First Community Church", "baptist_pair"),
    ("tradition_soccer", "tradition_token", "First Baptist Soccer Club", "soccer_pair"),
    ("control_soccer", "matched_control", "First Community Soccer Club", "soccer_pair"),
    ("saint_hospital", "saint_name", "St Mary's Hospital", None),
    ("saint_school", "saint_name", "Saint Luke Academy", None),
    ("faith_health", "faith_heritage", "Trinity Health", None),
    ("faith_school", "faith_heritage", "Grace Heritage Academy", None),
    ("jewish", "tradition_token", "Beth Shalom Synagogue", "synagogue_pair"),
    (
        "secular_jewish",
        "matched_control",
        "Beth Shalom Center",
        "synagogue_pair",
    ),
    ("muslim", "tradition_token", "Al Noor Mosque", "mosque_pair"),
    ("secular_muslim", "matched_control", "Al Noor Center", "mosque_pair"),
    ("hindu", "tradition_token", "Lakshmi Hindu Center", "hindu_pair"),
    ("secular_hindu", "matched_control", "Lakshmi Civic Center", "hindu_pair"),
    ("buddhist", "tradition_token", "Lotus Buddhist Center", "buddhist_pair"),
    ("secular_buddhist", "matched_control", "Lotus Community Center", "buddhist_pair"),
)
