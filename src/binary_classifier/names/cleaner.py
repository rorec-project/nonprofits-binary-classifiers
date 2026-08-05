"""Shared name cleaning and the panel divergence safety gate."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import ftfy
import pandas as pd
from cleanco import basename

from binary_classifier.data.quality import RELIGIOUS_LEXICON

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


_ACRONYMS = {
    "AME",
    "COGIC",
    "ELCA",
    "LDS",
}
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def normalize_name(raw: object, *, strip_suffix: bool = True) -> str:
    """Repair and truecase one raw organization name.

    Legal suffix removal is optional so transfer scoring can compare its primary
    suffix-stripped input with a suffix-retaining ablation under identical encoding
    repair and casing rules.
    """
    if raw is None or pd.isna(raw):
        return ""

    # Repair source encoding before suffix and casing transformations.
    value = ftfy.fix_text(str(raw)).strip()
    if not value:
        return ""
    if strip_suffix:
        value = basename(value).strip(" ,.;:")

    # Preserve known denominational acronyms while truecasing BMF uppercase names.
    words = value.split()
    return " ".join(
        word.upper()
        if word.upper() in _ACRONYMS
        else word[:1].upper() + word[1:].lower()
        for word in words
    )


def clean_names(cfg: "BinaryClassifierConfig", registry: "PathRegistry") -> None:
    """Clean both populations and gate on religious-token loss from raw input.

    Applying one transformation from each population's raw name avoids confounding
    population comparisons with preprocessing differences. The panel audit blocks
    only when this cleaner removes a religious token retained in selected raw input.
    """
    # Normalize both populations with the identical raw-name transformation.
    panel = pd.read_parquet(registry.names_panel_frame).copy()
    bmf_only = pd.read_parquet(registry.names_bmf_only_frame).copy()
    panel["name_cleaned"] = panel["name_raw"].map(normalize_name)
    bmf_only["name_cleaned"] = bmf_only["name_raw"].map(normalize_name)

    # Persist the audit before failing so a dangerous divergence is diagnosable.
    audit = _audit_panel(panel)
    registry.ensure_dirs()
    panel.to_parquet(registry.names_panel_cleaned, index=False)
    bmf_only.to_parquet(registry.names_bmf_only_cleaned, index=False)
    registry.names_divergence_audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    blocking = audit["blocking_divergences"]
    if not isinstance(blocking, list):
        raise TypeError(
            "Name divergence audit has an invalid blocking-divergences field"
        )
    if blocking:
        raise ValueError(
            "Name cleaning divergence audit failed: " + "; ".join(blocking[:5]),
        )


def _audit_panel(panel: pd.DataFrame) -> dict[str, object]:
    """Compare selected raw names with cleaned names and identify token loss."""
    blocking: list[str] = []
    for _, row in panel.iterrows():
        raw = _tokens(row.get("name_raw", ""))
        cleaned = _tokens(row.get("name_cleaned", ""))
        cleaned_lower = {token.lower() for token in cleaned}
        cleaned_exact = set(cleaned)
        lost_religious = [
            token for token in _religious_tokens(raw) if token not in cleaned_lower
        ]
        lost_acronyms = [
            token
            for token in raw
            if token.upper() in _ACRONYMS and token.upper() not in cleaned_exact
        ]
        if lost_religious:
            blocking.append(f"{row['EIN2']}: religious token(s) lost: {lost_religious}")
        if lost_acronyms:
            blocking.append(f"{row['EIN2']}: acronym(s) lost: {lost_acronyms}")
    return {
        "blocking_divergences": blocking,
        "nonblocking_divergence_count": 0,
        "nonblocking_divergences": [],
        "panel_rows_audited": len(panel),
    }


def _tokens(value: object) -> list[str]:
    """Extract comparable word tokens from a raw or cleaned name value."""
    if value is None or pd.isna(value):
        return []
    return _WORD_RE.findall(str(value))


def _religious_tokens(tokens: list[str]) -> set[str]:
    """Return tokens belonging to the shared religious lexicon."""
    lexicon_words = {
        word.lower() for entry in RELIGIOUS_LEXICON for word in entry.split()
    }
    return {token.lower() for token in tokens if token.lower() in lexicon_words}
