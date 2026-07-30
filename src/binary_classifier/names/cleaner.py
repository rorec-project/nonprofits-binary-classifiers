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


def normalize_name(raw: object) -> str:
    """Repair, suffix-strip, and truecase one organization name."""
    if raw is None or pd.isna(raw):
        return ""
    value = ftfy.fix_text(str(raw)).strip()
    if not value:
        return ""
    value = basename(value).strip(" ,.;:")
    words = value.split()
    return " ".join(
        word.upper()
        if word.upper() in _ACRONYMS
        else word[:1].upper() + word[1:].lower()
        for word in words
    )


def clean_names(cfg: "BinaryClassifierConfig", registry: "PathRegistry") -> None:
    """Clean both name populations and fail on dangerous panel divergence."""
    panel = pd.read_parquet(registry.names_panel_frame).copy()
    bmf_only = pd.read_parquet(registry.names_bmf_only_frame).copy()
    panel["name_cleaned"] = panel["name_raw"].map(normalize_name)
    bmf_only["name_cleaned"] = bmf_only["name_raw"].map(normalize_name)

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
    blocking: list[str] = []
    nonblocking: list[dict[str, str]] = []
    for _, row in panel.iterrows():
        upstream = _tokens(row.get("name_bare", ""))
        cleaned = _tokens(row.get("name_cleaned", ""))
        cleaned_lower = {token.lower() for token in cleaned}
        cleaned_exact = set(cleaned)
        lost_religious = [
            token for token in _religious_tokens(upstream) if token not in cleaned_lower
        ]
        lost_acronyms = [
            token
            for token in upstream
            if token in _ACRONYMS and token not in cleaned_exact
        ]
        if lost_religious:
            blocking.append(f"{row['EIN2']}: religious token(s) lost: {lost_religious}")
        if lost_acronyms:
            blocking.append(f"{row['EIN2']}: acronym(s) lost: {lost_acronyms}")
        if [token.lower() for token in upstream] != [
            token.lower() for token in cleaned
        ]:
            nonblocking.append(
                {
                    "EIN2": str(row["EIN2"]),
                    "upstream": " ".join(upstream),
                    "cleaned": " ".join(cleaned),
                },
            )
    return {
        "blocking_divergences": blocking,
        "nonblocking_divergence_count": len(nonblocking),
        "nonblocking_divergences": nonblocking,
        "panel_rows_audited": len(panel),
    }


def _tokens(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return _WORD_RE.findall(str(value))


def _religious_tokens(tokens: list[str]) -> set[str]:
    lexicon_words = {
        word.lower() for entry in RELIGIOUS_LEXICON for word in entry.split()
    }
    return {token.lower() for token in tokens if token.lower() in lexicon_words}
