"""Diagnostic-only synthetic probes and legal-name/DBA case study."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from binary_classifier.data.quality import (
    STRONG_TRADITION_LEXICON,
    apply_rule_label,
)
from binary_classifier.inference import load_selected_model, score_texts
from binary_classifier.names.cleaner import normalize_name
from binary_classifier.names.probes import PROBES, PROBE_SET_VERSION
from binary_classifier.names.score import _config_hash, _extract_model_provenance

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


_DBA_REQUIRED_COLUMNS = {
    "EIN2",
    "population",
    "name_cased",
    "dba_cased",
    "has_dba",
    "is_manifest_contaminated",
}
_TOKEN_PATTERN = re.compile(
    r"(?<!\w)"
    + "|".join(re.escape(term) for term in STRONG_TRADITION_LEXICON)
    + r"(?!\w)",
    re.IGNORECASE,
)


# ── Diagnostic artifact orchestration ─────────────────────────────────────────
def run_name_diagnostics(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    *,
    predictor: Any | None = None,
) -> None:
    """Write constructed-probe scores and an EIN2-level DBA case study.

    Both outputs are diagnostic-only. Probe scores are raw cross-field-transfer
    scores, and DBA names are never introduced as a production input variant.
    """
    selected = load_selected_model(registry, require_checkpoint=predictor is None)
    model_id, checkpoint_sha256 = _extract_model_provenance(selected, predictor)
    timestamp = datetime.now(UTC).isoformat()
    cache: dict[str, Any] = {}
    probes = _score_probes(
        cfg,
        selected,
        model_id=model_id,
        checkpoint_sha256=checkpoint_sha256,
        timestamp=timestamp,
        predictor=predictor,
        predictor_cache=cache,
    )
    cases, report = _build_dba_case_study(
        registry.names_panel_cleaned,
        model_id=model_id,
        checkpoint_sha256=checkpoint_sha256,
        timestamp=timestamp,
        config_hash=_config_hash(cfg),
        cfg=cfg,
        selected=selected,
        predictor=predictor,
        predictor_cache=cache,
    )
    registry.ensure_dirs()
    registry.names_probe_diagnostics.write_text(
        json.dumps(probes, indent=2, sort_keys=True) + "\n",
    )
    cases.to_parquet(registry.names_dba_case_study, index=False)
    registry.names_dba_case_study_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )


# ── Synthetic probe scoring ───────────────────────────────────────────────────
def _score_probes(
    cfg: BinaryClassifierConfig,
    selected: dict[str, Any],
    *,
    model_id: str,
    checkpoint_sha256: str,
    timestamp: str,
    predictor: Any | None,
    predictor_cache: dict[str, Any],
) -> dict[str, object]:
    raw_texts = [probe[2] for probe in PROBES]
    cleaned_texts = [normalize_name(text) for text in raw_texts]
    scores = score_texts(
        cfg,
        selected,
        cleaned_texts,
        predictor=predictor,
        predictor_cache=predictor_cache,
    )
    records = [
        {
            "probe_id": probe_id,
            "category": category,
            "pair_id": pair_id,
            "text_raw": raw_text,
            "text_cleaned": cleaned_text,
            "prob_raw": float(score),
            "lexicon_rule_label": apply_rule_label(cleaned_text),
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_sha256,
            "inference_date": timestamp,
            "config_hash": _config_hash(cfg),
        }
        for (probe_id, category, raw_text, pair_id), cleaned_text, score in zip(
            PROBES, cleaned_texts, scores, strict=True
        )
    ]
    return {
        "diagnostic_only": True,
        "interpretation": "diagnosis_not_accuracy",
        "probe_set_version": PROBE_SET_VERSION,
        "records": records,
    }


# ── DBA case-study construction ───────────────────────────────────────────────
def _build_dba_case_study(
    path: Path,
    *,
    model_id: str,
    checkpoint_sha256: str,
    timestamp: str,
    config_hash: str,
    cfg: BinaryClassifierConfig,
    selected: dict[str, Any],
    predictor: Any | None,
    predictor_cache: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    panel = pd.read_parquet(path)
    missing = sorted(_DBA_REQUIRED_COLUMNS.difference(panel.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required DBA columns: {', '.join(missing)}"
        )
    # Compare legal and DBA names only for diagnosis; DBA is never a production input.
    candidates = panel.loc[panel["has_dba"].fillna(False).astype(bool)].copy()
    candidates["legal_name_cleaned"] = candidates["name_cased"].map(normalize_name)
    candidates["dba_name_cleaned"] = candidates["dba_cased"].map(normalize_name)
    candidates["legal_name_religious_tokens"] = candidates["legal_name_cleaned"].map(
        _religious_tokens
    )
    candidates["dba_religious_tokens"] = candidates["dba_name_cleaned"].map(
        _religious_tokens
    )
    candidates["dba_only_religious_tokens"] = [
        sorted(set(dba_tokens) - set(legal_tokens))
        for legal_tokens, dba_tokens in zip(
            candidates["legal_name_religious_tokens"],
            candidates["dba_religious_tokens"],
            strict=True,
        )
    ]
    candidates["legal_name_only_religious_tokens"] = [
        sorted(set(legal_tokens) - set(dba_tokens))
        for legal_tokens, dba_tokens in zip(
            candidates["legal_name_religious_tokens"],
            candidates["dba_religious_tokens"],
            strict=True,
        )
    ]
    cases = candidates.loc[
        candidates["dba_only_religious_tokens"].map(bool)
        | candidates["legal_name_only_religious_tokens"].map(bool)
    ].copy()
    cases["token_direction"] = [
        _token_direction(dba_tokens, legal_tokens)
        for dba_tokens, legal_tokens in zip(
            cases["dba_only_religious_tokens"],
            cases["legal_name_only_religious_tokens"],
            strict=True,
        )
    ]
    cases["legal_name_prob_raw"] = score_texts(
        cfg,
        selected,
        cases["legal_name_cleaned"].tolist(),
        predictor=predictor,
        predictor_cache=predictor_cache,
    )
    cases["dba_name_prob_raw"] = score_texts(
        cfg,
        selected,
        cases["dba_name_cleaned"].tolist(),
        predictor=predictor,
        predictor_cache=predictor_cache,
    )
    cases["diagnostic_only"] = True
    cases["production_input_variant"] = False
    cases["model_id"] = model_id
    cases["checkpoint_sha256"] = checkpoint_sha256
    cases["inference_date"] = timestamp
    cases["config_hash"] = config_hash
    columns = [
        "EIN2",
        "population",
        "name_cased",
        "dba_cased",
        "legal_name_cleaned",
        "dba_name_cleaned",
        "legal_name_religious_tokens",
        "dba_religious_tokens",
        "dba_only_religious_tokens",
        "legal_name_only_religious_tokens",
        "token_direction",
        "legal_name_prob_raw",
        "dba_name_prob_raw",
        "is_manifest_contaminated",
        "diagnostic_only",
        "production_input_variant",
        "model_id",
        "checkpoint_sha256",
        "inference_date",
        "config_hash",
    ]
    cases = (
        cases[columns].sort_values(["token_direction", "EIN2"]).reset_index(drop=True)
    )
    report: dict[str, object] = {
        "diagnostic_only": True,
        "production_input_variant": False,
        "population": "panel_501c3",
        "grain": "EIN2 using BEST_NAME_CASED and BEST_DBA_CASED",
        "dba_having_organizations": len(candidates),
        "dba_adds_religious_token_count": int(
            cases["dba_only_religious_tokens"].map(bool).sum()
        ),
        "legal_name_adds_religious_token_count": int(
            cases["legal_name_only_religious_tokens"].map(bool).sum()
        ),
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "inference_date": timestamp,
        "config_hash": config_hash,
    }
    return cases, report


# ── Religious-token helpers ───────────────────────────────────────────────────
def _religious_tokens(text: str) -> list[str]:
    """Return normalized shared-lexicon matches, including multi-word entries."""
    return sorted({match.group().lower() for match in _TOKEN_PATTERN.finditer(text)})


def _token_direction(
    dba_only_tokens: list[str], legal_name_only_tokens: list[str]
) -> str:
    if dba_only_tokens and legal_name_only_tokens:
        return "both_names_add_religious_token"
    if dba_only_tokens:
        return "dba_adds_religious_token"
    return "legal_name_adds_religious_token"
