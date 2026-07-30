"""Draw and gate the BMF-only human-coded names gold sample."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry


logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {
    "EIN2",
    "population",
    "name_raw",
    "is_bmf_only",
    "is_manifest_contaminated",
    "is_ntee_x",
    "is_church_foundation",
}
_STRATA = {
    "ntee_x_only",
    "church_foundation_only",
    "both_external_flags",
    "neither_external_flag",
}
_CONFLICT_PATTERNS = {
    "saint_name": re.compile(r"\b(?:st\.?|saint)\s+\w+", re.IGNORECASE),
    "faith_heritage": re.compile(
        r"\b(?:faith|grace|trinity|covenant|heritage)\b", re.IGNORECASE
    ),
    "non_christian_tradition": re.compile(
        r"\b(?:buddh(?:ist|a)?|gurdwara|hindu|islam(?:ic)?|jain|masjid|mosque|"
        r"muslim|sikh|synagogue|temple)\b",
        re.IGNORECASE,
    ),
    "non_english_name": re.compile(
        r"\b(?:amigos|asociacion|centro|comunidad|escuela|fundacion|iglesia|"
        r"sociedad)\b|[^\x00-\x7f]",
        re.IGNORECASE,
    ),
}
_CODING_RUBRIC = (
    "Use the unchanged mission construct: positive means observable religious or "
    "spiritual purpose, tradition, or motivation as a core driver of the work. "
    "A saint name alone is not religious. Faith-founded identity without religious "
    "purpose is not religious. Enter only 0 or 1 in human_label."
)
_CODING_TEMPLATE_COLUMNS = ["EIN2", "split", "text", "human_label"]


def draw_name_gold(cfg: BinaryClassifierConfig, registry: PathRegistry) -> None:
    """Draw a seeded, stratified BMF-only names sample and coding template."""
    frame = pd.read_parquet(registry.names_bmf_only_frame)
    _require_columns(frame)
    quotas = _validate_quotas(cfg)
    eligible = frame.loc[
        frame["is_bmf_only"].astype(bool)
        & frame["population"].eq("bmf_only")
        & ~frame["is_manifest_contaminated"].astype(bool)
    ].copy()
    if eligible["EIN2"].duplicated().any():
        raise ValueError("BMF-only gold frame must contain one row per EIN2.")
    eligible["gold_stratum"] = _strata(eligible)
    eligible["conflict_categories"] = eligible["name_raw"].map(_conflict_categories)
    seed = int(cfg.names.gold_seed if cfg.names.gold_seed is not None else cfg.SEED)
    draw = _sample(eligible, quotas, cfg.names.gold_conflict_quotas, seed)
    draw["conflict_categories"] = draw["conflict_categories"].map(
        lambda categories: "|".join(categories) if categories else "none"
    )
    _write_artifacts(draw, registry)
    logger.info(
        "Drew %d BMF-only names gold rows (seed=%d): %s",
        len(draw),
        seed,
        draw["gold_stratum"].value_counts().sort_index().to_dict(),
    )


def require_name_gold_coding_complete(registry: PathRegistry) -> None:
    """Block names validation until every drawn organization has a binary label."""
    template_path = registry.names_gold_coding_template
    manifest_path = registry.names_gold_manifest
    if not template_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Names gold manifest and coding template are required before validation: "
            f"{manifest_path}, {template_path}"
        )
    template = pd.read_csv(template_path)
    manifest = pd.read_csv(manifest_path)
    _require_ein2_column(template, template_path)
    _require_ein2_column(manifest, manifest_path)
    if template.columns.tolist() != _CODING_TEMPLATE_COLUMNS:
        raise ValueError(
            f"{template_path} must have columns: {', '.join(_CODING_TEMPLATE_COLUMNS)}"
        )
    if template["EIN2"].duplicated().any() or manifest["EIN2"].duplicated().any():
        raise ValueError(
            "Names gold manifest and coding template must have unique EIN2."
        )
    if set(template["EIN2"]) != set(manifest["EIN2"]):
        raise ValueError(
            "Names gold coding template EIN2 values must match the manifest."
        )
    labels = pd.to_numeric(template["human_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Names gold coding is incomplete; human_label must be 0 or 1.")
    instructions = registry.names_gold_coding_instructions
    if not instructions.exists() or _CODING_RUBRIC not in instructions.read_text():
        raise ValueError("Names gold coding instructions are missing or altered.")


def _validate_quotas(cfg: BinaryClassifierConfig) -> dict[str, int]:
    quotas = cfg.names.gold_stratum_quotas
    if set(quotas) != _STRATA or any(count < 0 for count in quotas.values()):
        raise ValueError(
            f"gold_stratum_quotas must define nonnegative counts for {_STRATA}."
        )
    if sum(quotas.values()) != cfg.names.gold_sample_size:
        raise ValueError("gold_stratum_quotas must sum to gold_sample_size.")
    unknown_conflicts = set(cfg.names.gold_conflict_quotas).difference(
        _CONFLICT_PATTERNS
    )
    if unknown_conflicts or any(
        count < 0 for count in cfg.names.gold_conflict_quotas.values()
    ):
        raise ValueError("gold_conflict_quotas contains an unknown or negative quota.")
    return quotas


def _sample(
    eligible: pd.DataFrame,
    quotas: dict[str, int],
    conflict_quotas: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    selected_by_stratum = {stratum: 0 for stratum in _STRATA}
    for category, quota in conflict_quotas.items():
        candidates = eligible.loc[
            eligible["conflict_categories"].map(lambda values: category in values)
        ]
        picked = _sample_conflict_candidates(
            candidates, selected, selected_by_stratum, quotas, quota, rng, category
        )
        new_picks = picked.loc[~picked.index.isin(selected)]
        selected.extend(new_picks.index.tolist())
        for stratum, count in new_picks["gold_stratum"].value_counts().items():
            selected_by_stratum[stratum] += int(count)
    for stratum, quota in quotas.items():
        candidates = eligible.loc[
            eligible["gold_stratum"].eq(stratum) & ~eligible.index.isin(selected)
        ]
        picked = _sample_candidates(
            candidates,
            quota - selected_by_stratum[stratum],
            rng,
            stratum,
        )
        selected.extend(picked.index.tolist())
    return eligible.loc[selected].sort_values("EIN2").reset_index(drop=True)


def _sample_candidates(
    candidates: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    description: str,
) -> pd.DataFrame:
    if len(candidates) < count:
        raise ValueError(
            f"Insufficient eligible BMF-only rows for {description}: need {count}, "
            f"found {len(candidates)}."
        )
    if count == 0:
        return candidates.iloc[0:0]
    positions = rng.choice(len(candidates), size=count, replace=False)
    return candidates.iloc[positions]


def _sample_conflict_candidates(
    candidates: pd.DataFrame,
    selected: list[int],
    selected_by_stratum: dict[str, int],
    quotas: dict[str, int],
    quota: int,
    rng: np.random.Generator,
    category: str,
) -> pd.DataFrame:
    """Reuse selected overlaps before consuming capacity in another stratum."""
    selected_candidates = candidates.loc[candidates.index.isin(selected)]
    if len(selected_candidates) >= quota:
        return _sample_candidates(
            selected_candidates,
            quota,
            rng,
            f"conflict category {category}",
        )
    available = candidates.loc[
        ~candidates.index.isin(selected)
        & candidates["gold_stratum"].map(
            lambda stratum: selected_by_stratum[stratum] < quotas[stratum]
        )
    ]
    remainder = _sample_candidates(
        available,
        quota - len(selected_candidates),
        rng,
        f"conflict category {category}",
    )
    return pd.concat([selected_candidates, remainder])


def _strata(frame: pd.DataFrame) -> pd.Series:
    ntee = frame["is_ntee_x"].astype(bool)
    church = frame["is_church_foundation"].astype(bool)
    return pd.Series(
        np.select(
            [ntee & ~church, ~ntee & church, ntee & church],
            ["ntee_x_only", "church_foundation_only", "both_external_flags"],
            default="neither_external_flag",
        ),
        index=frame.index,
        dtype="string",
    )


def _conflict_categories(name: object) -> list[str]:
    text = str(name)
    return [
        category
        for category, pattern in _CONFLICT_PATTERNS.items()
        if pattern.search(text) is not None
    ]


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "BMF-only names frame is missing required columns: " + ", ".join(missing)
        )


def _require_ein2_column(frame: pd.DataFrame, path: object) -> None:
    if "EIN2" not in frame:
        raise ValueError(f"{path} is missing required column: EIN2")


def _write_artifacts(draw: pd.DataFrame, registry: PathRegistry) -> None:
    registry.ensure_dirs()
    template_path = registry.names_gold_coding_template
    manifest_path = registry.names_gold_manifest
    if manifest_path.exists() and manifest_path.read_text() != draw.to_csv(index=False):
        raise ValueError(
            "Existing names gold manifest does not match the seeded draw; "
            "preserving human labels."
        )
    if template_path.exists():
        existing = pd.read_csv(template_path)
        _require_ein2_column(existing, template_path)
        expected = pd.DataFrame(
            {
                "EIN2": draw["EIN2"],
                "split": "names_gold",
                "text": draw["name_raw"],
            }
        )
        if (
            existing.columns.tolist() != _CODING_TEMPLATE_COLUMNS
            or existing["EIN2"].duplicated().any()
            or existing.set_index("EIN2")[["split", "text"]].to_dict("index")
            != expected.set_index("EIN2")[["split", "text"]].to_dict("index")
        ):
            raise ValueError(
                "Existing names gold coding template does not match the seeded draw; "
                "preserving human labels."
            )
        logger.info(
            "Names gold coding template exists at %s; preserving it.", template_path
        )
    else:
        template = pd.DataFrame(
            {
                "EIN2": draw["EIN2"],
                "split": "names_gold",
                "text": draw["name_raw"],
                "human_label": pd.NA,
            }
        )
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(template_path, index=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    draw.to_csv(manifest_path, index=False)
    registry.names_gold_coding_instructions.write_text(_CODING_RUBRIC + "\n")
