"""Draw and gate the BMF-only human-coded names gold sample.

The draw deliberately excludes panel organizations and manifest-contaminated rows
so human labels estimate the names arm's target population without reusing mission
training or evaluation records. It preserves the mission-label construct while
oversampling documented ambiguity cases for coder review.
"""

from __future__ import annotations

import logging
import re
from hashlib import sha256
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

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
_SAINT_NAME_PATTERN = re.compile(r"\b(?:st\.?|saint)\s+\w+", re.IGNORECASE)
_CONFLICT_PATTERNS = {
    "faith_heritage": re.compile(
        r"\b(?:faith|grace|trinity|covenant|heritage)\b", re.IGNORECASE
    ),
    "non_christian_tradition": re.compile(
        r"\b(?:buddh(?:ist|a)?|gurdwara|hindu|islam(?:ic)?|jain|masjid|mosque|"
        r"muslim|sikh|synagogue|temple)\b",
        re.IGNORECASE,
    ),
    "non_english_name": re.compile(
        r"\b(?:amigos|asociacion|associazione|associacao|centro|comunidad|"
        r"cultura|culturelle|ecole|escuela|fondation|fundacion|gemeinde|iglesia|"
        r"societe|sociedad|verein)\b|[^\x00-\x7f]",
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
_CONFLICT_CATEGORIES = frozenset({"saint_name", *_CONFLICT_PATTERNS})


# ── Gold draw and human-label gate ────────────────────────────────────────────
def draw_name_gold(cfg: BinaryClassifierConfig, registry: PathRegistry) -> None:
    """Draw a seeded, stratified BMF-only names sample and coding template.

    The BMF-only source isolates the coverage population absent from the panel.
    External flags allocate adequate representation without changing the unchanged
    mission-purpose coding rubric written with the human-label template.
    """
    frame = pd.read_parquet(registry.names_bmf_only_frame)
    _require_columns(frame)
    quotas = _validate_quotas(cfg)
    # Keep the draw on the target BMF-only population and outside mission manifests.
    eligible = frame.loc[
        frame["is_bmf_only"].astype(bool)
        & frame["population"].eq("bmf_only")
        & ~frame["is_manifest_contaminated"].astype(bool)
    ].copy()
    if eligible["EIN2"].duplicated().any():
        raise ValueError("BMF-only gold frame must contain one row per EIN2.")
    eligible["gold_stratum"] = _strata(eligible)
    eligible["conflict_categories"] = eligible.apply(_conflict_categories, axis=1)
    seed = int(cfg.names.gold_seed if cfg.names.gold_seed is not None else cfg.SEED)
    draw = _sample(eligible, quotas, cfg.names.gold_conflict_quotas, seed)
    conflict_counts = _conflict_counts(draw)
    draw["conflict_categories"] = draw["conflict_categories"].map(
        lambda categories: "|".join(categories) if categories else "none"
    )
    _write_artifacts(draw, registry)
    logger.info(
        "Drew %d BMF-only names gold rows (seed=%d): strata=%s conflicts=%s",
        len(draw),
        seed,
        draw["gold_stratum"].value_counts().sort_index().to_dict(),
        conflict_counts,
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
    template = pd.read_csv(template_path, dtype={"EIN2": "string"})
    manifest = pd.read_csv(manifest_path, dtype={"EIN2": "string"})
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
    if "name_raw" not in manifest:
        raise ValueError(f"{manifest_path} is missing required column: name_raw")
    template_by_ein2 = template.set_index("EIN2")
    manifest_by_ein2 = manifest.set_index("EIN2")
    if (
        not template_by_ein2["split"].eq("names_gold").all()
        or not template_by_ein2["text"]
        .eq(manifest_by_ein2.loc[template_by_ein2.index, "name_raw"])
        .all()
    ):
        raise ValueError(
            "Names gold coding template text and split must match the manifest."
        )
    labels = pd.to_numeric(template["human_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Names gold coding is incomplete; human_label must be 0 or 1.")
    instructions = registry.names_gold_coding_instructions
    if not instructions.exists() or _CODING_RUBRIC not in instructions.read_text():
        raise ValueError("Names gold coding instructions are missing or altered.")


# ── Quota validation and seeded joint allocation ──────────────────────────────
def _validate_quotas(cfg: BinaryClassifierConfig) -> dict[str, int]:
    quotas = cfg.names.gold_stratum_quotas
    if set(quotas) != _STRATA or any(count < 0 for count in quotas.values()):
        raise ValueError(
            f"gold_stratum_quotas must define nonnegative counts for {_STRATA}."
        )
    if sum(quotas.values()) != cfg.names.gold_sample_size:
        raise ValueError("gold_stratum_quotas must sum to gold_sample_size.")
    if set(cfg.names.gold_conflict_quotas) != _CONFLICT_CATEGORIES or any(
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
    """Allocate and sample reproducible BMF-only stratum/conflict cells jointly.

    Joint allocation keeps configured external-flag representation while ensuring
    ambiguity cases receive their requested oversampling. Seeded priorities make
    repeated draws stable without changing the probability recorded per cell.
    """
    eligible = eligible.copy()
    eligible["conflict_mask"] = eligible["conflict_categories"].map(
        lambda categories: "|".join(categories) if categories else "none"
    )
    grouped = [
        (cast(tuple[str, str], key), frame)
        for key, frame in eligible.groupby(["gold_stratum", "conflict_mask"], sort=True)
    ]
    keys = [key for key, _ in grouped]
    sizes = np.array([len(frame) for _, frame in grouped], dtype=float)
    constraints: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for stratum in sorted(_STRATA):
        constraints.append(
            np.array([float(key[0] == stratum) for key in keys], dtype=float)
        )
        lower.append(float(quotas[stratum]))
        upper.append(float(quotas[stratum]))
    for category in sorted(_CONFLICT_CATEGORIES):
        constraints.append(
            np.array(
                [float(category in key[1].split("|")) for key in keys], dtype=float
            )
        )
        lower.append(float(conflict_quotas[category]))
        upper.append(np.inf)
    objective = np.array(
        [_seeded_priority(seed, f"{stratum}\0{mask}") for stratum, mask in keys]
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(keys)),
        bounds=Bounds(np.zeros(len(keys)), sizes),
        constraints=LinearConstraint(np.vstack(constraints), lower, upper),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "Configured stratum and conflict quotas are jointly infeasible for "
            "the BMF-only names frame."
        )
    samples: list[pd.DataFrame] = []
    for (stratum, mask), frame, count in zip(
        keys,
        (frame for _, frame in grouped),
        np.rint(result.x).astype(int),
        strict=True,
    ):
        if count == 0:
            continue
        ordered = frame.assign(
            _priority=frame["EIN2"].map(lambda ein2: _seeded_priority(seed, str(ein2)))
        ).sort_values(["_priority", "EIN2"])
        sample = ordered.iloc[:count].copy()
        sample["sampling_cell"] = f"{stratum}|{mask}"
        sample["sampling_cell_population"] = len(frame)
        sample["inclusion_probability"] = count / len(frame)
        samples.append(sample.drop(columns="_priority"))
    return (
        pd.concat(samples, ignore_index=True).sort_values("EIN2").reset_index(drop=True)
    )


def _seeded_priority(seed: int, value: str) -> float:
    digest = sha256(f"{seed}\0{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# ── Conflict enrichment diagnostics ───────────────────────────────────────────
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


def _conflict_categories(row: pd.Series) -> list[str]:
    """Return review-enrichment cues available before human coding.

    Saint names enter the conflict quota only when both external auspice flags are
    absent. This is a pre-coding secular proxy, not a claim about the gold label.
    Non-English cues are multilingual lexical indicators rather than language ID.
    """
    text = str(row["name_raw"])
    categories = [
        category
        for category, pattern in _CONFLICT_PATTERNS.items()
        if pattern.search(text) is not None
    ]
    if (
        _SAINT_NAME_PATTERN.search(text) is not None
        and not bool(row["is_ntee_x"])
        and not bool(row["is_church_foundation"])
    ):
        categories.insert(0, "saint_name")
    return categories


def _conflict_counts(draw: pd.DataFrame) -> dict[str, int]:
    """Count realized quota memberships, allowing rows in several categories."""
    return {
        category: int(
            draw["conflict_categories"]
            .map(lambda categories: category in categories)
            .sum()
        )
        for category in sorted(_CONFLICT_CATEGORIES)
    }


# ── Artifact validation and persistence ───────────────────────────────────────
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
        existing = pd.read_csv(template_path, dtype={"EIN2": "string"})
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
