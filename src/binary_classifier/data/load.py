"""Load upstream missions data and join with BMF for NTEE major groups.

A missing upstream parquet is a hard error unless ``data.allow_synthetic`` is
explicitly enabled, in which case a temporary synthetic dataset (~1 000 rows)
is generated for smoke-testing — loudly warned and stamped
``data_source="synthetic"`` so a fake run can never masquerade as a real one
(audit D3). Synthetic temp dirs are cleaned up at process exit.
"""

import atexit
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from binary_classifier.config import BinaryClassifierConfig

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

_TRUNCATION_THRESHOLD = 1000

# Temp dirs created for synthetic data, cleaned up at process exit (audit D3:
# the old "caller may delete" comment had no caller).
_SYNTHETIC_DIRS: list[Path] = []


def _cleanup_synthetic_dirs() -> None:
    """Remove every synthetic temp dir created this process."""
    for d in _SYNTHETIC_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _SYNTHETIC_DIRS.clear()


atexit.register(_cleanup_synthetic_dirs)


# ── Public API ─────────────────────────────────────────────────────────────────


def load_missions(cfg: BinaryClassifierConfig) -> pd.DataFrame:
    """Load missions cross-section and join with BMF for NTEE major groups.

    The function reads ``missions_cross_section.parquet`` and joins it with
    ``bmf_unified_processed.parquet`` on ``EIN2`` to extract the NTEE major
    group (first character of ``NTEE_IRS``). Exact duplicates on the target
    text field are dropped (first kept). Rows with non-alphabetic NTEE codes
    are assigned ``'?'`` but excluded from downstream sampling pools.

    If either upstream parquet is missing, a temporary synthetic dataset is
    generated, used, and then its path is returned so the caller can clean it
    up if desired.

    Args:
        cfg: Validated configuration object.

    Returns:
        DataFrame with columns ``EIN2``, ``mission_text``,
        ``ntee_major_group``, ``is_truncated``, ``NTEE_IRS``, and
        ``data_source`` (``"upstream"`` or ``"synthetic"``).

    Raises:
        FileNotFoundError: If an upstream parquet is missing and
            ``data.allow_synthetic`` is ``False``.
    """
    paths = cfg.paths
    upstream_repo = Path(paths.upstream_repo).resolve()
    missions_path = upstream_repo / paths.missions_parquet
    bmf_path = upstream_repo / paths.bmf_parquet

    synthetic = False
    if not missions_path.exists() or not bmf_path.exists():
        if not cfg.data.allow_synthetic:
            raise FileNotFoundError(
                f"Upstream parquet not found (missions: {missions_path}; "
                f"bmf: {bmf_path}). Provide the NonProfitData parquet, or set "
                f"data.allow_synthetic: true in the config to generate "
                f"synthetic smoke-test data."
            )
        logger.warning(
            "[load] Upstream parquet missing — generating SYNTHETIC data. "
            "This is NOT a real run; set data.allow_synthetic: false for "
            "production."
        )
        missions_path, bmf_path = _generate_synthetic_parquets(cfg)
        synthetic = True

    df_missions = pd.read_parquet(missions_path)
    df_bmf = pd.read_parquet(bmf_path)

    # Join on EIN2 to get NTEE_IRS
    df = df_missions.merge(
        df_bmf[["EIN2", "NTEE_IRS"]],
        on="EIN2",
        how="left",
    )

    # Extract major group (first char, upper-case A-Z only)
    df["ntee_major_group"] = (
        df["NTEE_IRS"]
        .astype(str)
        .str[0]
        .str.upper()
        .where(lambda s: s.str.match(r"^[A-Z]$"), "?")
    )

    # Flag potentially soft-truncated text
    field = cfg.field
    df["is_truncated"] = df[field].astype(str).apply(_is_truncated)

    # Drop exact duplicates on the text field (keep first)
    df = df.drop_duplicates(subset=[field], keep="first")

    # Rename field to a canonical name for downstream consistency
    df = df.rename(columns={field: "mission_text"})

    # Stamp provenance so a synthetic build can never masquerade as a real one.
    df["data_source"] = "synthetic" if synthetic else "upstream"

    return df[
        [
            "EIN2",
            "mission_text",
            "ntee_major_group",
            "is_truncated",
            "NTEE_IRS",
            "data_source",
        ]
    ]


def _is_truncated(text: str, threshold: int = _TRUNCATION_THRESHOLD) -> bool:
    """Return True if the text ends abruptly near the threshold length.

    A mission is considered potentially truncated when its length is within
    5 characters of ``threshold`` and does not end with terminal punctuation.
    """
    if len(text) < threshold - 5:
        return False
    return not text.strip().endswith((".", "!", "?"))


# ── Synthetic-data fallback ────────────────────────────────────────────────────


def _generate_synthetic_parquets(
    cfg: BinaryClassifierConfig,
) -> tuple[Path, Path]:
    """Generate temporary synthetic parquet files for smoke testing.

    Creates ~1 000 synthetic missions with realistic NTEE codes and a
    matching BMF extract so the join succeeds.

    Returns:
        Tuple of (missions_parquet_path, bmf_parquet_path). Both files live
        in a temporary directory registered for cleanup at process exit.
    """
    rng = np.random.default_rng(seed=cfg.SEED)
    n = 1_000

    ntee_codes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ntee_weights = np.array(
        [
            56,
            94,
            13,
            16,
            32,
            13,
            10,
            4,
            8,
            5,
            8,
            23,
            11,
            43,
            19,
            74,
            13,
            4,
            22,
            26,
            3,
            1,
            8,
            41,
            1,
            4,
        ]
    )
    ntee_weights = ntee_weights / ntee_weights.sum()

    major_groups = rng.choice(ntee_codes, size=n, p=ntee_weights)
    ntee_irse = [f"{g}{rng.integers(10, 100):02d}" for g in major_groups]

    # Simple synthetic mission generator
    templates = [
        "to provide {service} for {beneficiary} in the {area}",
        "to {verb} {beneficiary} through {activity}",
        "our mission is to {verb} {service} and {activity} for {beneficiary}",
        "dedicated to {verb} {activity} that {outcome} for {beneficiary}",
        "to {verb} and {verb2} {beneficiary} with {service} and {activity}",
        "{org} exists to {verb} {beneficiary} by {activity}",
        "we are a {adj} organization that {verb} {service} for {beneficiary}",
    ]
    words = {
        "service": [
            "education",
            "housing",
            "healthcare",
            "food",
            "counseling",
            "support",
            "training",
            "scholarships",
            "relief",
            "aid",
        ],
        "beneficiary": [
            "children",
            "veterans",
            "families",
            "the homeless",
            "seniors",
            "youth",
            "patients",
            "refugees",
            "animals",
            "the community",
        ],
        "area": [
            "local community",
            "greater region",
            "state",
            "nation",
            "world",
            "city",
            "county",
        ],
        "verb": [
            "provide",
            "support",
            "promote",
            "serve",
            "educate",
            "deliver",
            "assist",
            "help",
            "empower",
            "strengthen",
        ],
        "verb2": [
            "advocate",
            "defend",
            "protect",
            "preserve",
            "advance",
            "improve",
            "enhance",
            "foster",
            "cultivate",
            "nurture",
        ],
        "activity": [
            "programs",
            "services",
            "outreach",
            "research",
            "advocacy",
            "training",
            "counseling",
            "camps",
            "events",
            "activities",
        ],
        "outcome": [
            "improves lives",
            "creates opportunity",
            "builds capacity",
            "reduces suffering",
            "advances equity",
            "promotes health",
            "strengthens community",
            "fosters growth",
        ],
        "org": [
            "the foundation",
            "our organization",
            "the coalition",
            "the initiative",
            "the center",
            "the institute",
            "the association",
            "the alliance",
        ],
        "adj": [
            "non-profit",
            "community-based",
            "faith-based",
            "secular",
            "grassroots",
            "national",
            "local",
            "regional",
        ],
    }

    missions = []
    for i in range(n):
        tmpl = rng.choice(templates)
        # 15 % chance of injecting a religious word for synthetic positives
        if rng.random() < 0.15:
            tmpl += " through christian education and bible study"
        mission = tmpl.format(
            **{k: rng.choice(v) for k, v in words.items()},
        )
        missions.append(mission)

    ein2s = [f"EIN-{i:02d}-{rng.integers(1_000_000, 10_000_000):07d}" for i in range(n)]

    df_missions = pd.DataFrame(
        {
            "EIN2": ein2s,
            "COMMON_LEVEL1": "501C3 CHARITY",
            "CANONICAL_MISSION": missions,
            "CANONICAL_MISSION_SOURCE_YEAR": rng.integers(2010, 2024, size=n),
            "LONGEST_MISSION": missions,
            "LONGEST_MISSION_SOURCE_YEAR": rng.integers(2010, 2024, size=n),
        }
    )

    df_bmf = pd.DataFrame(
        {
            "EIN2": ein2s,
            "NTEE_IRS": ntee_irse,
        }
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="binary_classifier_synthetic_"))
    _SYNTHETIC_DIRS.append(tmp_dir)
    missions_path = tmp_dir / "missions.parquet"
    bmf_path = tmp_dir / "bmf.parquet"
    df_missions.to_parquet(missions_path, index=False)
    df_bmf.to_parquet(bmf_path, index=False)

    return missions_path, bmf_path
