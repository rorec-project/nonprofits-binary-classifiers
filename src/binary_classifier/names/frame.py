"""Build the panel and BMF-only name cross-sections for cross-field transfer."""

import logging
from typing import TYPE_CHECKING

import pandas as pd

from binary_classifier.names.identifiers import normalize_ein2

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig, NamesExpectedCounts
    from binary_classifier.paths import PathRegistry


logger = logging.getLogger(__name__)

_MANIFEST_PATH_NAMES = (
    "silver_manifest",
    "gold_manifest",
    "prompt_dev_manifest",
    "validation_manifest",
    "test_manifest",
    "monitor_manifest",
    "anchor_manifest",
)
_PANEL_REQUIRED_COLUMNS = {"EIN2", "COMMON_LEVEL1", "BEST_NAME_CASED"}
_PANEL_BARE_REQUIRED_COLUMNS = {"EIN2", "BEST_NAME_BARE_CASED"}
_MISSIONS_REQUIRED_COLUMNS = {"EIN2", "LONGEST_MISSION"}
_BMF_REQUIRED_COLUMNS = {
    "EIN2",
    "ORG_NAME_CURRENT",
    "NTEE_IRS",
    "BMF_FOUNDATION_CODE",
}
_FRAME_COLUMNS = [
    "EIN2",
    "population",
    "name_raw",
    "name_raw_source",
    "name_cased",
    "name_bare",
    "dba_cased",
    "has_dba",
    "has_mission",
    "is_name_only",
    "is_bmf_only",
    "is_manifest_contaminated",
    "NTEE_IRS",
    "BMF_FOUNDATION_CODE",
    "ntee_major_group",
    "is_ntee_x",
    "is_church_foundation",
    "is_external_religious_flag",
]


def build_name_frame(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
) -> None:
    """Build isolated panel and BMF-only name frames keyed by ``EIN2``.

    The BMF anti-join deliberately uses the full panel universe. The emitted panel
    frame then applies the project's narrower 501(c)(3) charity scope.
    """
    panel = _load_panel(registry)
    missions = _load_missions(registry)
    bmf = _load_bmf(registry)
    contaminated_ein2s = _load_manifest_ein2s(registry)

    _assert_panel_bmf_coverage(panel, bmf)
    panel_ein2s = set(panel["EIN2"])
    panel_frame, panel_counts = _build_panel_frame(
        panel,
        missions,
        bmf,
        contaminated_ein2s,
    )
    bmf_only_frame, bmf_only_counts = _build_bmf_only_frame(
        bmf,
        panel_ein2s,
        contaminated_ein2s,
    )
    _assert_disjoint_frames(panel_frame, bmf_only_frame)
    observed_counts = {**panel_counts, **bmf_only_counts}
    _log_frame_counts(panel_frame, bmf_only_frame, observed_counts)
    _validate_expected_counts(cfg.names.expected_counts, observed_counts)

    registry.ensure_dirs()
    panel_frame.to_parquet(registry.names_panel_frame, index=False)
    bmf_only_frame.to_parquet(registry.names_bmf_only_frame, index=False)


def _load_panel(registry: "PathRegistry") -> pd.DataFrame:
    panel = pd.read_parquet(registry.panel_final_parquet)
    bare_names = pd.read_parquet(registry.panel_filled_gaps_parquet)
    _require_columns(panel, _PANEL_REQUIRED_COLUMNS, registry.panel_final_parquet)
    _require_columns(
        bare_names,
        _PANEL_BARE_REQUIRED_COLUMNS,
        registry.panel_filled_gaps_parquet,
    )
    panel_columns = ["EIN2", "COMMON_LEVEL1", "BEST_NAME_CASED"]
    panel_columns.extend(
        column
        for column in (
            "F9_00_ORG_NAME_L1",
            "NAME_CASED",
            "BEST_DBA_CASED",
            "HAS_DBA",
        )
        if column in panel.columns
    )
    panel = _collapse_panel(panel[panel_columns])
    bare_names = _collapse_panel(bare_names[["EIN2", "BEST_NAME_BARE_CASED"]])
    return panel.merge(bare_names, on="EIN2", how="left", validate="one_to_one")


def _load_missions(registry: "PathRegistry") -> pd.DataFrame:
    missions = pd.read_parquet(registry.missions_parquet)
    _require_columns(missions, _MISSIONS_REQUIRED_COLUMNS, registry.missions_parquet)
    missions = missions[["EIN2", "LONGEST_MISSION"]].copy()
    missions["EIN2"] = normalize_ein2(missions["EIN2"])
    _assert_unique_ein2(missions, "missions")
    missions["has_mission"] = _has_text(missions["LONGEST_MISSION"])
    return missions[["EIN2", "has_mission"]]


def _load_bmf(registry: "PathRegistry") -> pd.DataFrame:
    bmf = pd.read_parquet(registry.bmf_parquet)
    _require_columns(bmf, _BMF_REQUIRED_COLUMNS, registry.bmf_parquet)
    bmf = bmf.copy()
    bmf["EIN2"] = normalize_ein2(bmf["EIN2"])
    _assert_unique_ein2(bmf, "BMF")
    ntee_major_group = bmf["NTEE_IRS"].astype("string").str.strip().str.upper().str[0]
    bmf["ntee_major_group"] = ntee_major_group.where(
        ntee_major_group.str.fullmatch("[A-Z]"),
        "?",
    )
    bmf["is_ntee_x"] = bmf["ntee_major_group"].eq("X")
    bmf["is_church_foundation"] = pd.to_numeric(
        bmf["BMF_FOUNDATION_CODE"], errors="coerce"
    ).eq(10)
    bmf["is_external_religious_flag"] = bmf["is_ntee_x"] | bmf["is_church_foundation"]
    return bmf


def _build_panel_frame(
    panel: pd.DataFrame,
    missions: pd.DataFrame,
    bmf: pd.DataFrame,
    contaminated_ein2s: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    scoped = panel.loc[panel["COMMON_LEVEL1"].eq("501C3 CHARITY")].copy()
    scoped = scoped.merge(missions, on="EIN2", how="left", validate="one_to_one")
    scoped = scoped.merge(
        _bmf_frame_columns(bmf), on="EIN2", how="left", validate="one_to_one"
    )
    scoped["has_mission"] = scoped["has_mission"].fillna(False).astype(bool)
    name_raw_column = _panel_raw_name_column(scoped)
    has_name = _has_text(scoped[name_raw_column])
    name_only = ~scoped["has_mission"] & has_name
    counts = {
        "panel_has_mission": int(scoped["has_mission"].sum()),
        "panel_name_only": int(name_only.sum()),
        "panel_no_name_no_mission": int((~scoped["has_mission"] & ~has_name).sum()),
        "panel_name_only_flagged": int(
            scoped.loc[name_only, "is_external_religious_flag"].sum()
        ),
    }
    return _finalize_frame(
        scoped,
        name_raw_column=name_raw_column,
        name_raw_source=name_raw_column,
        name_cased_column="BEST_NAME_CASED",
        population="panel_501c3",
        contaminated_ein2s=contaminated_ein2s,
        is_bmf_only=False,
    ), counts


def _build_bmf_only_frame(
    bmf: pd.DataFrame,
    panel_ein2s: set[str],
    contaminated_ein2s: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    bmf_only = bmf.loc[~bmf["EIN2"].isin(panel_ein2s)].copy()
    bmf_only["has_mission"] = False
    counts = {
        "bmf_only": len(bmf_only),
        "bmf_only_flagged": int(bmf_only["is_external_religious_flag"].sum()),
    }
    return _finalize_frame(
        bmf_only,
        name_raw_column="ORG_NAME_CURRENT",
        name_raw_source="ORG_NAME_CURRENT",
        name_cased_column=None,
        population="bmf_only",
        contaminated_ein2s=contaminated_ein2s,
        is_bmf_only=True,
    ), counts


def _finalize_frame(
    frame: pd.DataFrame,
    *,
    name_raw_column: str,
    name_raw_source: str,
    name_cased_column: str | None,
    population: str,
    contaminated_ein2s: set[str],
    is_bmf_only: bool,
) -> pd.DataFrame:
    has_name = _has_text(frame[name_raw_column])
    missing_count = int((~has_name).sum())
    logger.info("Excluded %d %s rows without usable names.", missing_count, population)
    result = frame.loc[has_name].copy()
    result["name_raw"] = result[name_raw_column]
    result["name_raw_source"] = name_raw_source
    result["name_cased"] = (
        result[name_cased_column] if name_cased_column is not None else pd.NA
    )
    result["name_bare"] = result.get("BEST_NAME_BARE_CASED", pd.NA)
    result["dba_cased"] = result.get("BEST_DBA_CASED", pd.NA)
    result["has_dba"] = result.get("HAS_DBA", False)
    result["population"] = population
    result["is_name_only"] = ~result["has_mission"] & (not is_bmf_only)
    result["is_bmf_only"] = is_bmf_only
    result["is_manifest_contaminated"] = result["EIN2"].isin(contaminated_ein2s)
    _assert_unique_ein2(result, population)
    return result[_FRAME_COLUMNS]


def _bmf_frame_columns(bmf: pd.DataFrame) -> pd.DataFrame:
    return bmf[
        [
            "EIN2",
            "NTEE_IRS",
            "BMF_FOUNDATION_CODE",
            "ntee_major_group",
            "is_ntee_x",
            "is_church_foundation",
            "is_external_religious_flag",
        ]
    ]


def _panel_raw_name_column(panel: pd.DataFrame) -> str:
    """Select the raw panel-name field without substituting canonical variants."""
    for column in ("F9_00_ORG_NAME_L1", "NAME_CASED"):
        if column in panel.columns:
            return column
    raise ValueError("Panel requires F9_00_ORG_NAME_L1 or NAME_CASED for raw names")


def _collapse_panel(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["EIN2"] = normalize_ein2(frame["EIN2"])
    _assert_nonempty_ein2(frame, "panel")
    varying = frame.groupby("EIN2", dropna=False).nunique(dropna=False)
    conflicting = varying.gt(1).any(axis=1)
    if bool(conflicting.any()):
        ein2s = varying.index[conflicting].tolist()
        raise ValueError(f"Panel fields vary within EIN2: {ein2s[:5]}")
    return frame.drop_duplicates(subset="EIN2", keep="first")


def _load_manifest_ein2s(registry: "PathRegistry") -> set[str]:
    manifests: list[pd.Series] = []
    for attribute in _MANIFEST_PATH_NAMES:
        path = getattr(registry, attribute)
        if not path.exists():
            continue
        manifest = pd.read_csv(path, usecols=["EIN2"])
        manifests.append(normalize_ein2(manifest["EIN2"]))
    if not manifests:
        return set()
    return set(pd.concat(manifests, ignore_index=True).dropna())


def _assert_disjoint_frames(panel: pd.DataFrame, bmf_only: pd.DataFrame) -> None:
    overlap = set(panel["EIN2"]).intersection(bmf_only["EIN2"])
    if overlap:
        raise ValueError(f"Panel and BMF-only frames overlap: {sorted(overlap)[:5]}")


def _assert_panel_bmf_coverage(panel: pd.DataFrame, bmf: pd.DataFrame) -> None:
    missing = sorted(set(panel["EIN2"]).difference(bmf["EIN2"]))
    if missing:
        raise ValueError(
            f"Panel EIN2 values missing from BMF: {', '.join(missing[:5])}"
        )


def _assert_unique_ein2(frame: pd.DataFrame, source: str) -> None:
    _assert_nonempty_ein2(frame, source)
    duplicated = frame["EIN2"].duplicated(keep=False)
    if bool(duplicated.any()):
        ein2s = frame.loc[duplicated, "EIN2"].tolist()
        raise ValueError(f"{source} must contain one row per EIN2: {ein2s[:5]}")


def _assert_nonempty_ein2(frame: pd.DataFrame, source: str) -> None:
    missing = frame["EIN2"].isna() | frame["EIN2"].eq("")
    if bool(missing.any()):
        raise ValueError(f"{source} contains missing EIN2 values")


def _require_columns(frame: pd.DataFrame, required: set[str], source: object) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _has_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().notna() & values.astype(
        "string"
    ).str.strip().ne("")


def _log_frame_counts(
    panel: pd.DataFrame,
    bmf_only: pd.DataFrame,
    observed_counts: dict[str, int],
) -> None:
    logger.info(
        "Names source strata: panel_has_mission=%d panel_name_only=%d "
        "panel_no_name_no_mission=%d panel_name_only_flagged=%d bmf_only=%d "
        "bmf_only_flagged=%d.",
        observed_counts["panel_has_mission"],
        observed_counts["panel_name_only"],
        observed_counts["panel_no_name_no_mission"],
        observed_counts["panel_name_only_flagged"],
        observed_counts["bmf_only"],
        observed_counts["bmf_only_flagged"],
    )
    logger.info(
        "Wrote panel name frame: %d rows (%d mission-having, %d name-only).",
        len(panel),
        int(panel["has_mission"].sum()),
        int(panel["is_name_only"].sum()),
    )
    _log_external_flag_count("panel name frame", panel)
    logger.info("Wrote BMF-only name frame: %d rows.", len(bmf_only))
    _log_external_flag_count("BMF-only name frame", bmf_only)


def _validate_expected_counts(
    expected_counts: "NamesExpectedCounts | None",
    observed_counts: dict[str, int],
) -> None:
    if expected_counts is None:
        return
    mismatches = [
        f"{name}: expected {expected}, observed {observed_counts[name]}"
        for name, expected in expected_counts.model_dump().items()
        if observed_counts[name] != expected
    ]
    if mismatches:
        raise ValueError(
            "Name-frame count reconciliation failed: " + "; ".join(mismatches)
        )


def _log_external_flag_count(frame_name: str, frame: pd.DataFrame) -> None:
    flagged = int(frame["is_external_religious_flag"].sum())
    rate = flagged / len(frame) if len(frame) else 0.0
    logger.info(
        "%s external religious flag: %d/%d (%.1f%%).",
        frame_name,
        flagged,
        len(frame),
        rate * 100,
    )
