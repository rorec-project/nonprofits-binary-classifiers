"""Evidence-span verification: guard against fabricated LLM citations.

Each stored ``evidence_span`` is checked against the upstream source text
(joined via ``EIN2``). Fabricated spans are flagged and, optionally, positive
labels that carry them can be downgraded to abstain before aggregation.
"""

import json
import logging

import pandas as pd

from binary_classifier.annotate.schema import BinaryLabel
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def verify_evidence_spans(registry: PathRegistry, store_df: pd.DataFrame) -> dict:
    """Verify that every stored evidence span is a verbatim substring of the source text.

    Args:
        registry: Path registry with resolved ``missions_parquet`` path.
        store_df: DataFrame from ``AnnotationStore.to_frame()``.

    Returns:
        Dict with ``total_spans``, ``verified_spans``, ``fabricated_spans``,
        ``fabrication_rate``, and ``fabricated_records`` (list of
        ``(EIN2, source_id, span)`` tuples).

    Raises:
        FileNotFoundError: If the missions parquet is missing.
        ValueError: If the missions parquet lacks required columns.

    """
    missions_path = registry.missions_parquet
    if not missions_path.exists():
        raise FileNotFoundError(
            f"Missions parquet not found at {missions_path}. "
            f"Run upstream data generation first.",
        )

    missions = pd.read_parquet(missions_path)
    text_col = registry.cfg.field

    if "EIN2" not in missions.columns:
        raise ValueError("Missions parquet missing EIN2 column")
    if text_col not in missions.columns:
        raise ValueError(f"Missions parquet missing text column '{text_col}'")

    text_map = missions.set_index("EIN2")[text_col].fillna("").astype(str).to_dict()

    total_spans = 0
    verified_spans = 0
    fabricated_spans = 0
    fabricated_records: list[tuple[str, str, str]] = []

    for _, row in store_df.iterrows():
        ein2 = row["EIN2"]
        source_id = row["source_id"]
        raw_spans = row.get("evidence_spans")

        if pd.isna(raw_spans):
            continue

        spans: list[str] | None = None
        if isinstance(raw_spans, str):
            spans = json.loads(raw_spans)
        elif isinstance(raw_spans, list):
            spans = raw_spans

        if not spans:
            continue

        source_text = text_map.get(ein2, "")

        for span in spans:
            total_spans += 1
            if span in source_text:
                verified_spans += 1
            else:
                fabricated_spans += 1
                fabricated_records.append((ein2, source_id, span))

    fabrication_rate = fabricated_spans / total_spans if total_spans > 0 else 0.0

    return {
        "total_spans": total_spans,
        "verified_spans": verified_spans,
        "fabricated_spans": fabricated_spans,
        "fabrication_rate": fabrication_rate,
        "fabricated_records": fabricated_records,
    }


def abstain_fabricated_positives(
    store_df: pd.DataFrame,
    fabricated_records: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """Downgrade positive labels with fabricated evidence spans to abstain.

    Sets ``label`` to NaN and ``binary_label`` to ``None`` for rows where
    ``binary_label == "religious"`` and the record has a fabricated span.

    Args:
        store_df: DataFrame from ``AnnotationStore.to_frame()``.
        fabricated_records: List of ``(EIN2, source_id, span)`` tuples.

    Returns:
        Modified DataFrame (copy).

    """
    df = store_df.copy()
    fabricated_set = {(e, s) for e, s, _ in fabricated_records}

    mask = df.apply(
        lambda row: (
            (row["EIN2"], row["source_id"]) in fabricated_set
            and row.get("binary_label") == BinaryLabel.RELIGIOUS.value
        ),
        axis=1,
    )

    if mask.any():
        n = int(mask.sum())
        logger.info("Abstaining %d fabricated positive record(s)", n)
        df.loc[mask, "label"] = float("nan")
        df.loc[mask, "binary_label"] = None

    return df
