"""Long/tidy label schema and pydantic JSON parser for the annotation pipeline.

Defines the ``LabelRecord`` pydantic model that is the canonical output of every
annotator, and the ``AnnotationStore`` helper that reads/writes the long/tidy
label table. The schema is designed to be weak-supervision-ready: one row per
(EIN2, source_id) so that multiple model x prompt labels can be aggregated later.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────────────────


class SourceType(str, Enum):
    """Origin of a label in the long/tidy store."""

    LLM_PROMPT = "llm_prompt"
    RULE = "rule"
    HUMAN = "human"


class BinaryLabel(str, Enum):
    """Source-of-truth label from the LLM codebook.

    ``religious`` → 1, ``nonreligious`` → 0, ``ambiguous_review`` and
    ``insufficient_information`` → abstain (NaN in the tidy store).
    """

    RELIGIOUS = "religious"
    NONRELIGIOUS = "nonreligious"
    AMBIGUOUS_REVIEW = "ambiguous_review"
    INSUFFICIENT_INFORMATION = "insufficient_information"


# ── Pydantic model ───────────────────────────────────────────────────────────


class LabelRecord(BaseModel):
    """Canonical record produced by every annotator.

    Fields map 1:1 to the long/tidy label-store columns. The ``binary_label``
    field is the source of truth; ``label`` is the numeric translation (1/0/NaN)
    used for model training.
    """

    # Identifiers
    EIN2: str = Field(..., description="IRS identifier (join key).")
    source_id: str = Field(
        ..., description="Unique run identifier: model_id + prompt_id."
    )

    # Source metadata
    source_type: SourceType = Field(..., description="Origin of the label.")
    model_id: str = Field(..., description="Model name or API identifier.")
    prompt_id: str = Field(..., description="Prompt version (e.g. v1, v2, v3).")
    temperature: float = Field(..., description="Sampling temperature used.")
    seed: int | None = Field(None, description="Random seed used (if any).")
    run_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the annotation.",
    )

    # Label content
    binary_label: BinaryLabel | None = Field(
        None,
        description="Codebook source-of-truth label.",
    )
    label: float | None = Field(
        None,
        description="Numeric label: 1=religious, 0=nonreligious, NaN=abstain.",
    )
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Annotator confidence (0–1).",
    )
    reason: str | None = Field(
        None,
        description="Short reasoning string from the annotator.",
    )
    domains_present: list[str] | None = Field(
        None,
        description="Domain codes present in the text (e.g. 'faith_tradition').",
    )
    evidence_spans: list[str] | None = Field(
        None,
        description="Verbatim text spans that justify the label.",
    )
    boundary_notes: str | None = Field(
        None,
        description="Notes on edge cases or ambiguities.",
    )

    # Raw LLM output
    raw_response: str | None = Field(
        None,
        description="Raw JSON string returned by the LLM.",
    )

    # ── Computed fields ────────────────────────────────────────────────────

    def compute_numeric_label(self) -> float | None:
        """Translate ``binary_label`` into numeric training label.

        Returns:
            ``1.0`` for religious, ``0.0`` for nonreligious, ``None`` for
            abstain (ambiguous_review or insufficient_information).

        """
        mapping = {
            BinaryLabel.RELIGIOUS: 1.0,
            BinaryLabel.NONRELIGIOUS: 0.0,
        }
        if self.binary_label is None:
            return None
        return mapping.get(self.binary_label)

    def model_post_init(self, __context: Any) -> None:
        """Auto-fill ``label`` from ``binary_label`` if not already set."""
        if self.label is None:
            self.label = self.compute_numeric_label()

    # ── Serialization helpers ──────────────────────────────────────────────

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for CSV/Parquet rows.

        Lists are JSON-serialised to strings so that every column is scalar.
        """
        return {
            "EIN2": self.EIN2,
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "label": self.label,
            "confidence": self.confidence,
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "temperature": self.temperature,
            "seed": self.seed,
            "run_timestamp": self.run_timestamp.isoformat(),
            "raw_response": self.raw_response,
            "reason": self.reason,
            "domains_present": json.dumps(self.domains_present)
            if self.domains_present is not None
            else None,
            "evidence_spans": json.dumps(self.evidence_spans)
            if self.evidence_spans is not None
            else None,
            "boundary_notes": self.boundary_notes,
            "binary_label": self.binary_label.value if self.binary_label else None,
        }

    @classmethod
    def from_flat_dict(cls, row: dict[str, Any]) -> "LabelRecord":
        """Rehydrate a ``LabelRecord`` from a flat CSV/Parquet row."""
        return cls(
            EIN2=row["EIN2"],
            source_id=row["source_id"],
            source_type=SourceType(row["source_type"]),
            label=row.get("label"),
            confidence=row.get("confidence"),
            model_id=row["model_id"],
            prompt_id=row["prompt_id"],
            temperature=row["temperature"],
            seed=row.get("seed"),
            run_timestamp=(
                datetime.fromisoformat(row["run_timestamp"])
                if row.get("run_timestamp")
                else datetime.now(timezone.utc)
            ),
            raw_response=row.get("raw_response"),
            reason=row.get("reason"),
            domains_present=json.loads(row["domains_present"])
            if row.get("domains_present")
            else None,
            evidence_spans=json.loads(row["evidence_spans"])
            if row.get("evidence_spans")
            else None,
            boundary_notes=row.get("boundary_notes"),
            binary_label=BinaryLabel(row["binary_label"])
            if row.get("binary_label")
            else None,
        )


# ── Store helpers ────────────────────────────────────────────────────────────


class AnnotationStore:
    """Read/write the long/tidy label store as CSV or Parquet.

    The store is append-only. Resuming is done by checking the set of
    (EIN2, source_id) pairs already present.
    """

    # Exact column order for the tidy store
    COLUMNS: list[str] = [
        "EIN2",
        "source_id",
        "source_type",
        "label",
        "confidence",
        "model_id",
        "prompt_id",
        "temperature",
        "seed",
        "run_timestamp",
        "raw_response",
        "reason",
        "domains_present",
        "evidence_spans",
        "boundary_notes",
        "binary_label",
    ]

    def __init__(self, path: Path) -> None:
        """Initialise the store.

        Args:
            path: Path to the CSV or Parquet file.

        """
        self.path: Path = path
        self._df: pd.DataFrame | None = None

    # ── Internal helpers ───────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        """Lazy-load the backing dataframe."""
        if self._df is not None:
            return self._df
        if self.path.exists():
            if self.path.suffix == ".parquet":
                self._df = pd.read_parquet(self.path)
            else:
                self._df = pd.read_csv(self.path)
        else:
            self._df = pd.DataFrame(columns=self.COLUMNS)
        return self._df

    def _save(self) -> None:
        """Persist the backing dataframe."""
        df = self._load()
        if self.path.suffix == ".parquet":
            df.to_parquet(self.path, index=False)
        else:
            df.to_csv(self.path, index=False)

    # ── Public API ───────────────────────────────────────────────────────

    def already_done(self, ein2: str, source_id: str) -> bool:
        """Return ``True`` if the pair (EIN2, source_id) is already stored.

        This is the resume key — fixes audit R-08.
        """
        df = self._load()
        if df.empty:
            return False
        mask = (df["EIN2"] == ein2) & (df["source_id"] == source_id)
        return bool(mask.any())

    def append(self, record: LabelRecord) -> None:
        """Append a single record to the store."""
        df = self._load()
        row = record.to_flat_dict()
        row_df = pd.DataFrame([row], columns=self.COLUMNS)
        self._df = pd.concat([df, row_df], ignore_index=True)
        self._save()

    def append_many(self, records: list[LabelRecord]) -> None:
        """Append a batch of records efficiently."""
        if not records:
            return
        df = self._load()
        rows = [r.to_flat_dict() for r in records]
        rows_df = pd.DataFrame(rows, columns=self.COLUMNS)
        self._df = pd.concat([df, rows_df], ignore_index=True)
        self._save()

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the full store as a pandas DataFrame."""
        return self._load().copy()

    def unique_source_ids(self) -> list[str]:
        """Return the distinct source_ids already in the store."""
        df = self._load()
        if df.empty:
            return []
        return df["source_id"].dropna().unique().tolist()

    def records_for_ein2(self, ein2: str) -> list[LabelRecord]:
        """Return all records for a given EIN2."""
        df = self._load()
        rows = df[df["EIN2"] == ein2].to_dict("records")
        return [LabelRecord.from_flat_dict(r) for r in rows]
