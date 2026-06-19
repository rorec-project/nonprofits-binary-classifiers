"""Long/tidy label schema and pydantic JSON parser for the annotation pipeline.

Defines the ``LabelRecord`` pydantic model that is the canonical output of every
annotator, and the ``AnnotationStore`` helper that reads/writes the long/tidy
label table. The schema is designed to be weak-supervision-ready: one row per
(EIN2, source_id) so that multiple model x prompt labels can be aggregated later.
"""

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# ── Enums ────────────────────────────────────────────────────────────────────


class SourceType(StrEnum):
    """Origin of a label in the long/tidy store."""

    LLM_PROMPT = "llm_prompt"
    RULE = "rule"
    HUMAN = "human"


class BinaryLabel(StrEnum):
    """Source-of-truth label from the LLM codebook.

    ``religious`` → 1, ``nonreligious`` → 0, ``ambiguous_review`` and
    ``insufficient_information`` → abstain (NaN in the tidy store).
    """

    RELIGIOUS = "religious"
    NONRELIGIOUS = "nonreligious"
    AMBIGUOUS_REVIEW = "ambiguous_review"
    INSUFFICIENT_INFORMATION = "insufficient_information"


# ── JSON Schema builder ──────────────────────────────────────────────────────


def build_json_schema() -> dict[str, Any]:
    """Return a strict-compatible JSON Schema for LLM structured outputs.

    Compatible with OpenAI ``json_schema`` (``strict``: true) and vLLM
    ``guided_json``. All fields are in ``required``; nullable types use
    ``anyOf`` with ``{"type": "null"}``.
    """
    string_null: dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    string_array_null: dict[str, Any] = {
        "anyOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "null"},
        ],
    }

    return {
        "type": "object",
        "properties": {
            "binary_label": {
                "type": "string",
                "enum": [
                    BinaryLabel.RELIGIOUS.value,
                    BinaryLabel.NONRELIGIOUS.value,
                    BinaryLabel.AMBIGUOUS_REVIEW.value,
                    BinaryLabel.INSUFFICIENT_INFORMATION.value,
                ],
            },
            "confidence": {"type": "number"},
            "domains_present": string_array_null,
            "evidence_spans": string_array_null,
            "boundary_notes": string_null,
            "reason": string_null,
        },
        "required": [
            "binary_label",
            "confidence",
            "domains_present",
            "evidence_spans",
            "boundary_notes",
            "reason",
        ],
        "additionalProperties": False,
    }


# ── Pydantic model ───────────────────────────────────────────────────────────


def normalize_ein2(value: Any) -> Any:
    """Normalize annotation join keys for resume-safe comparisons."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if isinstance(value, float) and value.is_integer():
        return str(int(value)).strip()
    return str(value).strip()


def _csv_safe_text(value: str | None) -> str | None:
    """Encode free-text fields so CSV stores stay one physical row per record."""
    if value is None:
        return None
    return json.dumps(value)


def _restore_csv_safe_text(value: Any) -> Any:
    """Decode text written by ``_csv_safe_text`` while accepting legacy rows."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded if isinstance(decoded, str) else value


class LabelRecord(BaseModel):
    """Canonical record produced by every annotator.

    Fields map 1:1 to the long/tidy label-store columns. The ``binary_label``
    field is the source of truth; ``label`` is the numeric translation (1/0/NaN)
    used for model training.
    """

    # Identifiers
    EIN2: str = Field(..., description="IRS identifier (join key).")
    source_id: str = Field(
        ...,
        description="Unique run identifier: model_id + prompt_id.",
    )

    # Source metadata
    source_type: SourceType = Field(..., description="Origin of the label.")
    model_id: str = Field(..., description="Model name or API identifier.")
    prompt_id: str = Field(..., description="Prompt version (e.g. v1, v2, v3).")
    temperature: float = Field(..., description="Sampling temperature used.")
    seed: int | None = Field(None, description="Random seed used (if any).")
    run_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
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
        description="Annotator confidence (0-1).",
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

    # Optional system fingerprint from the provider response
    system_fingerprint: str | None = Field(
        None,
        description="OpenAI system_fingerprint or equivalent provider trace.",
    )

    # Raw LLM output
    raw_response: str | None = Field(
        None,
        description="Raw JSON string returned by the LLM.",
    )

    @field_validator("EIN2", mode="before")
    @classmethod
    def _normalize_ein2(cls, value: Any) -> Any:
        """Keep persisted annotation keys stable across CSV/parquet dtypes."""
        return normalize_ein2(value)

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
            "raw_response": _csv_safe_text(self.raw_response),
            "reason": _csv_safe_text(self.reason),
            "domains_present": json.dumps(self.domains_present)
            if self.domains_present is not None
            else None,
            "evidence_spans": json.dumps(self.evidence_spans)
            if self.evidence_spans is not None
            else None,
            "boundary_notes": _csv_safe_text(self.boundary_notes),
            "binary_label": self.binary_label.value if self.binary_label else None,
            "system_fingerprint": self.system_fingerprint,
        }

    @classmethod
    def from_flat_dict(cls, row: dict[str, Any]) -> "LabelRecord":
        """Rehydrate a ``LabelRecord`` from a flat CSV/Parquet row."""

        def _clean(val: Any) -> Any:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return val

        return cls(
            EIN2=normalize_ein2(row["EIN2"]),
            source_id=row["source_id"],
            source_type=SourceType(row["source_type"]),
            label=_clean(row.get("label")),
            confidence=_clean(row.get("confidence")),
            model_id=row["model_id"],
            prompt_id=row["prompt_id"],
            temperature=row["temperature"],
            seed=_clean(row.get("seed")),
            run_timestamp=(
                datetime.fromisoformat(row["run_timestamp"])
                if row.get("run_timestamp")
                else datetime.now(UTC)
            ),
            raw_response=_restore_csv_safe_text(row.get("raw_response")),
            reason=_restore_csv_safe_text(row.get("reason")),
            domains_present=(
                json.loads(row["domains_present"])
                if isinstance(row.get("domains_present"), str)
                and row["domains_present"]
                else None
            ),
            evidence_spans=(
                json.loads(row["evidence_spans"])
                if isinstance(row.get("evidence_spans"), str) and row["evidence_spans"]
                else None
            ),
            boundary_notes=_restore_csv_safe_text(row.get("boundary_notes")),
            binary_label=(
                BinaryLabel(row["binary_label"])
                if isinstance(row.get("binary_label"), str) and row["binary_label"]
                else None
            ),
            system_fingerprint=_clean(row.get("system_fingerprint")),
        )


# ── Store helpers ────────────────────────────────────────────────────────────


class AnnotationStore:
    """Read/write the long/tidy label store as CSV or Parquet.

    The store is append-only for CSV (new rows are written directly without
    loading the full file). Resuming is done by checking the set of
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
        "system_fingerprint",
    ]

    def __init__(self, path: Path) -> None:
        """Initialise the store.

        Args:
            path: Path to the CSV or Parquet file.

        """
        self.path: Path = path
        self._df: pd.DataFrame | None = None
        self._done_set: set[tuple[str, str]] | None = None

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
            # Reindex to canonical columns for backward compatibility
            for col in self.COLUMNS:
                if col not in self._df.columns:
                    self._df[col] = None
            self._df = self._df[self.COLUMNS]
            self._df["EIN2"] = self._df["EIN2"].map(normalize_ein2)
        else:
            self._df = pd.DataFrame(columns=self.COLUMNS)
        return self._df

    def _save(self) -> None:
        """Persist the backing dataframe."""
        df = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.suffix == ".parquet":
            df.to_parquet(self.path, index=False)
        else:
            df.to_csv(self.path, index=False)

    def _build_done_set(self) -> set[tuple[str, str]]:
        """Build a cached set of (EIN2, source_id) pairs from the store."""
        if self._done_set is not None:
            return self._done_set
        if not self.path.exists():
            self._done_set = set()
            return self._done_set
        if self.path.suffix == ".parquet":
            df = pd.read_parquet(self.path, columns=["EIN2", "source_id"])
        else:
            df = pd.read_csv(self.path, usecols=["EIN2", "source_id"])
        self._done_set = set(
            zip(df["EIN2"].map(normalize_ein2), df["source_id"], strict=True)
        )
        return self._done_set

    # ── Public API ───────────────────────────────────────────────────────

    def already_done(self, ein2: str, source_id: str) -> bool:
        """Return ``True`` if the pair (EIN2, source_id) is already stored.

        This is the resume key — fixes audit R-08.
        """
        return (normalize_ein2(ein2), source_id) in self._build_done_set()

    def done_pairs(self) -> set[tuple[str, str]]:
        """Return a copy of the set of (EIN2, source_id) pairs already stored."""
        return self._build_done_set().copy()

    def append(self, record: LabelRecord) -> None:
        """Append a single record to the store."""
        row = record.to_flat_dict()
        row_df = pd.DataFrame([row], columns=self.COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.suffix == ".parquet":
            df = self._load()
            self._df = pd.concat([df, row_df], ignore_index=True)
            self._save()
        else:
            if self.path.exists():
                row_df.to_csv(self.path, mode="a", index=False, header=False)
            else:
                row_df.to_csv(self.path, index=False)
            if self._df is not None:
                self._df = pd.concat([self._df, row_df], ignore_index=True)
        if self._done_set is not None:
            self._done_set.add((normalize_ein2(row["EIN2"]), row["source_id"]))

    def append_many(self, records: list[LabelRecord]) -> None:
        """Append a batch of records efficiently."""
        if not records:
            return
        rows = [r.to_flat_dict() for r in records]
        rows_df = pd.DataFrame(rows, columns=self.COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.suffix == ".parquet":
            df = self._load()
            self._df = pd.concat([df, rows_df], ignore_index=True)
            self._save()
        else:
            if self.path.exists():
                rows_df.to_csv(self.path, mode="a", index=False, header=False)
            else:
                rows_df.to_csv(self.path, index=False)
            if self._df is not None:
                self._df = pd.concat([self._df, rows_df], ignore_index=True)
        if self._done_set is not None:
            self._done_set.update(
                (normalize_ein2(r["EIN2"]), r["source_id"]) for r in rows
            )

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
        rows = df[df["EIN2"] == normalize_ein2(ein2)].to_dict("records")
        return [LabelRecord.from_flat_dict(r) for r in rows]
