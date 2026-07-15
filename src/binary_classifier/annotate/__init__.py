"""Annotation core package.

Public exports:
    * ``LabelRecord`` – canonical pydantic label record.
    * ``AnnotationStore`` – long/tidy CSV/Parquet store with resume support.
    * ``SourceType``, ``BinaryLabel`` – enums used by the schema.
"""

from binary_classifier.annotate.schema import (
    AnnotationStore,
    BinaryLabel,
    LabelRecord,
    SourceType,
)

__all__ = [
    "AnnotationStore",
    "BinaryLabel",
    "LabelRecord",
    "SourceType",
]
