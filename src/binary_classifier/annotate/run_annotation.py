"""Batch annotation runner with resume support.

Runs the full model x prompt matrix over a dataset of (EIN2, text) records.
Resume is keyed by (EIN2, source_id) — fixes audit R-08. Checkpoints are
written to the long/tidy store after every ``checkpoint_every`` records.

The module can be imported and used programmatically, or invoked via
``scripts/03_annotate.py``.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

from binary_classifier.annotate.annotators import (
    Annotator,
    OpenAIAnnotator,
    VLLMAnnotator,
)
from binary_classifier.annotate.schema import AnnotationStore

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


# ── Canary set ───────────────────────────────────────────────────────────────

# Small fixed EIN2 list used for drift detection. Re-run whenever model IDs
# change to detect closed-model drift.
CANARY_EIN2: list[str] = [
    "00-0000001",  # synthetic placeholder — replace with real canary EIN2s
    "00-0000002",
    "00-0000003",
    "00-0000004",
    "00-0000005",
]


# ── Pipeline entrypoint ──────────────────────────────────────────────────────


def run_annotation(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    limit: int | None = None,
    prompt_paths: list[Path] | None = None,
    model_ids: list[str] | None = None,
    store_path: Path | None = None,
    checkpoint_every: int = 100,
    resume: bool = True,
    canary_only: bool = False,
) -> AnnotationStore:
    """Run the full annotation matrix over the silver pool (pipeline entrypoint).

    This is the canonical function called by ``scripts/run_pipeline.py``.
    """
    if prompt_paths is None:
        prompt_paths = [
            Path("src/binary_classifier/annotate/prompts/v1.txt"),
            Path("src/binary_classifier/annotate/prompts/v2.txt"),
            Path("src/binary_classifier/annotate/prompts/v3.txt"),
        ]

    if model_ids is None:
        model_ids = [
            cfg.model_slate.closed_reference,
            cfg.model_slate.open_production,
            cfg.model_slate.open_lightweight,
        ]

    if store_path is None:
        store_path = Path("data/annotation_store.csv")

    # Load silver pool
    silver_manifest = pd.read_csv(registry.silver_manifest)
    if silver_manifest.empty:
        logger.error("Silver manifest is empty or missing.")
        return AnnotationStore(store_path)

    # Join text field from upstream
    if "text" not in silver_manifest.columns:
        missions = pd.read_parquet(registry.missions_parquet)
        silver_manifest = silver_manifest.merge(
            missions[["EIN2", cfg.field]],
            on="EIN2",
            how="left",
        )
        silver_manifest = silver_manifest.rename(columns={cfg.field: "text"})

    if limit:
        silver_manifest = silver_manifest.head(limit)

    def annotator_factory(model_id: str, prompt_text: str) -> Annotator:
        if model_id.startswith("gpt"):
            return OpenAIAnnotator(
                model_id=model_id,
                prompt_id="",
                prompt_text=prompt_text,
                temperature=cfg.annotation.temperature,
                seed=cfg.annotation.seed,
            )
        return VLLMAnnotator(
            model_id=model_id,
            prompt_id="",
            prompt_text=prompt_text,
            temperature=cfg.annotation.temperature,
            seed=cfg.annotation.seed,
        )

    return run_annotation_matrix(
        df=silver_manifest,
        annotator_factory=annotator_factory,
        model_ids=model_ids,
        prompt_paths=prompt_paths,
        store_path=store_path,
        checkpoint_every=checkpoint_every,
        resume=resume,
        canary_only=canary_only,
    )


def run_annotation_matrix(
    df: pd.DataFrame,
    annotator_factory: Callable[[str, str], Annotator],
    model_ids: list[str],
    prompt_paths: list[Path],
    store_path: Path,
    checkpoint_every: int = 100,
    resume: bool = True,
    canary_only: bool = False,
) -> AnnotationStore:
    """Run the full model x prompt matrix over a dataframe.

    Args:
        df: DataFrame with columns ``EIN2`` and ``text`` (the field to classify).
        annotator_factory: Callable that takes (model_id, prompt_text) and
            returns an ``Annotator`` instance.
        model_ids: List of model identifiers to run.
        prompt_paths: List of prompt text files (e.g. ``v1.txt``, ``v2.txt``).
        store_path: Path to the long/tidy store (CSV or Parquet).
        checkpoint_every: Flush to disk every N records.
        resume: If ``True``, skip (EIN2, source_id) pairs already in the store.
        canary_only: If ``True``, run only the canary EIN2 set.

    Returns:
        The populated ``AnnotationStore``.

    """
    store = AnnotationStore(store_path)

    # Load prompt texts
    prompt_texts: dict[str, str] = {}
    for p in prompt_paths:
        prompt_texts[p.stem] = p.read_text()

    # Build the full work matrix
    work_items: list[tuple[str, str, str, str]] = []
    for model_id in model_ids:
        for prompt_id, prompt_text in prompt_texts.items():
            work_items.extend(
                [
                    (ein2, text, model_id, prompt_id)
                    for ein2, text in zip(df["EIN2"], df["text"])
                ],
            )

    if canary_only:
        canary_set = set(CANARY_EIN2)
        work_items = [(e, t, m, p) for e, t, m, p in work_items if e in canary_set]

    # Resume filtering
    if resume:
        existing_pairs = {
            (row["EIN2"], row["source_id"]) for _, row in store.to_frame().iterrows()
        }
        work_items = [
            (e, t, m, p)
            for e, t, m, p in work_items
            if (e, f"{m}__{p}") not in existing_pairs
        ]
        logger.info(
            "Resume: %d items already done, %d remaining",
            len(existing_pairs),
            len(work_items),
        )

    # Batch execution
    batch: list = []
    for idx, (ein2, text, model_id, prompt_id) in enumerate(work_items):
        annotator = annotator_factory(model_id, prompt_texts[prompt_id])
        record = annotator.annotate(text, ein2=ein2)
        batch.append(record)

        if (idx + 1) % checkpoint_every == 0:
            store.append_many(batch)
            logger.info(
                "Checkpoint %d/%d — wrote %d records",
                idx + 1,
                len(work_items),
                len(batch),
            )
            batch = []

    if batch:
        store.append_many(batch)
        logger.info("Final flush — wrote %d records", len(batch))

    return store


def build_json_schema() -> dict:
    """Return the JSON schema used for guided JSON / structured output.

    This schema is passed to both OpenAI ``response_format`` and vLLM
    ``guided_json`` so that every annotator produces identical field shapes.
    """
    return {
        "type": "object",
        "properties": {
            "binary_label": {
                "type": "string",
                "enum": [
                    "religious",
                    "nonreligious",
                    "ambiguous_review",
                    "insufficient_information",
                ],
            },
            "confidence": {"type": "number"},
            "domains_present": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence_spans": {
                "type": "array",
                "items": {"type": "string"},
            },
            "boundary_notes": {"type": "string"},
        },
        "required": ["binary_label", "confidence"],
    }
