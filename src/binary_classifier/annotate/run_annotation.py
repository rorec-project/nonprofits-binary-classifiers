"""Batch annotation runner with resume support.

Runs the confirmed production model set × prompt matrix over the silver pool.
Resume is keyed by (EIN2, source_id) — fixes audit R-08 and D1: the real
``prompt_id`` (prompt-file stem) is threaded through the annotator factory so
``source_id = f"{model_id}__{prompt_id}"`` matches the resume key, keeping the
three prompt variants distinct in the store.

The module can be imported and used programmatically, or invoked via
``scripts/03_annotate.py``.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

from binary_classifier.annotate.annotators import Annotator
from binary_classifier.annotate.annotators.factory import make_annotator
from binary_classifier.annotate.schema import AnnotationStore
from binary_classifier.config import BakeoffCandidate, load_slate

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

# Default prompt variants (model × prompt matrix).
_DEFAULT_PROMPT_PATHS: list[Path] = [
    Path("src/binary_classifier/annotate/prompts/v1.txt"),
    Path("src/binary_classifier/annotate/prompts/v2.txt"),
    Path("src/binary_classifier/annotate/prompts/v3.txt"),
]

# Factory signature: (spec, prompt_id, prompt_text) -> Annotator.
AnnotatorFactory = Callable[[BakeoffCandidate, str, str], Annotator]


# ── Confirmed-slate resolution (stage-03 backstop, gate G2) ──────────────────


def resolve_production_specs(registry: "PathRegistry") -> list[BakeoffCandidate]:
    """Resolve the human-confirmed production model set.

    This is the stage-03 internal backstop behind gate G2: stage 03 refuses to
    annotate unless a confirmed ``production_slate.json`` exists and lists at
    least one model.

    Args:
        registry: Path registry with the resolved ``production_slate`` path.

    Returns:
        The list of confirmed model specs.

    Raises:
        FileNotFoundError: If no production slate exists.
        ValueError: If the slate is unconfirmed or lists no models.

    """
    path = registry.production_slate
    if not path.exists():
        raise FileNotFoundError(
            f"No confirmed production slate at {path}. Run stage 02, review "
            f"{registry.bakeoff_results}, then copy {registry.proposed_slate} "
            f"to {path} and set 'confirmed': true."
        )
    slate = load_slate(path)
    if not slate.confirmed:
        raise ValueError(
            f"{path} is not confirmed. Review the bake-off scores and set "
            f"'confirmed': true before running stage 03."
        )
    if not slate.models:
        raise ValueError(f"{path} lists no models under 'models'.")
    return slate.models


# ── Pipeline entrypoint ──────────────────────────────────────────────────────


def run_annotation(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    limit: int | None = None,
    prompt_paths: list[Path] | None = None,
    specs: list[BakeoffCandidate] | None = None,
    store_path: Path | None = None,
    checkpoint_every: int | None = None,
    resume: bool = True,
    canary_only: bool = False,
) -> AnnotationStore:
    """Run the production matrix over the silver pool (pipeline entrypoint).

    This is the canonical function called by ``scripts/run_pipeline.py``. The
    model set defaults to the human-confirmed production slate; the store path
    defaults to ``registry.annotation_store``.
    """
    if prompt_paths is None:
        prompt_paths = list(_DEFAULT_PROMPT_PATHS)
    if specs is None:
        specs = resolve_production_specs(registry)
    if store_path is None:
        store_path = registry.annotation_store
    if checkpoint_every is None:
        checkpoint_every = cfg.annotation.checkpoint_every

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

    def annotator_factory(
        spec: BakeoffCandidate,
        prompt_id: str,
        prompt_text: str,
    ) -> Annotator:
        return make_annotator(cfg, spec, prompt_id, prompt_text)

    return run_annotation_matrix(
        df=silver_manifest,
        specs=specs,
        prompt_paths=prompt_paths,
        store_path=store_path,
        annotator_factory=annotator_factory,
        checkpoint_every=checkpoint_every,
        resume=resume,
        canary_only=canary_only,
    )


def run_annotation_matrix(
    df: pd.DataFrame,
    specs: list[BakeoffCandidate],
    prompt_paths: list[Path],
    store_path: Path,
    annotator_factory: AnnotatorFactory,
    checkpoint_every: int = 100,
    resume: bool = True,
    canary_only: bool = False,
) -> AnnotationStore:
    """Run the full model x prompt matrix over a dataframe.

    Args:
        df: DataFrame with columns ``EIN2`` and ``text`` (the field to classify).
        specs: Model specs to run (each tagged with its provider).
        prompt_paths: List of prompt text files (e.g. ``v1.txt``, ``v2.txt``).
        store_path: Path to the long/tidy store (CSV or Parquet).
        annotator_factory: Callable ``(spec, prompt_id, prompt_text) ->
            Annotator``. Injected for testability.
        checkpoint_every: Flush to disk every N records.
        resume: If ``True``, skip (EIN2, source_id) pairs already in the store.
        canary_only: If ``True``, run only the canary EIN2 set.

    Returns:
        The populated ``AnnotationStore``.

    """
    store = AnnotationStore(store_path)

    # Load prompt texts (keyed by file stem = the real prompt_id).
    prompt_texts: dict[str, str] = {p.stem: p.read_text() for p in prompt_paths}

    # Build the full work matrix: (ein2, text, spec, prompt_id).
    work_items: list[tuple[str, str, BakeoffCandidate, str]] = []
    for spec in specs:
        for prompt_id in prompt_texts:
            work_items.extend(
                (ein2, text, spec, prompt_id)
                for ein2, text in zip(df["EIN2"], df["text"])
            )

    if canary_only:
        canary_set = set(CANARY_EIN2)
        work_items = [w for w in work_items if w[0] in canary_set]

    # Resume filtering — source_id == f"{spec.id}__{prompt_id}".
    if resume:
        existing_pairs = store.done_pairs()
        work_items = [
            (e, t, s, p)
            for e, t, s, p in work_items
            if (e, f"{s.id}__{p}") not in existing_pairs
        ]
        logger.info(
            "Resume: %d items already done, %d remaining",
            len(existing_pairs),
            len(work_items),
        )

    # Batch execution
    batch: list = []
    for idx, (ein2, text, spec, prompt_id) in enumerate(work_items):
        annotator = annotator_factory(spec, prompt_id, prompt_texts[prompt_id])
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
