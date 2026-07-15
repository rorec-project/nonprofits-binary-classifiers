"""Batch annotation runner with resume support (stage 03).

Runs the confirmed production model set x prompt matrix over the silver pool.
Resume is keyed by (EIN2, source_id) — fixes audit R-08 and D1: the real
``prompt_id`` (prompt-file stem) is threaded through the annotator factory so
``source_id = f"{model_id}__{prompt_id}"`` matches the resume key, keeping the
three prompt variants distinct in the store.

The design follows the LLM-as-annotator validation-first workflow: human-coded
labels remain the reference standard, and LLM labels are treated as a weak-
supervision source that must be validated before downstream use (Gilardi,
Alizadeh & Kubli 2023, https://doi.org/10.1073/pnas.2305016120; Pangakis,
Wolken & Fasching 2023, https://doi.org/10.48550/arXiv.2306.00176; Ziems et
al. 2024, https://doi.org/10.1162/coli_a_00502).  Prompt variants are kept
distinct in the store so inter-prompt disagreement can serve as an uncertainty
feature (Pangakis & Wolken 2025, https://doi.org/10.1609/icwsm.v19i1.35883).

The module can be imported and used programmatically, or invoked via
``scripts/03_annotate.py``.

References:
    - Gilardi, Alizadeh & Kubli (2023), "ChatGPT Outperforms Crowd Workers for
      Text-Annotation Tasks", PNAS. https://doi.org/10.1073/pnas.2305016120
    - Pangakis, Wolken & Fasching (2023), "Automated Annotation with
      Generative AI Requires Validation", arXiv.
      https://doi.org/10.48550/arXiv.2306.00176
    - Ziems et al. (2024), "Can Large Language Models Transform Computational
      Social Science?", Computational Linguistics.
      https://doi.org/10.1162/coli_a_00502
    - Pangakis & Wolken (2025), "Keeping Humans in the Loop: Human-Centered
      Automated Annotation with Generative AI", ICWSM.
      https://doi.org/10.1609/icwsm.v19i1.35883

"""

import concurrent.futures
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from binary_classifier.annotate.aggregate import aggregate_labels
from binary_classifier.annotate.annotators import Annotator
from binary_classifier.annotate.annotators.factory import make_annotator
from binary_classifier.annotate.annotators.openai_annotator import OpenAIAnnotator
from binary_classifier.annotate.concurrency import (
    ProviderLimiters,
    annotate_with_provider_limit,
    build_provider_limiters,
)
from binary_classifier.annotate.schema import (
    AnnotationStore,
    LabelRecord,
    normalize_ein2,
)
from binary_classifier.config import BakeoffCandidate, Slate, load_slate

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


# ── Canary set ───────────────────────────────────────────────────────────────


CANARY_AUDIT_FILENAME = "canary_drift_audit.jsonl"


def load_canary_ein2s(registry: "PathRegistry") -> set[str]:
    """Load the monitor-slice EIN2s that define the canary audit set.

    Stage 01 writes the held-out monitor manifest beside the other interim
    manifests. Stage 03 uses that real monitor slice for canary/drift checks so
    the synthetic placeholder list cannot accidentally tune prompts on toy rows
    or touch the frozen test set.

    Args:
        registry: Path registry that resolves ``monitor_manifest``.

    Returns:
        Set of monitor ``EIN2`` identifiers as strings.

    Raises:
        FileNotFoundError: If stage 01 has not produced the monitor manifest.
        ValueError: If the manifest does not expose the required ``EIN2``
            column.

    """
    path = registry.monitor_manifest
    if not path.exists():
        raise FileNotFoundError(
            f"No monitor manifest at {path}. run stage 01 first; monitor "
            "manifest lives under the cloud-symlinked interim tree.",
        )

    monitor = pd.read_csv(path)
    if "EIN2" not in monitor.columns:
        raise ValueError(f"{path} missing required EIN2 column.")
    return set(monitor["EIN2"].dropna().map(normalize_ein2))


# Factory signature: (spec, prompt_id, prompt_text) -> Annotator.
AnnotatorFactory = Callable[[BakeoffCandidate, str, str], Annotator]


# ── Confirmed-slate resolution (stage-03 backstop, gate G2) ──────────────────


def _load_confirmed_production_slate(registry: "PathRegistry") -> Slate:
    """Load and validate the human-confirmed production slate."""
    path = registry.production_slate
    if not path.exists():
        raise FileNotFoundError(
            f"No confirmed production slate at {path}. Run stage 02, review "
            f"{registry.bakeoff_results}, then copy {registry.proposed_slate} "
            f"to {path} and set 'confirmed': true.",
        )
    slate = load_slate(path)
    if not slate.confirmed:
        raise ValueError(
            f"{path} is not confirmed. Review the bake-off scores and set "
            f"'confirmed': true before running stage 03.",
        )
    if not slate.models:
        raise ValueError(f"{path} lists no models under 'models'.")
    return slate


def _selected_prompt_pairs(slate: Slate) -> set[tuple[str, str]] | None:
    """Return selected ``(model_id, prompt_id)`` pairs when present."""
    if not slate.selected:
        return None
    model_ids = {model.id for model in slate.models}
    pairs: set[tuple[str, str]] = set()
    invalid: list[dict[str, object]] = []
    for item in slate.selected:
        model_id = item.get("model_id") or item.get("id")
        prompt_id = item.get("prompt_id")
        if model_id in model_ids and prompt_id:
            pairs.add((str(model_id), str(prompt_id)))
        else:
            invalid.append(dict(item))
    if invalid:
        raise ValueError(
            "production_slate.selected contains invalid model/prompt entries: "
            f"{invalid[:3]}",
        )
    return pairs


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
    return _load_confirmed_production_slate(registry).models


def resolve_production_selection(
    registry: "PathRegistry",
) -> tuple[list[BakeoffCandidate], set[tuple[str, str]] | None]:
    """Resolve confirmed models plus optional selected model-prompt pairs.

    Args:
        registry: Path registry with the resolved ``production_slate`` path.

    Returns:
        Tuple of (confirmed model specs, optional selected model-prompt pairs).

    Raises:
        FileNotFoundError: If no production slate exists.
        ValueError: If the slate is unconfirmed or lists no models.

    """
    slate = _load_confirmed_production_slate(registry)
    return slate.models, _selected_prompt_pairs(slate)


# ── Pipeline entrypoint ──────────────────────────────────────────────────────
#
# ``run_annotation`` is the canonical entrypoint for stage 03, called by both
# ``scripts/03_annotate.py`` and the orchestrator.  It resolves the confirmed
# production slate (gate G2), builds the silver+gold annotation pool, and
# delegates to the matrix runner.  Gold rows are included because the stage-04
# validation gate and the monitor canary both compare LLM labels against human
# labels; annotating silver alone would leave those joins empty on independent
# draws.


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
    """Run the production matrix over silver plus gold rows.

    This is the canonical function called by ``scripts/run_pipeline.py``. The
    model set defaults to the human-confirmed production slate; the store path
    defaults to ``registry.annotation_store``. Gold rows are included because
    the stage-04 validation gate and the monitor canary both compare LLM labels
    against human-coded gold rows; annotating silver alone can leave those joins
    empty on realistic independent draws.

    When ``canary_only`` is true, the run is monitoring-only: it loads the
    held-out monitor manifest, leaves the freeze gate and frozen test set
    untouched, and must not be used for prompt tuning. The matrix runner owns
    the final filter so direct programmatic calls get the same hard-fail
    behaviour when the monitor slice does not overlap the annotation pool.

    Args:
        cfg: Validated task configuration.
        registry: Path registry with resolved artifact paths.
        limit: Optional row cap for the annotation pool (for smoke tests).
        prompt_paths: Optional explicit prompt files; defaults to the slate's
            selected prompts or the legacy ``v1``, ``v2``, ``v3`` set.
        specs: Optional explicit model specs; defaults to the confirmed
            production slate.
        store_path: Optional explicit annotation-store path.
        checkpoint_every: Flush to disk every N records.
        resume: Skip (EIN2, source_id) pairs already present in the store.
        canary_only: Run only the monitor-manifest canary slice for drift
            audit.

    Returns:
        The populated ``AnnotationStore``.

    Raises:
        FileNotFoundError: If the production slate or monitor manifest is
            missing when required.
        ValueError: If the canary monitor slice does not overlap the pool.

    """
    selected_pairs: set[tuple[str, str]] | None = None
    if specs is None:
        specs, selected_pairs = resolve_production_selection(registry)
    if prompt_paths is None:
        prompt_ids: tuple[str, ...] | list[str]
        if selected_pairs is None:
            prompt_ids = ("v1", "v2", "v3")
        else:
            prompt_ids = sorted({prompt_id for _, prompt_id in selected_pairs})
        prompt_paths = [registry.prompts_dir / f"{stem}.txt" for stem in prompt_ids]
    if store_path is None:
        store_path = registry.annotation_store
    if checkpoint_every is None:
        checkpoint_every = cfg.annotation.checkpoint_every

    # Load the annotation pool from the union of silver and gold manifests.
    silver_manifest = pd.read_csv(registry.silver_manifest)
    gold_manifest = pd.read_csv(registry.gold_manifest)
    silver_manifest["EIN2"] = silver_manifest["EIN2"].map(normalize_ein2)
    gold_manifest["EIN2"] = gold_manifest["EIN2"].map(normalize_ein2)

    # Validation and monitor rows live in gold, not necessarily silver. The QC
    # gate and canary therefore need the EIN2 union before text is joined.
    annotation_manifest = (
        pd.concat(
            [silver_manifest[["EIN2"]], gold_manifest[["EIN2"]]],
            ignore_index=True,
        )
        .drop_duplicates(subset=["EIN2"])
        .reset_index(drop=True)
    )
    if annotation_manifest.empty:
        logger.error("Annotation manifest is empty or missing.")
        return AnnotationStore(store_path)

    # Join text field from upstream
    if "text" not in annotation_manifest.columns:
        missions = pd.read_parquet(registry.missions_parquet)
        missions = missions[["EIN2", cfg.field]].copy()
        missions["EIN2"] = missions["EIN2"].map(normalize_ein2)
        annotation_manifest = annotation_manifest.merge(
            missions,
            on="EIN2",
            how="left",
        )
        annotation_manifest = annotation_manifest.rename(columns={cfg.field: "text"})

    if limit:
        annotation_manifest = annotation_manifest.head(limit)
    use_openai_batch = bool(
        cfg.annotation.openai_batch and limit is None and not canary_only,
    )
    if cfg.annotation.openai_batch and not use_openai_batch:
        logger.info("OpenAI batch mode disabled for limited/canary annotation run")

    def annotator_factory(
        spec: BakeoffCandidate,
        prompt_id: str,
        prompt_text: str,
    ) -> Annotator:
        return make_annotator(cfg, spec, prompt_id, prompt_text)

    return run_annotation_matrix(
        df=annotation_manifest,
        specs=specs,
        prompt_paths=prompt_paths,
        store_path=store_path,
        annotator_factory=annotator_factory,
        checkpoint_every=checkpoint_every,
        resume=resume,
        selected_pairs=selected_pairs,
        canary_only=canary_only,
        cfg=cfg,
        registry=registry,
        use_openai_batch=use_openai_batch,
    )


def run_annotation_matrix(
    df: pd.DataFrame,
    specs: list[BakeoffCandidate],
    prompt_paths: list[Path],
    store_path: Path,
    annotator_factory: AnnotatorFactory,
    checkpoint_every: int = 100,
    resume: bool = True,
    selected_pairs: set[tuple[str, str]] | None = None,
    canary_only: bool = False,
    cfg: "BinaryClassifierConfig | None" = None,
    registry: "PathRegistry | None" = None,
    provider_limiters: ProviderLimiters | None = None,
    use_openai_batch: bool | None = None,
) -> AnnotationStore:
    """Run the full model x prompt matrix over a dataframe.

    The optional canary mode is a monitor-only drift run sourced from the
    held-out monitor manifest. It is excluded from the stage-04 freeze gate,
    should never drive prompt tuning, and does not touch the frozen test set.

    Args:
        df: DataFrame with columns ``EIN2`` and ``text`` (the field to classify).
        specs: Model specs to run (each tagged with its provider).
        prompt_paths: List of prompt text files (e.g. ``v1.txt``, ``v2.txt``).
        store_path: Path to the long/tidy store (CSV or Parquet).
        annotator_factory: Callable ``(spec, prompt_id, prompt_text) ->
            Annotator``. Injected for testability.
        checkpoint_every: Flush to disk every N records.
        resume: If ``True``, skip (EIN2, source_id) pairs already in the store.
        selected_pairs: Optional confirmed ``(model_id, prompt_id)`` subset from
            the bake-off slate. ``None`` preserves the legacy full cartesian
            model x prompt run.
        canary_only: If ``True``, run only monitor-manifest canary rows and
            append a drift audit record.
        cfg: Configuration needed for canary fingerprint metadata.
        registry: Path registry needed to load the monitor manifest and write
            the canary audit history.
        provider_limiters: Optional provider semaphores. When omitted and
            ``cfg`` is supplied, built from ``cfg.annotation``.
        use_openai_batch: Optional override for routing OpenAI groups through
            the Batch API. ``None`` follows ``cfg.annotation.openai_batch``.

    Returns:
        The populated ``AnnotationStore``.

    """
    if not resume and store_path.exists():
        store_path.unlink()
        logger.info(
            "Start-from-scratch: removed existing annotation store %s",
            store_path,
        )

    store = AnnotationStore(
        store_path,
        persist_raw_response=False,
        persist_text=True,
    )
    df = df.copy()
    df["EIN2"] = df["EIN2"].map(normalize_ein2)
    if provider_limiters is None and cfg is not None:
        provider_limiters = build_provider_limiters(cfg)

    # Load prompt texts (keyed by file stem = the real prompt_id).
    prompt_texts: dict[str, str] = {p.stem: p.read_text() for p in prompt_paths}

    canary_ein2s: set[str] | None = None
    canary_pool_ein2s: set[str] | None = None
    if canary_only:
        if cfg is None or registry is None:
            raise ValueError(
                "canary_only requires cfg and registry so stage 03 can load "
                "the held-out monitor manifest and record drift metadata.",
            )

        canary_ein2s = load_canary_ein2s(registry)
        pool_ein2s = set(df["EIN2"].dropna().map(normalize_ein2))
        canary_pool_ein2s = canary_ein2s & pool_ein2s
        if not canary_pool_ein2s:
            raise ValueError(
                "Canary monitor EIN2s do not overlap the annotation pool; "
                "run stage 01 and stage 03 on the same manifests before "
                "requesting --canary.",
            )
        if canary_pool_ein2s != canary_ein2s:
            logger.warning(
                "Canary monitor manifest has %d EIN2s but only %d are in the "
                "annotation pool",
                len(canary_ein2s),
                len(canary_pool_ein2s),
            )

        # The canary is a held-out monitor slice, not validation or test. Keep
        # this filter inside the matrix runner so all stage-03 entrypoints share
        # the same no-silent-no-op guard.
        df = df[df["EIN2"].astype(str).isin(canary_pool_ein2s)].copy()

    # Resume filtering — source_id == f"{spec.id}__{prompt_id}".
    existing_pairs: set[tuple[str, str]] = set()
    if resume:
        existing_pairs = store.done_pairs()

    if selected_pairs is not None:
        available_pairs = {
            (spec.id, prompt_id) for spec in specs for prompt_id in prompt_texts
        }
        missing_pairs = selected_pairs - available_pairs
        if missing_pairs:
            formatted = ", ".join(
                f"{model_id}__{prompt_id}"
                for model_id, prompt_id in sorted(missing_pairs)
            )
            raise ValueError(
                f"Production slate selected prompt pairs are unavailable: {formatted}",
            )

    # Build work groups by (spec, prompt_id).  Each group uses one annotator
    # instance (created once per pair, not once per row).
    groups: list[tuple[BakeoffCandidate, str, str, list[tuple[str, str]]]] = []
    for spec in specs:
        for prompt_id in prompt_texts:
            if (
                selected_pairs is not None
                and (spec.id, prompt_id) not in selected_pairs
            ):
                continue
            prompt_text = prompt_texts[prompt_id]
            source_id = f"{spec.id}__{prompt_id}"
            rows: list[tuple[str, str]] = [
                (ein2, text)
                for ein2, text in zip(df["EIN2"], df["text"], strict=True)
                if (ein2, source_id) not in existing_pairs
            ]
            if rows:
                groups.append((spec, prompt_id, prompt_text, rows))

    if resume:
        logger.info(
            "Resume: %d pairs already done, %d remaining",
            len(existing_pairs),
            sum(len(rows) for _, _, _, rows in groups),
        )

    total_items = sum(len(rows) for _, _, _, rows in groups)
    openai_batch_groups: list[
        tuple[BakeoffCandidate, str, str, list[tuple[str, str]]]
    ] = []
    live_groups = groups
    batch_enabled = (
        bool(cfg.annotation.openai_batch and not canary_only)
        if use_openai_batch is None and cfg
        else False
    )
    if use_openai_batch is not None:
        batch_enabled = use_openai_batch
    if batch_enabled:
        openai_batch_groups = [
            group for group in groups if group[0].provider == "openai"
        ]
        live_groups = [group for group in groups if group[0].provider != "openai"]
        if openai_batch_groups:
            logger.info(
                "OpenAI Batch API enabled for %d Stage 03 model×prompt groups",
                len(openai_batch_groups),
            )

    # Parallel execution: one thread per live (spec, prompt_id) group. OpenAI
    # batch groups are submitted through the provider's asynchronous Batch API.
    run_records: list[LabelRecord] = []
    if total_items > 0:
        store_lock = threading.Lock()
        progress_lock = threading.Lock()
        run_records_lock = threading.Lock()
        completed = 0

        def process_group(
            spec: BakeoffCandidate,
            prompt_id: str,
            prompt_text: str,
            rows: list[tuple[str, str]],
        ) -> list[LabelRecord]:
            nonlocal completed
            annotator = annotator_factory(spec, prompt_id, prompt_text)
            batch: list[LabelRecord] = []
            local_records: list[LabelRecord] = []
            local_count = 0

            for ein2, text in rows:
                record = annotate_with_provider_limit(
                    annotator,
                    spec.provider,
                    text,
                    ein2,
                    provider_limiters,
                )
                batch.append(record)
                local_records.append(record)
                local_count += 1

                if local_count % checkpoint_every == 0:
                    with store_lock:
                        store.append_many(batch)
                    with progress_lock:
                        completed += len(batch)
                        logger.info(
                            "Checkpoint %d/%d — wrote %d records",
                            completed,
                            total_items,
                            len(batch),
                        )
                    batch = []

            if batch:
                with store_lock:
                    store.append_many(batch)
                with progress_lock:
                    completed += len(batch)
                    logger.info(
                        "Checkpoint %d/%d — wrote %d records",
                        completed,
                        total_items,
                        len(batch),
                    )

            return local_records

        if openai_batch_groups:
            assert cfg is not None
            for spec, prompt_id, prompt_text, rows in openai_batch_groups:
                local_records = _run_openai_batch_group(
                    spec=spec,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    rows=rows,
                    store_path=store_path,
                    annotator_factory=annotator_factory,
                    poll_seconds=cfg.annotation.openai_batch_poll_seconds,
                    completion_window=cfg.annotation.openai_batch_completion_window,
                )
                with store_lock:
                    store.append_many(local_records)
                with progress_lock:
                    completed += len(local_records)
                    logger.info(
                        "OpenAI batch checkpoint %d/%d — wrote %d records",
                        completed,
                        total_items,
                        len(local_records),
                    )
                with run_records_lock:
                    run_records.extend(local_records)

        if live_groups:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(live_groups),
            ) as executor:
                futures = [
                    executor.submit(process_group, spec, prompt_id, prompt_text, rows)
                    for spec, prompt_id, prompt_text, rows in live_groups
                ]
                for future in concurrent.futures.as_completed(futures):
                    local_records = future.result()
                    with run_records_lock:
                        run_records.extend(local_records)

    else:
        logger.info("All work items already done — nothing to annotate.")

    if canary_only and cfg is not None and registry is not None:
        _record_canary_audit(
            cfg=cfg,
            registry=registry,
            specs=specs,
            prompt_ids=list(prompt_texts),
            canary_ein2s=canary_ein2s or set(),
            canary_pool_ein2s=canary_pool_ein2s or set(),
            run_records=run_records,
            store=store,
        )

    return store


# ── OpenAI Batch API path ─────────────────────────────────────────────────────
#
# Stage 02 intentionally remains on live calls because the prompt-dev bake-off is
# small and needs immediate scored feedback. Stage 03 can optionally batch only
# OpenAI-backed production groups while vLLM groups continue through the live
# concurrency-limited path.


def _run_openai_batch_group(
    *,
    spec: BakeoffCandidate,
    prompt_id: str,
    prompt_text: str,
    rows: list[tuple[str, str]],
    store_path: Path,
    annotator_factory: AnnotatorFactory,
    poll_seconds: int,
    completion_window: Literal["24h"],
) -> list[LabelRecord]:
    """Submit, poll, and parse one OpenAI model×prompt batch group."""
    annotator = annotator_factory(spec, prompt_id, prompt_text)
    if not isinstance(annotator, OpenAIAnnotator):
        raise TypeError(
            "OpenAI batch mode requires the OpenAIAnnotator factory path for "
            f"{spec.id}__{prompt_id}.",
        )

    source_id = f"{spec.id}__{prompt_id}"
    batch_dir = store_path.parent / "openai_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    safe_source = _safe_batch_stem(source_id)
    requests_path = batch_dir / f"{safe_source}.requests.jsonl"
    output_path = batch_dir / f"{safe_source}.output.jsonl"
    errors_path = batch_dir / f"{safe_source}.errors.jsonl"
    metadata_path = batch_dir / f"{safe_source}.batch.json"

    custom_to_ein2: dict[str, str] = {}
    with requests_path.open("w", encoding="utf-8") as handle:
        for ein2, text in rows:
            custom_id = f"{source_id}::{ein2}"
            custom_to_ein2[custom_id] = ein2
            request = annotator.build_batch_request(text, custom_id)
            handle.write(json.dumps(request, sort_keys=True) + "\n")

    expected_custom_ids = set(custom_to_ein2)
    if _batch_outputs_cover(output_path, errors_path, expected_custom_ids):
        logger.info("Reusing existing OpenAI batch output for %s", source_id)
    else:
        _submit_and_download_openai_batch(
            annotator=annotator,
            requests_path=requests_path,
            output_path=output_path,
            errors_path=errors_path,
            metadata_path=metadata_path,
            poll_seconds=poll_seconds,
            completion_window=completion_window,
        )

    records: list[LabelRecord] = []
    seen_custom_ids: set[str] = set()
    for item, line in _iter_batch_result_lines(output_path, errors_path):
        custom_id = item.get("custom_id")
        if custom_id not in custom_to_ein2:
            logger.warning("Ignoring unknown OpenAI batch custom_id: %s", custom_id)
            continue
        seen_custom_ids.add(custom_id)
        records.append(
            annotator.parse_batch_response_line(line, custom_to_ein2[custom_id]),
        )

    for custom_id in sorted(set(custom_to_ein2) - seen_custom_ids):
        records.append(
            annotator._error_record(
                custom_to_ein2[custom_id],
                "missing_batch_response",
            ),
        )
    return records


def _submit_and_download_openai_batch(
    *,
    annotator: OpenAIAnnotator,
    requests_path: Path,
    output_path: Path,
    errors_path: Path,
    metadata_path: Path,
    poll_seconds: int,
    completion_window: Literal["24h"],
) -> None:
    """Submit an OpenAI batch file and write its output JSONL locally."""
    request_sha256 = _file_sha256(requests_path)
    metadata = _load_batch_metadata(metadata_path, request_sha256)
    if metadata and metadata.get("batch_id"):
        batch = annotator.client.batches.retrieve(str(metadata["batch_id"]))
        logger.info("Resuming OpenAI batch %s", batch.id)
    else:
        with requests_path.open("rb") as handle:
            input_file = annotator.client.files.create(file=handle, purpose="batch")

        batch = annotator.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )
        _write_batch_metadata(
            metadata_path,
            {
                "batch_id": batch.id,
                "input_file_id": input_file.id,
                "endpoint": "/v1/chat/completions",
                "completion_window": completion_window,
                "request_sha256": request_sha256,
                "status": batch.status,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    while batch.status not in terminal_statuses:
        logger.info("OpenAI batch %s status=%s", batch.id, batch.status)
        time.sleep(poll_seconds)
        batch = annotator.client.batches.retrieve(batch.id)
        _update_batch_metadata_status(metadata_path, batch)

    _update_batch_metadata_status(metadata_path, batch)
    output_file_id = getattr(batch, "output_file_id", None)
    error_file_id = getattr(batch, "error_file_id", None)
    if output_file_id:
        _download_openai_file(annotator, output_file_id, output_path)
    if error_file_id:
        _download_openai_file(annotator, error_file_id, errors_path)

    if batch.status not in {"completed", "expired"}:
        raise RuntimeError(f"OpenAI batch {batch.id} ended with status={batch.status}")
    if not output_file_id and not error_file_id:
        raise RuntimeError(
            f"OpenAI batch {batch.id} ended with status={batch.status} but no "
            "output_file_id or error_file_id",
        )


def _download_openai_file(
    annotator: OpenAIAnnotator,
    file_id: str,
    path: Path,
) -> None:
    """Download an OpenAI file resource to ``path``."""
    content = annotator.client.files.content(file_id)
    if hasattr(content, "write_to_file"):
        content.write_to_file(path)
        return

    data = content.read() if hasattr(content, "read") else content
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(str(data), encoding="utf-8")


def _batch_outputs_cover(
    output_path: Path,
    errors_path: Path,
    expected_custom_ids: set[str],
) -> bool:
    """Return whether existing Batch output/error JSONL covers all requests."""
    seen: set[str] = set()
    try:
        for item, _line in _iter_batch_result_lines(output_path, errors_path):
            custom_id = item.get("custom_id")
            if isinstance(custom_id, str):
                seen.add(custom_id)
    except json.JSONDecodeError:
        return False
    return expected_custom_ids <= seen


def _iter_batch_result_lines(
    output_path: Path,
    errors_path: Path,
) -> list[tuple[dict[str, Any], str]]:
    """Load parsed Batch output and error JSONL lines from existing files."""
    items: list[tuple[dict[str, Any], str]] = []
    for path in (output_path, errors_path):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                items.append((item, line))
    return items


def _file_sha256(path: Path) -> str:
    """Return SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_batch_metadata(
    metadata_path: Path,
    request_sha256: str,
) -> dict[str, Any] | None:
    """Load persisted Batch metadata when it matches the current request file."""
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if metadata.get("request_sha256") != request_sha256:
        return None
    return metadata if isinstance(metadata, dict) else None


def _write_batch_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """Persist OpenAI Batch metadata for crash-safe reruns."""
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _update_batch_metadata_status(metadata_path: Path, batch: Any) -> None:
    """Update persisted Batch status without dropping submission metadata."""
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except json.JSONDecodeError:
            metadata = {}
    metadata.update(
        {
            "batch_id": batch.id,
            "status": batch.status,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    _write_batch_metadata(metadata_path, metadata)


def _safe_batch_stem(value: str) -> str:
    """Return a filesystem-safe stem for persisted batch request/output files."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


# ── Canary drift audit ────────────────────────────────────────────────────────
#
# The canary records a κ/α change test against the first run with the same
# monitor-set hash (SILICON, arXiv:2412.14461; Variance-Aware protocol,
# arXiv:2601.02370).  It is a drift monitor, not a prompt-selection target.


def _record_canary_audit(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    specs: list[BakeoffCandidate],
    prompt_ids: list[str],
    canary_ein2s: set[str],
    canary_pool_ein2s: set[str],
    run_records: list[LabelRecord],
    store: AnnotationStore,
) -> None:
    """Append canary drift metadata for the current monitor-only run.

    The audit file records the monitor-set hash, model fingerprint, aggregate
    canary predictions, and a κ/α comparison against the first run with the same
    monitor-set hash. Keeping this sidecar local to stage 03 avoids changing the
    long/tidy label schema while still making provider/model-snapshot drift
    visible to the human reviewer.
    """
    audit_path = registry.interim_dir / CANARY_AUDIT_FILENAME
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    current_df = _canary_records_frame(
        run_records=run_records,
        store=store,
        canary_pool_ein2s=canary_pool_ein2s,
        specs=specs,
        prompt_ids=prompt_ids,
    )
    predictions = _aggregate_canary_predictions(current_df, canary_pool_ein2s)
    canary_set_hash = _hash_ein2s(canary_pool_ein2s)
    baseline = _load_baseline_canary_audit(audit_path, canary_set_hash)
    change_test = _compare_canary_to_baseline(baseline, predictions)

    # The canary history is a monitor for model-snapshot/provider drift and
    # model×prompt variance, not a prompt-selection target. See SILICON
    # (arXiv:2412.14461), Variance-Aware protocol (arXiv:2601.02370), and
    # "LLM Hacking" model×prompt variance (arXiv:2509.08825).
    audit_record = {
        "run_timestamp": datetime.now(UTC).isoformat(),
        "canary_set_hash": canary_set_hash,
        "canary_set_version": f"sha256:{canary_set_hash[:12]}",
        "monitor_manifest": str(registry.monitor_manifest),
        "monitor_manifest_n": len(canary_ein2s),
        "canary_pool_n": len(canary_pool_ein2s),
        "records_written": len(run_records),
        "model_fingerprints": _model_fingerprints(
            cfg=cfg,
            specs=specs,
            prompt_ids=prompt_ids,
            records_df=current_df,
        ),
        "predictions": predictions,
        "kappa_alpha_change_test": change_test,
    }

    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_record, sort_keys=True) + "\n")

    if change_test.get("changed"):
        logger.warning(
            "Canary labels changed vs baseline: kappa=%.3f alpha=%.3f (%d/%d changed)",
            change_test["cohens_kappa"],
            change_test["krippendorff_alpha"],
            change_test["n_changed"],
            change_test["n_common"],
        )


def _canary_records_frame(
    run_records: list[LabelRecord],
    store: AnnotationStore,
    canary_pool_ein2s: set[str],
    specs: list[BakeoffCandidate],
    prompt_ids: list[str],
) -> pd.DataFrame:
    """Return the label rows that represent the current canary run.

    A normal canary invocation writes fresh rows and those records are the clean
    run snapshot. If resume skipped all work, fall back to the existing store for
    the requested model×prompt sources so a no-op rerun still records the active
    canary baseline instead of writing an empty audit row.
    """
    if run_records:
        df = pd.DataFrame([record.to_flat_dict() for record in run_records])
    else:
        df = store.to_frame()

    if df.empty:
        return df

    expected_sources = {
        f"{spec.id}__{prompt_id}" for spec in specs for prompt_id in prompt_ids
    }
    keep = df["EIN2"].astype(str).isin(canary_pool_ein2s)
    keep &= df["source_id"].astype(str).isin(expected_sources)
    return df.loc[keep].copy()


def _aggregate_canary_predictions(
    df: pd.DataFrame,
    canary_pool_ein2s: set[str],
) -> list[dict[str, Any]]:
    """Aggregate current canary labels into JSON-serializable predictions.

    Majority vote matches the stage-04 default, but the monitor slice remains a
    drift audit set only: these predictions are persisted for over-time
    comparison and are not frozen into ``silver_labels.csv``.
    """
    if df.empty:
        return []

    aggregated = aggregate_labels(df, method="majority")
    if aggregated.empty:
        return []

    aggregated = aggregated[
        aggregated["EIN2"].astype(str).isin(canary_pool_ein2s)
    ].copy()
    aggregated["EIN2"] = aggregated["EIN2"].astype(str)
    aggregated = aggregated.sort_values("EIN2")

    predictions: list[dict[str, Any]] = []
    for row in aggregated.to_dict("records"):
        label = row.get("silver_label")
        predictions.append(
            {
                "EIN2": row["EIN2"],
                "silver_label": None if pd.isna(label) else int(label),
                "num_votes": int(row["num_votes"]),
                "num_abstain": int(row["num_abstain"]),
                "agreement": None
                if pd.isna(row["agreement"])
                else float(row["agreement"]),
                "tie": bool(row["tie"]),
            },
        )
    return predictions


def _model_fingerprints(
    cfg: "BinaryClassifierConfig",
    specs: list[BakeoffCandidate],
    prompt_ids: list[str],
    records_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build model×prompt fingerprints for a canary audit record.

    The production slate should use pinned snapshot IDs. Recording that ID with
    provider, seed, temperature, and any provider system fingerprint makes later
    canary changes attributable to either prompt/model changes or closed-backend
    drift.
    """
    fingerprints: list[dict[str, Any]] = []
    for spec in specs:
        for prompt_id in prompt_ids:
            source_id = f"{spec.id}__{prompt_id}"
            source_rows = records_df
            if not records_df.empty:
                source_rows = records_df[
                    records_df["source_id"].astype(str) == source_id
                ]
            system_fingerprints = []
            if not source_rows.empty and "system_fingerprint" in source_rows.columns:
                system_fingerprints = sorted(
                    source_rows["system_fingerprint"].dropna().astype(str).unique(),
                )
            fingerprints.append(
                {
                    "source_id": source_id,
                    "provider": spec.provider,
                    "pinned_snapshot_id": spec.id,
                    "prompt_id": prompt_id,
                    "seed": cfg.SEED,
                    "temperature": cfg.annotation.temperature,
                    "system_fingerprints": system_fingerprints,
                },
            )
    return fingerprints


def _hash_ein2s(ein2s: set[str]) -> str:
    """Return a stable SHA-256 hash for the canary audit set.

    Sorting before hashing makes the monitor-set version independent of CSV row
    order, which is important when comparing canary runs across rerenders of the
    stage-01 manifests.
    """
    payload = "\n".join(sorted(ein2s)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_baseline_canary_audit(
    audit_path: Path,
    canary_set_hash: str,
) -> dict[str, Any] | None:
    """Load the first canary audit row for this monitor-set hash.

    The first run is the baseline because the monitor slice is held out and
    should remain untouched by prompt tuning; subsequent runs are drift checks
    against that fixed reference.
    """
    if not audit_path.exists():
        return None

    with audit_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("canary_set_hash") == canary_set_hash and row.get("predictions"):
                return row
    return None


def _compare_canary_to_baseline(
    baseline: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare current canary labels with the baseline using κ and α.

    Cohen's κ and Krippendorff's α are chance-corrected agreement diagnostics;
    here they are an over-time change test rather than a freeze gate (Landis &
    Koch 1977; Krippendorff's α).
    """
    if baseline is None:
        return {
            "status": "baseline",
            "changed": False,
            "n_common": 0,
            "n_changed": 0,
            "cohens_kappa": None,
            "krippendorff_alpha": None,
        }

    baseline_labels = _prediction_label_map(baseline.get("predictions", []))
    current_labels = _prediction_label_map(predictions)
    common = sorted(set(baseline_labels) & set(current_labels))
    if not common:
        return {
            "status": "insufficient_overlap",
            "baseline_run_timestamp": baseline.get("run_timestamp"),
            "changed": False,
            "n_common": 0,
            "n_changed": 0,
            "cohens_kappa": None,
            "krippendorff_alpha": None,
        }

    y_base = [baseline_labels[ein2] for ein2 in common]
    y_current = [current_labels[ein2] for ein2 in common]
    n_changed = sum(
        1 for before, after in zip(y_base, y_current, strict=True) if before != after
    )
    return {
        "status": "compared",
        "baseline_run_timestamp": baseline.get("run_timestamp"),
        "changed": n_changed > 0,
        "n_common": len(common),
        "n_changed": n_changed,
        "cohens_kappa": _cohens_kappa(y_base, y_current),
        "krippendorff_alpha": _krippendorff_alpha_nominal_two_raters(
            y_base,
            y_current,
        ),
    }


def _prediction_label_map(predictions: list[dict[str, Any]]) -> dict[str, int]:
    """Map non-abstaining canary predictions by EIN2 for drift comparison.

    Abstains/ties are excluded because κ/α require concrete categorical labels;
    their counts remain visible in the persisted prediction rows.
    """
    labels: dict[str, int] = {}
    for row in predictions:
        label = row.get("silver_label")
        if label is None:
            continue
        labels[str(row["EIN2"])] = int(label)
    return labels


def _cohens_kappa(y_a: list[int], y_b: list[int]) -> float:
    """Compute Cohen's κ for two aligned canary label vectors.

    The local implementation avoids sklearn's undefined-value warning when all
    labels fall in one class, a common outcome for tiny monitor fixtures.
    """
    labels = sorted(set(y_a) | set(y_b))
    observed = sum(1 for a, b in zip(y_a, y_b, strict=True) if a == b) / len(y_a)
    expected = 0.0
    for label in labels:
        p_a = sum(1 for value in y_a if value == label) / len(y_a)
        p_b = sum(1 for value in y_b if value == label) / len(y_b)
        expected += p_a * p_b
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return float((observed - expected) / (1 - expected))


def _krippendorff_alpha_nominal_two_raters(
    y_a: list[int],
    y_b: list[int],
) -> float:
    """Compute nominal Krippendorff's α for two canary label vectors.

    For two complete raters, observed disagreement is the share of changed
    canary labels; expected disagreement comes from the pooled label
    distribution across both runs.
    """
    observed_disagreement = sum(
        1 for a, b in zip(y_a, y_b, strict=True) if a != b
    ) / len(y_a)
    pooled = y_a + y_b
    labels = set(pooled)
    expected_agreement = sum(
        (pooled.count(label) / len(pooled)) ** 2 for label in labels
    )
    expected_disagreement = 1 - expected_agreement
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else 0.0
    return float(1 - observed_disagreement / expected_disagreement)
