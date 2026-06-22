"""Bake-off harness: candidate x prompt scoring on prompt-dev vs human labels.

The module exposes :func:`run_bakeoff`, the canonical entrypoint used by both
``scripts/02_bakeoff_prompts.py`` and ``run_pipeline.py``. It scores every
configured bake-off candidate (closed OpenAI tiers + an open-weight vLLM arm)
on the coded prompt-dev split, writes the full score bundle, and proposes an
**unconfirmed** slate. Promoting it to ``production_slate.json`` is the human's
gate-G2 step (see :func:`run_annotation.resolve_production_specs`).

Derived bake-off artifacts are rebuilt from the full append-only label store by
default. Removing a model from YAML no longer removes its historical arms from
``bakeoff_results.json`` or the slate review table; prune the store if an arm
should disappear from derived artifacts.
"""

import concurrent.futures
import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from binary_classifier.annotate.annotators import Annotator
from binary_classifier.annotate.annotators.factory import make_annotator
from binary_classifier.annotate.concurrency import (
    annotate_with_provider_limit,
    build_provider_limiters,
)
from binary_classifier.annotate.schema import AnnotationStore
from binary_classifier.config import BakeoffCandidate, BinaryClassifierConfig
from binary_classifier.metrics import compute_metric_bundle
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_FILES: tuple[str, ...] = ("v1.txt", "v2.txt", "v3.txt")

AnnotatorFactory = Callable[[BakeoffCandidate, str, str], Annotator]

_BakeoffSetup = tuple[
    list[Path],
    list[BakeoffCandidate],
    Path,
    Path,
    Path,
    pd.DataFrame,
    pd.DataFrame,
    list[tuple[BakeoffCandidate, Path, str, str]],
]


def _resolve_bakeoff_setup(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    *,
    prompt_paths: list[Path] | None = None,
    candidates: list[BakeoffCandidate] | None = None,
    human_labels_path: Path | None = None,
    output_path: Path | None = None,
    store_path: Path | None = None,
    limit: int | None = None,
) -> _BakeoffSetup:
    if prompt_paths is None:
        prompt_paths = [registry.prompts_dir / f for f in _DEFAULT_PROMPT_FILES]
    if candidates is None:
        candidates = cfg.model_slate.bakeoff_candidates
    if human_labels_path is None:
        human_labels_path = registry.gold_coding_template
    if output_path is None:
        output_path = registry.bakeoff_results
    if store_path is None:
        store_path = registry.bakeoff_store

    prompt_dev = _load_coded_prompt_dev(human_labels_path, limit)
    human_df = prompt_dev[["EIN2", "human_label"]]
    groups: list[tuple[BakeoffCandidate, Path, str, str]] = []
    for spec in candidates:
        for prompt_path in prompt_paths:
            prompt_id = prompt_path.stem
            source_id = f"{spec.id}__{prompt_id}"
            groups.append((spec, prompt_path, prompt_id, source_id))

    return (
        prompt_paths,
        candidates,
        human_labels_path,
        output_path,
        store_path,
        prompt_dev,
        human_df,
        groups,
    )


# ── Public API ───────────────────────────────────────────────────────────────


def run_bakeoff(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    prompt_paths: list[Path] | None = None,
    candidates: list[BakeoffCandidate] | None = None,
    human_labels_path: Path | None = None,
    output_path: Path | None = None,
    store_path: Path | None = None,
    limit: int | None = None,
    annotator_factory: AnnotatorFactory | None = None,
) -> dict:
    """Score the candidate x prompt matrix on prompt-dev against human labels.

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved manifest/artifact paths.
        prompt_paths: Prompt text files to evaluate. Defaults to v1, v2, v3.
        candidates: Override the slate. Defaults to ``cfg.model_slate``.
        human_labels_path: Coded human labels. Defaults to the gold coding
            template (``EIN2, split, text, human_label``); the ``prompt_dev``
            rows are used.
        output_path: Where to write the full score bundle. Defaults to
            ``registry.bakeoff_results``.
        store_path: Long/tidy bake-off label store. Defaults to
            ``registry.bakeoff_store``.
        limit: Limit prompt-dev records for a quick smoke test.
        annotator_factory: Injected ``(spec, prompt_id, prompt_text) ->
            Annotator`` (defaults to the config-driven factory). For tests.

    Returns:
        Dict with ``results`` (full bundle) and ``proposed_slate``.

    Raises:
        FileNotFoundError: If the human coding template is absent.
        ValueError: If the template carries no coded prompt-dev labels.

    """
    if annotator_factory is None:

        def annotator_factory(
            spec: BakeoffCandidate,
            prompt_id: str,
            prompt_text: str,
        ) -> Annotator:
            return make_annotator(cfg, spec, prompt_id, prompt_text)

    (
        _prompt_paths,
        _candidates,
        _human_labels_path,
        _output_path,
        _store_path,
        prompt_dev,
        human_df,
        groups,
    ) = _resolve_bakeoff_setup(
        cfg,
        registry,
        prompt_paths=prompt_paths,
        candidates=candidates,
        human_labels_path=human_labels_path,
        output_path=output_path,
        store_path=store_path,
        limit=limit,
    )

    store = AnnotationStore(_store_path)

    store_lock = threading.Lock()
    errors_lock = threading.Lock()
    arm_errors: dict[str, str] = {}
    provider_limiters = build_provider_limiters(cfg)

    def _annotate_group(
        spec: BakeoffCandidate,
        prompt_path: Path,
        prompt_id: str,
        source_id: str,
    ) -> None:
        prompt_text = prompt_path.read_text()
        annotator = annotator_factory(spec, prompt_id, prompt_text)
        batch = []
        for _, row in prompt_dev.iterrows():
            if store.already_done(row["EIN2"], source_id):
                continue
            batch.append(
                annotate_with_provider_limit(
                    annotator,
                    spec.provider,
                    row["text"],
                    row["EIN2"],
                    provider_limiters,
                ),
            )
        if batch:
            with store_lock:
                store.append_many(batch)

    if groups:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(groups),
        ) as executor:
            future_to_group = {
                executor.submit(
                    _annotate_group,
                    spec,
                    prompt_path,
                    prompt_id,
                    source_id,
                ): (spec, prompt_path, prompt_id, source_id)
                for spec, prompt_path, prompt_id, source_id in groups
            }
            for future in concurrent.futures.as_completed(future_to_group):
                _spec, _prompt_path, _prompt_id, source_id = future_to_group[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.warning("Bake-off arm %s failed: %s", source_id, exc)
                    with errors_lock:
                        arm_errors[source_id] = str(exc)

    results = _score_results_from_store(
        store=store,
        human_df=human_df,
        groups=groups,
        provider_specs=_provider_specs(cfg, _candidates),
        arm_errors=arm_errors,
        seed=int(cfg.SEED),
        n_resamples=int(cfg.evaluation.bootstrap_resamples),
    )
    proposed = _write_bakeoff_artifacts(
        registry=registry,
        output_path=_output_path,
        results=results,
        kappa_threshold=cfg.qc.kappa_threshold,
        f1_ci_floor=cfg.qc.f1_ci_floor,
        agreement_threshold=cfg.qc.agreement_threshold,
    )

    return {
        "results": results,
        "proposed_slate": proposed,
        "bakeoff_results_path": str(_output_path),
        "proposed_slate_path": str(registry.proposed_slate),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_coded_prompt_dev(human_labels_path: Path, limit: int | None) -> pd.DataFrame:
    """Load coded prompt-dev rows from the gold coding template.

    Args:
        human_labels_path: Path to the ``gold_to_code.csv`` template.
        limit: Optional head() truncation for a smoke test.

    Returns:
        DataFrame with ``EIN2``, ``text``, ``human_label`` for prompt-dev.

    Raises:
        FileNotFoundError: If the template is missing.
        ValueError: If columns are missing or no prompt-dev labels are coded.

    """
    if not human_labels_path.exists():
        raise FileNotFoundError(
            f"No human coding template at {human_labels_path}. Run stage 01, "
            f"then code human_label (0/1) for the prompt_dev rows before the "
            f"bake-off.",
        )
    df = pd.read_csv(human_labels_path)
    required = {"EIN2", "split", "text", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{human_labels_path} missing columns: {sorted(missing)}")

    sub = df[df["split"] == "prompt_dev"].copy()
    sub = sub.dropna(subset=["human_label"])
    if sub.empty:
        raise ValueError(
            f"No coded prompt_dev rows in {human_labels_path}. Fill human_label "
            f"(0/1) for the prompt_dev split before the bake-off.",
        )
    if limit:
        sub = sub.head(limit)
    return sub


def _build_proposed_slate(
    results: list[dict],
    *,
    kappa_threshold: float,
    f1_ci_floor: float,
    agreement_threshold: float,
) -> dict:
    """Build the unconfirmed proposed slate from scored results.

    Selects every (model, prompt) combo on the same criteria the stage-04
    freeze gate enforces: Cohen's κ ≥ ``kappa_threshold`` AND the bootstrap
    minority-F1 CI-lower ≥ ``f1_ci_floor``. If no combo clears, falls back to
    the single best combo ranked by minority-F1 (point estimate). Combos with a
    missing/degenerate metric bundle are skipped rather than selected.

    ``agreement_threshold`` (the legacy raw-accuracy benchmark) is reported for
    continuity but no longer drives selection.

    Args:
        results: Scored bake-off results.
        kappa_threshold: Minimum Cohen's κ to clear (``cfg.qc.kappa_threshold``).
        f1_ci_floor: Minimum bootstrap minority-F1 CI-lower to clear
            (``cfg.qc.f1_ci_floor``).
        agreement_threshold: Legacy raw-accuracy benchmark, reported only.

    Returns:
        Proposed-slate dict with ``confirmed``, ``models``, a compact ranked
        ``selected`` summary table, recommendation metadata, and the reported
        (non-driving) ``agreement_threshold``.

    """

    def _scores(r: dict) -> dict:
        scores = r.get("scores", {})
        return scores if isinstance(scores, dict) else {}

    def _bundle(r: dict) -> dict | None:
        scores = _scores(r)
        bundle = scores.get("metrics")
        return bundle if isinstance(bundle, dict) else None

    scored = [r for r in results if _bundle(r) is not None]
    eligible_scored = [r for r in scored if _has_known_provider(r)]

    def _clears(r: dict) -> bool:
        bundle = _bundle(r)
        assert bundle is not None
        kappa = bundle["cohens_kappa"]
        ci_lower = bundle["bootstrap_ci"]["minority_f1"]["lower"]
        kappa_pass = bool(np.isfinite(kappa) and kappa >= kappa_threshold)
        f1_ci_pass = bool(np.isfinite(ci_lower) and ci_lower >= f1_ci_floor)
        return kappa_pass and f1_ci_pass

    def _minority_f1(r: dict) -> float:
        bundle = _bundle(r)
        assert bundle is not None
        f1 = bundle["f1"]
        return f1 if np.isfinite(f1) else float("-inf")

    def _summary_row(r: dict, *, clears: bool) -> dict:
        scores = _scores(r)
        bundle = _bundle(r)
        assert bundle is not None
        return {
            "model_id": r["model_id"],
            "prompt_id": r["prompt_id"],
            "accuracy": scores.get("accuracy"),
            "f1": bundle.get("f1"),
            "cohens_kappa": bundle.get("cohens_kappa"),
            "f1_ci_lower": bundle.get("bootstrap_ci", {})
            .get("minority_f1", {})
            .get("lower"),
            "abstain_rate": scores.get("abstain_rate"),
            "n_valid": scores.get("n_valid"),
            "clears": clears,
        }

    def _review_row(r: dict) -> dict:
        row = {
            "model_id": r.get("model_id"),
            "prompt_id": r.get("prompt_id"),
            "source_id": r.get("source_id"),
            "provider": r.get("provider"),
            "reasoning_effort": r.get("reasoning_effort"),
            "currently_configured": bool(r.get("currently_configured", False)),
        }
        scores = _scores(r)
        bundle = _bundle(r)
        if bundle is None:
            row.update(
                {
                    "accuracy": scores.get("accuracy"),
                    "f1": None,
                    "cohens_kappa": None,
                    "f1_ci_lower": None,
                    "abstain_rate": scores.get("abstain_rate"),
                    "n_valid": scores.get("n_valid"),
                    "clears": False,
                    "error": scores.get("error") or "missing metric bundle",
                },
            )
            return row
        row.update(_summary_row(r, clears=_clears(r)))
        return row

    def _sort_key(row: dict) -> tuple[float, float, int]:
        f1 = row["f1"]
        kappa = row["cohens_kappa"]
        n_valid = row["n_valid"]
        return (
            float(f1) if f1 is not None and np.isfinite(f1) else float("-inf"),
            float(kappa) if kappa is not None and np.isfinite(kappa) else float("-inf"),
            int(n_valid) if n_valid is not None else -1,
        )

    clearing = [r for r in eligible_scored if _clears(r)]
    if clearing:
        selected_results = clearing
    elif eligible_scored:
        selected_results = [max(eligible_scored, key=_minority_f1)]
    else:
        selected_results = []

    selected = sorted(
        (_summary_row(r, clears=_clears(r)) for r in selected_results),
        key=_sort_key,
        reverse=True,
    )

    models: list[dict] = []
    seen: set[str] = set()
    selected_keys = {(row["model_id"], row["prompt_id"]) for row in selected}
    for r in results:
        if (r.get("model_id"), r.get("prompt_id")) not in selected_keys:
            continue
        if r["model_id"] not in seen:
            seen.add(r["model_id"])
            models.append(
                {
                    "id": r["model_id"],
                    "provider": r["provider"],
                    "reasoning_effort": r["reasoning_effort"],
                },
            )

    arms = sorted((_review_row(r) for r in results), key=_sort_key, reverse=True)
    not_currently_configured = [
        row for row in arms if not bool(row.get("currently_configured", False))
    ]

    recommended = selected[0] if selected else None
    if recommended is None:
        rationale = (
            "No currently configured, provider-backed model-prompt arms had usable "
            "metrics. Store-only arms are listed for review but are not eligible "
            "for recommendation until configured with a provider."
        )
    elif recommended["clears"]:
        rationale = (
            "Top-ranked arm clears both Cohen's kappa and minority-F1 CI gates; "
            "review abstain_rate and n_valid before confirming."
        )
    else:
        rationale = (
            "No arm clears both gates; recommended arm is the single-best "
            "fallback by minority F1."
        )

    return {
        "confirmed": False,
        "models": models,
        "selected": selected,
        "arms": arms,
        "not_currently_configured": not_currently_configured,
        "recommended": recommended,
        "rationale": rationale,
        "selection_scope": "full_store",
        "kappa_threshold": kappa_threshold,
        "f1_ci_floor": f1_ci_floor,
        "agreement_threshold": agreement_threshold,
    }


def _has_known_provider(result: dict) -> bool:
    return result.get("provider") in {"openai", "vllm"}


def _parse_source_id(source_id: str) -> tuple[str, str] | None:
    """Parse ``model_id__prompt_id`` source identifiers."""
    if "__" not in source_id:
        return None
    model_id, prompt_id = source_id.rsplit("__", 1)
    if not model_id or not prompt_id:
        return None
    return model_id, prompt_id


def _provider_specs(
    cfg: BinaryClassifierConfig,
    candidates: list[BakeoffCandidate],
) -> dict[str, BakeoffCandidate]:
    """Return provider metadata from YAML plus any explicit candidate override."""
    specs = {spec.id: spec for spec in cfg.model_slate.bakeoff_candidates}
    specs.update({spec.id: spec for spec in candidates})
    return specs


def _group_metadata(
    groups: list[tuple[BakeoffCandidate, Path, str, str]],
) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for spec, _prompt_path, prompt_id, source_id in groups:
        metadata[source_id] = {
            "model_id": spec.id,
            "provider": spec.provider,
            "reasoning_effort": spec.reasoning_effort,
            "prompt_id": prompt_id,
            "source_id": source_id,
            "currently_configured": True,
        }
    return metadata


def _score_results_from_store(
    *,
    store: AnnotationStore,
    human_df: pd.DataFrame,
    groups: list[tuple[BakeoffCandidate, Path, str, str]],
    provider_specs: dict[str, BakeoffCandidate],
    arm_errors: dict[str, str] | None,
    seed: int,
    n_resamples: int,
) -> list[dict]:
    """Score every current and stored bake-off arm from the label store."""
    current_metadata = _group_metadata(groups)
    store_df = store.to_frame()
    source_ids = list(current_metadata)
    if not store_df.empty:
        for source_id in store_df["source_id"].dropna().unique().tolist():
            if source_id not in current_metadata:
                source_ids.append(str(source_id))

    results: list[dict] = []
    for source_id in source_ids:
        metadata = current_metadata.get(source_id)
        if metadata is None:
            parsed = _parse_source_id(source_id)
            if parsed is None:
                logger.warning("Skipping malformed bake-off source_id: %s", source_id)
                continue
            model_id, prompt_id = parsed
            spec = provider_specs.get(model_id)
            metadata = {
                "model_id": model_id,
                "provider": spec.provider if spec is not None else None,
                "reasoning_effort": spec.reasoning_effort if spec is not None else None,
                "prompt_id": prompt_id,
                "source_id": source_id,
                "currently_configured": False,
            }

        if arm_errors and source_id in arm_errors:
            scores = {"error": arm_errors[source_id]}
        else:
            try:
                pred_df = store_df[store_df["source_id"] == source_id]
                pred_df = pred_df.drop_duplicates(
                    subset=["EIN2", "source_id"],
                    keep="last",
                )[["EIN2", "label"]]
                scores = _score_vs_human(
                    pred_df,
                    human_df,
                    seed=seed,
                    n_resamples=n_resamples,
                )
            except Exception as exc:
                logger.warning(
                    "Bake-off arm %s scoring failed: %s",
                    source_id,
                    exc,
                )
                scores = {"error": str(exc)}

        result = {**metadata, "scores": scores}
        results.append(result)
        logger.info("Bake-off: %s | scores: %s", source_id, json.dumps(scores))

    return results


def _write_bakeoff_artifacts(
    *,
    registry: PathRegistry,
    output_path: Path,
    results: list[dict],
    kappa_threshold: float,
    f1_ci_floor: float,
    agreement_threshold: float,
) -> dict:
    """Write full-store results and the unconfirmed proposed slate."""
    proposed = _build_proposed_slate(
        results,
        kappa_threshold=kappa_threshold,
        f1_ci_floor=f1_ci_floor,
        agreement_threshold=agreement_threshold,
    )

    # Do NOT write production_slate.json — that is the human's gate-G2 step.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Bake-off results written to %s", output_path)

    registry.proposed_slate.parent.mkdir(parents=True, exist_ok=True)
    registry.proposed_slate.write_text(json.dumps(proposed, indent=2))
    logger.info("Proposed (unconfirmed) slate written to %s", registry.proposed_slate)
    return proposed


def rebuild_proposed_slate_from_results(
    registry: PathRegistry,
    *,
    kappa_threshold: float,
    f1_ci_floor: float,
    agreement_threshold: float,
) -> dict:
    """Rebuild ``proposed_slate.json`` from existing bake-off scores only.

    This is a local, zero-cost Gate-G2 utility: it reads the persisted full
    ``bakeoff_results.json`` bundle, rewrites the compact proposed slate, and
    never calls annotators or touches the label store.

    Args:
        registry: Path registry resolving bake-off artifact paths.
        kappa_threshold: Minimum Cohen's κ to clear.
        f1_ci_floor: Minimum bootstrap minority-F1 CI-lower to clear.
        agreement_threshold: Legacy raw-accuracy benchmark, reported only.

    Returns:
        The rebuilt proposed slate dict.

    Raises:
        FileNotFoundError: If ``registry.bakeoff_results`` does not exist.

    """
    results = json.loads(registry.bakeoff_results.read_text())
    proposed = _build_proposed_slate(
        results,
        kappa_threshold=kappa_threshold,
        f1_ci_floor=f1_ci_floor,
        agreement_threshold=agreement_threshold,
    )
    registry.proposed_slate.parent.mkdir(parents=True, exist_ok=True)
    registry.proposed_slate.write_text(json.dumps(proposed, indent=2))
    logger.info("Rebuilt proposed slate from %s", registry.bakeoff_results)
    return proposed


def rebuild_bakeoff_artifacts_from_store(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    *,
    prompt_paths: list[Path] | None = None,
    candidates: list[BakeoffCandidate] | None = None,
    human_labels_path: Path | None = None,
    output_path: Path | None = None,
    store_path: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Rebuild full-store bake-off results and proposed slate without annotation.

    This is the zero-cost additive rebuild path. It scores the union of current
    configured groups and every distinct ``source_id`` already present in the
    append-only bake-off label store. Store-only arms keep their historical
    scores in derived artifacts; if their model is not configured in the YAML,
    ``provider`` is ``None`` and they are excluded from recommendation.
    """
    (
        _prompt_paths,
        _candidates,
        _human_labels_path,
        _output_path,
        _store_path,
        prompt_dev,
        human_df,
        groups,
    ) = _resolve_bakeoff_setup(
        cfg,
        registry,
        prompt_paths=prompt_paths,
        candidates=candidates,
        human_labels_path=human_labels_path,
        output_path=output_path,
        store_path=store_path,
        limit=limit,
    )

    results = _score_results_from_store(
        store=AnnotationStore(_store_path),
        human_df=human_df,
        groups=groups,
        provider_specs=_provider_specs(cfg, _candidates),
        arm_errors=None,
        seed=int(cfg.SEED),
        n_resamples=int(cfg.evaluation.bootstrap_resamples),
    )
    proposed = _write_bakeoff_artifacts(
        registry=registry,
        output_path=_output_path,
        results=results,
        kappa_threshold=cfg.qc.kappa_threshold,
        f1_ci_floor=cfg.qc.f1_ci_floor,
        agreement_threshold=cfg.qc.agreement_threshold,
    )

    return {
        "results": results,
        "proposed_slate": proposed,
        "bakeoff_results_path": str(_output_path),
        "proposed_slate_path": str(registry.proposed_slate),
    }


def _bakeoff_minority_class(y_true: "np.ndarray") -> int:
    """Return the minority class, choosing the positive class on ties.

    Args:
        y_true: Ground-truth label vector (0/1).

    Returns:
        ``1`` when the positive class is no larger than the negative class
        (including ties), else ``0``. Mirrors the freeze/stage-11 tie rule
        without importing a private helper.

    """
    counts = np.bincount(y_true.astype(int), minlength=2)
    if counts[1] <= counts[0]:
        return 1
    return 0


def _score_vs_human(
    pred_df: pd.DataFrame,
    human_df: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int,
) -> dict:
    """Score predictions against human labels on the prompt-dev overlap.

    Keeps the legacy flat accuracy/precision/recall/f1/abstain-rate keys for
    continuity and adds the imbalanced bundle under ``metrics`` (minority
    P/R/F1, MCC, balanced accuracy, PR-AUC, Cohen's κ, and the bootstrap
    minority-F1 CI) so the bake-off can select on the same quantities the
    stage-04 freeze gate enforces.

    Args:
        pred_df: Predictions with ``EIN2`` and ``label`` (may be NaN/abstain).
        human_df: Human labels with ``EIN2`` and ``human_label``.
        seed: Bootstrap seed (``cfg.SEED``).
        n_resamples: Bootstrap resamples (``cfg.evaluation.bootstrap_resamples``).

    Returns:
        Score dict, or an ``error`` dict when there is no human overlap.

    """
    pred_df = pred_df.drop_duplicates(subset=["EIN2"], keep="last")
    merged = pred_df.merge(human_df, on="EIN2", how="inner")
    if merged.empty:
        return {"error": "no overlap with human labels"}

    # Drop rows where either side abstained
    valid = merged.dropna(subset=["label", "human_label"])
    abstain_rate = 1 - len(valid) / len(merged)

    if valid.empty:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "abstain_rate": abstain_rate,
            "metrics": None,
            "n_valid": 0,
            "n_total": len(merged),
        }

    tp = int(((valid["label"] == 1.0) & (valid["human_label"] == 1.0)).sum())
    fp = int(((valid["label"] == 1.0) & (valid["human_label"] == 0.0)).sum())
    fn = int(((valid["label"] == 0.0) & (valid["human_label"] == 1.0)).sum())
    tn = int(((valid["label"] == 0.0) & (valid["human_label"] == 0.0)).sum())

    accuracy = (tp + tn) / len(valid)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Imbalanced bundle — selection metrics matching the freeze gate. A single
    # class present (or too few valid rows) can make the bundle degenerate;
    # guard so a bad combo is skipped at selection rather than crashing here.
    y_true = valid["human_label"].astype(int).to_numpy()
    y_pred = valid["label"].astype(int).to_numpy()
    metrics: dict | None
    try:
        metrics = compute_metric_bundle(
            y_true,
            y_pred,
            y_score=None,
            minority_class=_bakeoff_minority_class(y_true),
            seed=seed,
            n_resamples=n_resamples,
        )
    except Exception as exc:  # noqa: BLE001 - degrade, do not abort the bake-off
        logger.warning("Imbalanced metric bundle failed: %s", exc)
        metrics = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "abstain_rate": abstain_rate,
        "metrics": metrics,
        "n_valid": len(valid),
        "n_total": len(merged),
    }
