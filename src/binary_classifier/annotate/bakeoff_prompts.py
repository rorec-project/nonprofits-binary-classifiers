"""Bake-off harness: candidate x prompt scoring on prompt-dev vs human labels.

The module exposes :func:`run_bakeoff`, the canonical entrypoint used by both
``scripts/02_bakeoff_prompts.py`` and ``run_pipeline.py``. It scores every
configured bake-off candidate (closed OpenAI tiers + an open-weight vLLM arm)
on the coded prompt-dev split, writes the full score bundle, and proposes an
**unconfirmed** slate. Promoting it to ``production_slate.json`` is the human's
gate-G2 step (see :func:`run_annotation.resolve_production_specs`).
"""

import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from binary_classifier.annotate.annotators import Annotator
from binary_classifier.annotate.annotators.factory import make_annotator
from binary_classifier.annotate.schema import AnnotationStore
from binary_classifier.config import BakeoffCandidate, BinaryClassifierConfig
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_FILES: tuple[str, ...] = ("v1.txt", "v2.txt", "v3.txt")

AnnotatorFactory = Callable[[BakeoffCandidate, str, str], Annotator]


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
    if annotator_factory is None:

        def annotator_factory(
            spec: BakeoffCandidate,
            prompt_id: str,
            prompt_text: str,
        ) -> Annotator:
            return make_annotator(cfg, spec, prompt_id, prompt_text)

    # Load coded prompt-dev rows (text + human_label) — raises if absent.
    prompt_dev = _load_coded_prompt_dev(human_labels_path, limit)
    human_df = prompt_dev[["EIN2", "human_label"]]

    store = AnnotationStore(store_path)
    results: list[dict] = []
    threshold = cfg.qc.agreement_threshold

    for spec in candidates:
        for prompt_path in prompt_paths:
            prompt_id = prompt_path.stem
            source_id = f"{spec.id}__{prompt_id}"
            try:
                prompt_text = prompt_path.read_text()
                annotator = annotator_factory(spec, prompt_id, prompt_text)
                batch = []
                for _, row in prompt_dev.iterrows():
                    if store.already_done(row["EIN2"], source_id):
                        continue
                    batch.append(annotator.annotate(row["text"], ein2=row["EIN2"]))
                if batch:
                    store.append_many(batch)

                pred_df = store.to_frame()
                pred_df = pred_df[pred_df["source_id"] == source_id][["EIN2", "label"]]
                scores = _score_vs_human(pred_df, human_df)
            except Exception as exc:
                # A vLLM connection failure (server down) or a provider error
                # degrades this one arm; the OpenAI arms still score.
                logger.warning("Bake-off arm %s failed: %s", source_id, exc)
                scores = {"error": str(exc)}

            results.append(
                {
                    "model_id": spec.id,
                    "provider": spec.provider,
                    "reasoning_effort": spec.reasoning_effort,
                    "prompt_id": prompt_id,
                    "source_id": source_id,
                    "scores": scores,
                }
            )
            logger.info("Bake-off: %s | scores: %s", source_id, json.dumps(scores))

    proposed = _build_proposed_slate(results, threshold)

    # Write the full bundle and the proposed (unconfirmed) slate. Do NOT write
    # production_slate.json — that is the human's gate-G2 step.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Bake-off results written to %s", output_path)

    registry.proposed_slate.parent.mkdir(parents=True, exist_ok=True)
    registry.proposed_slate.write_text(json.dumps(proposed, indent=2))
    logger.info("Proposed (unconfirmed) slate written to %s", registry.proposed_slate)

    return {
        "results": results,
        "proposed_slate": proposed,
        "bakeoff_results_path": str(output_path),
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
            f"bake-off."
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
            f"(0/1) for the prompt_dev split before the bake-off."
        )
    if limit:
        sub = sub.head(limit)
    return sub


def _build_proposed_slate(results: list[dict], threshold: float) -> dict:
    """Build the unconfirmed proposed slate from scored results.

    Selects every (model, prompt) combo whose accuracy clears ``threshold``;
    if none clear, falls back to the single best-scoring combo.
    """
    scored = [
        r
        for r in results
        if isinstance(r["scores"].get("accuracy"), (int, float))
        and r["scores"].get("accuracy") is not None
    ]
    clearing = [r for r in scored if r["scores"]["accuracy"] >= threshold]
    if clearing:
        selected = clearing
    elif scored:
        selected = [max(scored, key=lambda r: r["scores"]["accuracy"])]
    else:
        selected = []

    models: list[dict] = []
    seen: set[str] = set()
    for r in selected:
        if r["model_id"] not in seen:
            seen.add(r["model_id"])
            models.append(
                {
                    "id": r["model_id"],
                    "provider": r["provider"],
                    "reasoning_effort": r["reasoning_effort"],
                }
            )

    return {"confirmed": False, "models": models, "selected": selected}


def _score_vs_human(
    pred_df: pd.DataFrame,
    human_df: pd.DataFrame,
) -> dict:
    """Compute agreement metrics between predictions and human labels.

    Returns a dict with accuracy, precision, recall, f1, and abstain rate.
    The fuller imbalanced bundle (MCC, balanced accuracy, bootstrap CIs) is
    Phase 2 (T2.1).
    """
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "abstain_rate": abstain_rate,
        "n_valid": len(valid),
        "n_total": len(merged),
    }
