"""Bake-off harness: model×prompt scoring on prompt-dev vs human labels.

The module exposes :func:`run_bakeoff`, which is the canonical entrypoint
used by both ``scripts/02_bakeoff_prompts.py`` and ``run_pipeline.py``.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from binary_classifier.annotate.annotators import OpenAIAnnotator, VLLMAnnotator
from binary_classifier.annotate.schema import AnnotationStore
from binary_classifier.config import BinaryClassifierConfig
from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────────


def run_bakeoff(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    prompt_paths: list[Path] | None = None,
    model_ids: list[str] | None = None,
    human_labels_path: Path | None = None,
    output_path: Path | None = None,
    store_path: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run the model x prompt bake-off on prompt-dev and score against human labels.

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved manifest paths.
        prompt_paths: Prompt text files to evaluate. Defaults to v1, v2, v3.
        model_ids: Override the model slate. Defaults to config model_slate.
        human_labels_path: CSV of human labels (EIN2, human_label, source_type).
        output_path: Where to write the bake-off results JSON.
        store_path: Where to write the long/tidy bake-off label store.
        limit: Limit prompt-dev records for a quick smoke test.

    Returns:
        Dict with per-model x prompt scores.

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

    if output_path is None:
        output_path = Path("results/bakeoff_results.json")
    if store_path is None:
        store_path = Path("data/bakeoff_labels.csv")

    # Load prompt-dev data
    prompt_dev = _load_prompt_dev(registry, limit)
    if prompt_dev.empty:
        logger.error("Prompt-dev manifest is empty or missing.")
        return {"error": "empty prompt-dev"}

    # Join text field from upstream if needed
    if "text" not in prompt_dev.columns:
        missions = pd.read_parquet(registry.missions_parquet)
        prompt_dev = prompt_dev.merge(
            missions[["EIN2", cfg.field]],
            on="EIN2",
            how="left",
        )
        prompt_dev = prompt_dev.rename(columns={cfg.field: "text"})

    store = AnnotationStore(store_path)
    results: list[dict] = []

    for model_id in model_ids:
        for prompt_path in prompt_paths:
            prompt_id = prompt_path.stem
            prompt_text = prompt_path.read_text()
            source_id = f"{model_id}__{prompt_id}"

            annotator = _make_annotator(cfg, model_id, prompt_id, prompt_text)

            # Run on prompt-dev
            batch = []
            for _, row in prompt_dev.iterrows():
                ein2 = row["EIN2"]
                text = row["text"]
                if store.already_done(ein2, source_id):
                    continue
                record = annotator.annotate(text, ein2=ein2)
                batch.append(record)

            if batch:
                store.append_many(batch)

            # Score
            pred_df = store.to_frame()
            pred_df = pred_df[pred_df["source_id"] == source_id][["EIN2", "label"]]

            if human_labels_path and human_labels_path.exists():
                human_df = pd.read_csv(human_labels_path)
                human_df = human_df.rename(columns={"label": "human_label"})
                scores = _score_vs_human(pred_df, human_df)
            else:
                scores = {"note": "no human labels provided — skipping scoring"}

            results.append(
                {
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                    "source_id": source_id,
                    "scores": scores,
                }
            )
            logger.info(
                "Bake-off: %s | scores: %s",
                source_id,
                json.dumps(scores, indent=2),
            )

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Bake-off results written to %s", output_path)

    return {"results": results, "output_path": str(output_path)}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_prompt_dev(registry: PathRegistry, limit: int | None) -> pd.DataFrame:
    """Load the prompt-dev manifest and optionally truncate."""
    manifest = pd.read_csv(registry.prompt_dev_manifest)
    if limit:
        manifest = manifest.head(limit)
    return manifest


def _make_annotator(
    cfg: BinaryClassifierConfig,
    model_id: str,
    prompt_id: str,
    prompt_text: str,
):
    """Instantiate the correct annotator for a model_id."""
    if model_id.startswith("gpt"):
        return OpenAIAnnotator(
            model_id=model_id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            temperature=cfg.annotation.temperature,
            seed=cfg.annotation.seed,
        )
    return VLLMAnnotator(
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        temperature=cfg.annotation.temperature,
        seed=cfg.annotation.seed,
    )


def _score_vs_human(
    pred_df: pd.DataFrame,
    human_df: pd.DataFrame,
) -> dict:
    """Compute agreement metrics between predictions and human labels.

    Returns a dict with accuracy, precision, recall, f1, and abstain rate.
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

    tp = ((valid["label"] == 1.0) & (valid["human_label"] == 1.0)).sum()
    fp = ((valid["label"] == 1.0) & (valid["human_label"] == 0.0)).sum()
    fn = ((valid["label"] == 0.0) & (valid["human_label"] == 1.0)).sum()
    tn = ((valid["label"] == 0.0) & (valid["human_label"] == 0.0)).sum()

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
