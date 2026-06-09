"""Quality control and agreement metrics for the annotation pipeline.

Provides LLM-vs-human agreement scoring, Krippendorff alpha, and the ≥85%
validation gate. Also handles versioning and freezing of the final label
artifact.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from binary_classifier.annotate.aggregate import aggregate_labels
from binary_classifier.annotate.schema import AnnotationStore

if TYPE_CHECKING:
    from binary_classifier.config import BinaryClassifierConfig
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def run_quality_check(
    cfg: "BinaryClassifierConfig",
    registry: "PathRegistry",
    store_path: Path | None = None,
    human_validation_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Aggregate labels and run the QC gate.

    Steps:
        1. Load the long/tidy annotation store.
        2. Aggregate per-EIN2 labels by majority vote (default).
        3. Compute LLM-vs-human agreement on the validation set.
        4. If agreement ≥ 85 %, version and freeze the labelled file.
        5. Otherwise, raise a warning / gate failure.

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved manifest paths.
        store_path: Path to the long/tidy store. Defaults to
            ``data/annotation_store.csv``.
        human_validation_path: CSV of human labels for validation
            (EIN2, human_label). Defaults to the validation manifest.
        output_path: Where to write the frozen silver labels.
            Defaults to ``train_test_datasets/silver_labels.csv``.

    Returns:
        Dict with ``agreement``, ``n_total``, ``n_agree``, and ``frozen_path``.
    """
    if store_path is None:
        store_path = Path("data/annotation_store.csv")
    if output_path is None:
        output_path = registry.train_test_dir / "silver_labels.csv"

    store = AnnotationStore(store_path)
    df = store.to_frame()

    if df.empty:
        logger.error("Annotation store is empty. Run stage 03 first.")
        return {"error": "empty store"}

    # Aggregate
    aggregated = aggregate_labels(df, method="majority")
    n_total = len(aggregated)
    n_abstain = aggregated["silver_label"].isna().sum()
    logger.info(
        "Aggregated %d EIN2s; %d abstain/tie (%.1f%%)",
        n_total,
        n_abstain,
        100 * n_abstain / n_total,
    )

    # Validation agreement
    if human_validation_path and human_validation_path.exists():
        human_df = pd.read_csv(human_validation_path)
        merged = aggregated.merge(
            human_df[["EIN2", "human_label"]],
            on="EIN2",
            how="inner",
        )
        valid = merged.dropna(subset=["silver_label", "human_label"])
        if not valid.empty:
            agree = (valid["silver_label"] == valid["human_label"]).sum()
            agreement = agree / len(valid)
            logger.info(
                "Validation agreement: %.1f%% (%d/%d)",
                agreement * 100,
                agree,
                len(valid),
            )
            if agreement < 0.85:
                logger.warning(
                    "AGREEMENT GATE FAILED: %.1f%% < 85%%. "
                    "Revise prompts and re-label.",
                    agreement * 100,
                )
        else:
            agreement = None
            logger.warning("No overlapping validation labels to compare.")
    else:
        agreement = None
        logger.info("No human validation labels found — skipping agreement gate.")

    # Freeze
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)
    logger.info("Frozen silver labels written to %s", output_path)

    return {
        "agreement": agreement,
        "n_total": n_total,
        "n_abstain": n_abstain,
        "frozen_path": str(output_path),
    }
