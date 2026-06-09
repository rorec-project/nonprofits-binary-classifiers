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
    """Aggregate labels and run the blocking QC gate.

    Steps:
        1. Load the long/tidy annotation store.
        2. Aggregate per-EIN2 labels by majority vote (default).
        3. Compute LLM-vs-human agreement on the coded validation split.
        4. Freeze the silver labels **only if** agreement meets the configured
           threshold; otherwise raise (write nothing).

    Args:
        cfg: Validated configuration object.
        registry: Path registry with resolved manifest paths.
        store_path: Path to the long/tidy store. Defaults to
            ``registry.annotation_store``.
        human_validation_path: Coded human labels. Defaults to the gold coding
            template; the ``validation`` split (``EIN2, human_label``) is used.
        output_path: Where to write the frozen silver labels.
            Defaults to ``train_test_datasets/silver_labels.csv``.

    Returns:
        Dict with ``agreement``, ``n_total``, ``n_abstain``, ``n_valid``, and
        ``frozen_path``.

    Raises:
        ValueError: If the store is empty, no validation labels overlap, or the
            agreement gate fails (below the configured threshold).
        FileNotFoundError: If no coded validation labels are available.
    """
    if store_path is None:
        store_path = registry.annotation_store
    if output_path is None:
        output_path = registry.train_test_dir / "silver_labels.csv"
    if human_validation_path is None:
        human_validation_path = registry.gold_coding_template

    store = AnnotationStore(store_path)
    df = store.to_frame()

    if df.empty:
        raise ValueError(
            f"Annotation store at {store_path} is empty. Run stage 03 first."
        )

    # Aggregate
    aggregated = aggregate_labels(df, method="majority")
    n_total = len(aggregated)
    n_abstain = int(aggregated["silver_label"].isna().sum())
    logger.info(
        "Aggregated %d EIN2s; %d abstain/tie (%.1f%%)",
        n_total,
        n_abstain,
        100 * n_abstain / n_total,
    )

    # Validation agreement (blocking gate).
    human_df = _load_validation_labels(human_validation_path)
    merged = aggregated.merge(
        human_df[["EIN2", "human_label"]],
        on="EIN2",
        how="inner",
    )
    valid = merged.dropna(subset=["silver_label", "human_label"])
    if valid.empty:
        raise ValueError(
            "No overlapping validation labels to compare; cannot run the "
            "agreement gate. Check that the silver run and the coded "
            "validation split share EIN2s."
        )

    agree = int((valid["silver_label"] == valid["human_label"]).sum())
    agreement = agree / len(valid)
    threshold = cfg.qc.agreement_threshold
    logger.info(
        "Validation agreement: %.1f%% (%d/%d); threshold %.0f%%",
        agreement * 100,
        agree,
        len(valid),
        threshold * 100,
    )

    if agreement < threshold:
        raise ValueError(
            f"AGREEMENT GATE FAILED: {agreement:.1%} < {threshold:.0%}. "
            f"Revise prompts and re-label; nothing was frozen."
        )

    # Freeze (gate passed).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)
    logger.info("Frozen silver labels written to %s", output_path)

    return {
        "agreement": agreement,
        "n_total": n_total,
        "n_abstain": n_abstain,
        "n_valid": len(valid),
        "frozen_path": str(output_path),
    }


def _load_validation_labels(human_validation_path: Path) -> pd.DataFrame:
    """Load coded validation labels from the gold coding template.

    Args:
        human_validation_path: Path to ``gold_to_code.csv``.

    Returns:
        DataFrame with ``EIN2`` and ``human_label`` for the validation split.

    Raises:
        FileNotFoundError: If the template is missing.
        ValueError: If columns are missing or no validation labels are coded.

    """
    if not human_validation_path.exists():
        raise FileNotFoundError(
            f"No human coding template at {human_validation_path}. Code "
            f"human_label (0/1) for the validation split before stage 04."
        )
    df = pd.read_csv(human_validation_path)
    required = {"EIN2", "split", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{human_validation_path} missing columns: {sorted(missing)}.")
    sub = df[df["split"] == "validation"].dropna(subset=["human_label"])
    if sub.empty:
        raise ValueError(
            f"No coded validation labels in {human_validation_path}. Fill "
            f"human_label (0/1) for the validation split before stage 04."
        )
    return sub
