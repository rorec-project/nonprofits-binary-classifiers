"""Thin CLI wrapper for Stage 4: quality control and aggregation.

Calls :func:`binary_classifier.qc.agreement.run_quality_check`.
"""

import argparse
import logging
import tempfile
from pathlib import Path

from binary_classifier.annotate.schema import AnnotationStore
from binary_classifier.config import load_config
from binary_classifier.paths import PathRegistry
from binary_classifier.qc.agreement import run_quality_check
from binary_classifier.qc.evidence import (
    abstain_fabricated_positives,
    verify_evidence_spans,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate labels and run the QC gate (agreement).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/religious_missions.yaml"),
        help="Path to the task configuration YAML file.",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=None,
        help="Path to the long/tidy annotation store "
        "(defaults to registry.annotation_store).",
    )
    parser.add_argument(
        "--human-validation",
        type=Path,
        default=None,
        help="Coded human validation labels (defaults to the gold coding "
        "template; the validation split is used).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the frozen silver labels "
        "(defaults to train_test_datasets/silver_labels.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    registry = PathRegistry(args.config)

    store_path = args.store_path or registry.annotation_store

    # Evidence-span verification (before aggregation)
    store = AnnotationStore(store_path)
    df = store.to_frame()

    if not df.empty:
        try:
            ev = verify_evidence_spans(registry, df)
        except FileNotFoundError as exc:
            logger.warning("Skipping evidence verification: %s", exc)
            ev = None
        else:
            logger.info(
                "Evidence verification: %d/%d spans verified, %d fabricated (%.1f%%)",
                ev["verified_spans"],
                ev["total_spans"],
                ev["fabricated_spans"],
                ev["fabrication_rate"] * 100,
            )
            if ev["fabricated_records"]:
                logger.warning(
                    "Fabricated spans found for %d record(s)",
                    len({(e, s) for e, s, _ in ev["fabricated_records"]}),
                )

            if cfg.qc.abstain_on_fabricated_positive and ev["fabricated_records"]:
                df = abstain_fabricated_positives(df, ev["fabricated_records"])
                tmp_dir = Path(tempfile.mkdtemp(prefix="qc_"))
                tmp_store = tmp_dir / "annotation_store.csv"
                df.to_csv(tmp_store, index=False)
                store_path = tmp_store
                logger.info("Wrote abstained store to %s", tmp_store)

    run_quality_check(
        cfg,
        registry,
        store_path=store_path,
        human_validation_path=args.human_validation,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
