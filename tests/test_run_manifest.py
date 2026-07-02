"""Tests for reproducibility run manifests."""

import json

import pandas as pd

from binary_classifier.repro.manifest import write_run_manifest


def test_write_run_manifest_records_thresholds_and_completeness(
    tiny_config,
    tiny_registry,
) -> None:
    tiny_registry.calibrator_path.write_text(
        json.dumps({"threshold": 0.05, "max_f1_threshold": 0.60}) + "\n"
    )
    tiny_registry.base_rate_precision.write_text(json.dumps({"threshold": 0.70}) + "\n")
    tiny_registry.predictions_full_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "EIN2": ["E1", "E2"],
            "pred_label": [1, 0],
        }
    ).to_parquet(tiny_registry.predictions_full_parquet, index=False)

    payload = write_run_manifest(tiny_config, tiny_registry)

    written = json.loads(tiny_registry.run_manifest.read_text())
    assert written == payload
    assert payload["thresholds"] == {
        "operating": 0.05,
        "max_f1": 0.60,
        "base_rate": 0.70,
    }
    assert payload["wave2_completeness"] == {
        "status": "ok",
        "path": str(tiny_registry.predictions_full_parquet),
        "rows": 2,
        "unique_ein2": 2,
        "null_pred_label": 0,
    }
    assert payload["environment_lock"]["uv_lock"]["sha256"]
    assert payload["post_sprint_reproduce_assertion"] is None
