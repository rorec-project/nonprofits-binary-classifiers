"""Tests for names-arm synthetic probes and DBA case-study diagnostics."""

import json

import pandas as pd

from binary_classifier.names.diagnostics import run_name_diagnostics


def test_run_name_diagnostics_writes_probe_scores_and_dba_token_cases(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """The diagnostic seam emits raw probe scores and both DBA token directions."""
    tiny_config.inference.device = "cpu"
    _write_selected_model(tiny_registry)
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "population": "panel_scoped",
                "name_cased": "Heritage Academy",
                "dba_cased": "Grace Bible Church",
                "has_dba": True,
                "is_manifest_contaminated": False,
            },
            {
                "EIN2": "P002",
                "population": "panel_scoped",
                "name_cased": "First Baptist Church",
                "dba_cased": "Community Center",
                "has_dba": True,
                "is_manifest_contaminated": False,
            },
            {
                "EIN2": "P003",
                "population": "panel_scoped",
                "name_cased": "No DBA Organization",
                "dba_cased": None,
                "has_dba": False,
                "is_manifest_contaminated": False,
            },
            {
                "EIN2": "P004",
                "population": "panel_scoped",
                "name_cased": "First Baptist Church",
                "dba_cased": "Grace Bible Church",
                "has_dba": True,
                "is_manifest_contaminated": False,
            },
        ],
    ).to_parquet(tiny_registry.names_panel_cleaned, index=False)
    monkeypatch.setattr(
        "binary_classifier.inference.router.route",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("router called")),
    )

    run_name_diagnostics(tiny_config, tiny_registry, predictor=_TextScorePredictor())

    probes = json.loads(tiny_registry.names_probe_diagnostics.read_text())
    cases = pd.read_parquet(tiny_registry.names_dba_case_study)
    report = json.loads(tiny_registry.names_dba_case_study_report.read_text())

    assert probes["diagnostic_only"] is True
    assert probes["interpretation"] == "diagnosis_not_accuracy"
    assert {record["category"] for record in probes["records"]} >= {
        "tradition_token",
        "saint_name",
        "faith_heritage",
        "matched_control",
    }
    assert probes["probe_set_version"] == "v2"
    assert {record["probe_id"] for record in probes["records"]} >= {
        "tradition_baptist",
        "control_community",
        "saint_hospital",
        "faith_health",
        "muslim",
        "hindu",
        "buddhist",
    }
    assert next(
        record["pair_id"]
        for record in probes["records"]
        if record["probe_id"] == "tradition_baptist"
    ) == "baptist_pair"
    assert all(record["prob_raw"] in {0.2, 0.8} for record in probes["records"])
    assert all(record["model_id"] == "stub-model" for record in probes["records"])
    assert set(cases["EIN2"]) == {"P001", "P002", "P004"}
    assert cases.set_index("EIN2")["token_direction"].to_dict() == {
        "P001": "dba_adds_religious_token",
        "P002": "legal_name_adds_religious_token",
        "P004": "both_names_add_religious_token",
    }
    assert cases.set_index("EIN2")["legal_name_prob_raw"].to_dict() == {
        "P001": 0.2,
        "P002": 0.8,
        "P004": 0.8,
    }
    assert cases.set_index("EIN2")["dba_name_prob_raw"].to_dict() == {
        "P001": 0.8,
        "P002": 0.2,
        "P004": 0.8,
    }
    assert report["dba_adds_religious_token_count"] == 2
    assert report["legal_name_adds_religious_token_count"] == 2
    assert report["diagnostic_only"] is True
    assert report["production_input_variant"] is False


def _write_selected_model(registry) -> None:
    registry.selected_model.write_text(
        json.dumps(
            {
                "encoder_id": "stub-model",
                "checkpoint_sha256": "stub-sha",
                "checkpoint_relpath": "unused/model.safetensors",
            },
        ),
    )


class _TextScorePredictor:
    def predict_proba(self, texts):
        scores = [0.8 if "Church" in text else 0.2 for text in texts]
        return [[1.0 - score, score] for score in scores]
