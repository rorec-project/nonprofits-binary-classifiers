"""Tests for names-arm cross-field transfer scoring."""

import json

import pandas as pd
import pytest

from binary_classifier.names import score as score_mod
from binary_classifier.names.score import score_names


def test_score_names_writes_both_variants_without_mission_routing(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """Every cleaned name is classifier-scored even when the lexicon rule is negative."""
    tiny_config.inference.device = "cpu"
    tiny_config.inference.batch_size = 2
    _write_cleaned_frames(tiny_registry)
    _write_scoring_artifacts(tiny_registry)
    monkeypatch.setattr(
        "binary_classifier.inference.router.route",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("router called")),
    )

    score_names(tiny_config, tiny_registry, predictor=_TextScorePredictor())

    scores = pd.read_parquet(tiny_registry.names_scores)
    assert len(scores) == 4
    assert set(scores["EIN2"]) == {"P001", "B001"}
    assert scores.groupby("input_variant")["EIN2"].apply(set).to_dict() == {
        "suffix_retaining": {"P001", "B001"},
        "suffix_stripped": {"P001", "B001"},
    }
    assert scores["prob_raw"].notna().all()
    assert scores.loc[
        (scores["EIN2"] == "B001") & (scores["input_variant"] == "suffix_stripped"),
        "lexicon_rule_label",
    ].iloc[0] == 0
    assert scores.loc[scores["EIN2"] == "B001", "prob_raw"].tolist() == [0.2, 0.2]


def test_score_names_records_nontransferable_calibration_and_provenance(
    tiny_config,
    tiny_registry,
) -> None:
    """Names scores cannot be mistaken for calibrated mission probabilities."""
    tiny_config.inference.device = "cpu"
    _write_cleaned_frames(tiny_registry)
    _write_scoring_artifacts(tiny_registry)

    score_names(tiny_config, tiny_registry, predictor=_TextScorePredictor())

    scores = pd.read_parquet(tiny_registry.names_scores)
    assert scores["calibration_status"].eq("mission_calibration_invalid").all()
    assert not scores["thresholds_transferable"].any()
    assert scores["threshold"].eq(0.4).all()
    assert scores["threshold_maxf1"].eq(0.6).all()
    assert scores["threshold_baserate"].eq(0.8).all()
    assert scores["model_id"].eq("stub-model").all()
    assert scores["checkpoint_sha256"].eq("stub-sha").all()
    assert scores["inference_date"].astype(str).str.len().gt(0).all()
    assert scores["config_hash"].astype(str).str.len().eq(64).all()
    assert set(scores["name_input"]) == {
        "First Baptist Church",
        "First Baptist Church, Inc.",
        "Community Food Bank",
        "Community Food Bank Llc",
    }


def test_score_names_rejects_missing_checkpoint_provenance(
    tiny_config,
    tiny_registry,
) -> None:
    """An injected predictor cannot bypass the artifact provenance contract."""
    tiny_config.inference.device = "cpu"
    _write_cleaned_frames(tiny_registry)
    _write_mission_thresholds(tiny_registry)
    tiny_registry.selected_model.write_text(json.dumps({"encoder_id": "stub-model"}))

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        score_names(tiny_config, tiny_registry, predictor=_TextScorePredictor())


def test_score_names_reuses_one_predictor_cache_for_both_variants(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """Both variants share a checkpoint load through the public scorer seam."""
    _write_cleaned_frames(tiny_registry)
    _write_scoring_artifacts(tiny_registry)
    caches = []

    def fake_score(cfg, selected, texts, **kwargs):
        caches.append(kwargs["predictor_cache"])
        return [0.5] * len(texts)

    monkeypatch.setattr(score_mod, "score_texts", fake_score)

    score_names(tiny_config, tiny_registry, predictor=_TextScorePredictor())

    assert len(caches) == 2
    assert caches[0] is caches[1]


def _write_cleaned_frames(registry) -> None:
    pd.DataFrame(
        [
            {
                "EIN2": "P001",
                "population": "panel_scoped",
                "name_raw": "FIRST BAPTIST CHURCH, INC.",
                "name_cleaned": "First Baptist Church",
            },
        ],
    ).to_parquet(registry.names_panel_cleaned, index=False)
    pd.DataFrame(
        [
            {
                "EIN2": "B001",
                "population": "bmf_only",
                "name_raw": "COMMUNITY FOOD BANK LLC",
                "name_cleaned": "Community Food Bank",
            },
        ],
    ).to_parquet(registry.names_bmf_only_cleaned, index=False)


def _write_scoring_artifacts(registry) -> None:
    _write_selected_model(registry)
    _write_mission_thresholds(registry)


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


def _write_mission_thresholds(registry) -> None:
    registry.calibrator_path.write_text(
        json.dumps({"threshold": 0.4, "max_f1_threshold": 0.6}),
    )
    registry.base_rate_precision.write_text(json.dumps({"threshold": 0.8}))


class _TextScorePredictor:
    def predict_proba(self, texts):
        scores = [0.8 if "Baptist" in text else 0.2 for text in texts]
        return [[1.0 - score, score] for score in scores]
