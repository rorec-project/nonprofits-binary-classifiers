"""Tests for stage 08 inference sharding and monitor scoring."""

from __future__ import annotations

import json

import pandas as pd

from binary_classifier.inference import predict as predict_mod
from binary_classifier.inference.predict import run_inference


def test_resolve_device_precision_encoder_override(tiny_config) -> None:
    import torch

    from binary_classifier.inference.predict import resolve_device_precision

    original_cuda_available = torch.cuda.is_available
    original_bf16_supported = torch.cuda.is_bf16_supported

    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.is_bf16_supported = lambda: True

        # No override - should be bf16
        device, precision = resolve_device_precision(tiny_config)
        assert precision == "bf16"

        # Encoder without explicit precision - no override
        device, precision = resolve_device_precision(
            tiny_config, encoder_id="answerdotai/ModernBERT-base"
        )
        assert precision == "bf16"

        # Set DeBERTa to have explicit fp32, matching the production config
        for enc in tiny_config.training.encoders:
            if enc.id == "microsoft/deberta-v3-base":
                enc.precision = "fp32"

        device, precision = resolve_device_precision(
            tiny_config, encoder_id="microsoft/deberta-v3-base"
        )
        assert precision == "fp32"

        # Non-existent encoder - no override
        device, precision = resolve_device_precision(
            tiny_config, encoder_id="nonexistent/model"
        )
        assert precision == "bf16"
    finally:
        torch.cuda.is_available = original_cuda_available
        torch.cuda.is_bf16_supported = original_bf16_supported


def test_run_inference_writes_schema_rules_monitor_and_metadata(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 3
    tiny_config.inference.batch_size = 2
    missions = _missions_frame()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_selected_model(tiny_registry)
    _write_monitor(tiny_registry, ["E0002", "E0004"])

    run_inference(tiny_config, tiny_registry, predictor=_StubPredictor())

    predictions = pd.read_parquet(tiny_registry.predictions_parquet)
    assert predictions.columns.tolist() == _prediction_columns()
    assert set(predictions["EIN2"]) == set(missions["EIN2"].astype(str))
    assert len(predictions) == len(missions)
    assert set(predictions["decision_source"]) >= {
        "classifier",
        "rule_strong_positive",
        "rule_short_negative",
        "low_via_classifier",
    }

    rule_rows = predictions[
        predictions["decision_source"].isin(
            ["rule_strong_positive", "rule_short_negative"]
        )
    ]
    assert rule_rows["prob_raw"].isna().all()
    assert rule_rows["prob_calibrated"].isna().all()
    assert predictions.loc[
        predictions["decision_source"] == "rule_strong_positive", "pred_label"
    ].eq(1).all()
    assert predictions.loc[
        predictions["decision_source"] == "rule_short_negative", "pred_label"
    ].eq(0).all()
    assert predictions["model_id"].eq("stub-model").all()
    assert predictions["checkpoint_sha256"].eq("stub-sha").all()
    assert predictions["calibrator_method"].eq("platt").all()
    assert predictions["calibrator_params_hash"].astype(str).str.len().eq(64).all()
    assert predictions["threshold"].eq(0.5).all()
    assert predictions["threshold_maxf1"].eq(0.7).all()
    assert predictions["threshold_baserate"].eq(0.9).all()
    assert predictions["inference_date"].astype(str).str.len().gt(0).all()
    assert predictions["pipeline_version"].astype(str).str.len().gt(0).all()
    assert predictions["config_hash"].astype(str).str.len().eq(64).all()

    monitor = json.loads(tiny_registry.monitor_scores.read_text())
    assert monitor["metadata"]["n_monitor"] == 2
    assert {row["EIN2"] for row in monitor["rows"]} == {"E0002", "E0004"}
    assert monitor["metadata"]["calibrator_method"] == "platt"
    assert {"pred_label_maxf1", "pred_label_baserate"} <= set(monitor["rows"][0])


def test_predict_shard_release_labels_use_thresholds_for_classifier_rows(
    tiny_config,
) -> None:
    shard = pd.DataFrame(
        [
            _classifier_row("A001", "alpha"),
            _classifier_row("A002", "beta"),
        ]
    )
    predictions = predict_mod._predict_shard(
        tiny_config,
        shard,
        predictor=_FixedPredictor([0.65, 0.85]),
        calibrator=_calibrator_payload(),
        metadata=_metadata_payload(tiny_config),
        batch_size=2,
        device="cpu",
        precision="fp32",
    )

    assert predictions["pred_label"].tolist() == [1, 1]
    assert predictions["pred_label_maxf1"].tolist() == [0, 1]
    assert predictions["pred_label_baserate"].tolist() == [0, 0]


def test_predict_shard_release_labels_equal_pred_label_for_rule_rows(
    tiny_config,
) -> None:
    shard = pd.DataFrame(
        [
            _rule_row("R001", "rule_strong_positive", 1),
            _rule_row("R002", "rule_short_negative", 0),
            _rule_row("R003", "rule_abstain", None),
        ]
    )
    predictions = predict_mod._predict_shard(
        tiny_config,
        shard,
        predictor=_FixedPredictor([]),
        calibrator=_calibrator_payload(),
        metadata=_metadata_payload(tiny_config),
        batch_size=2,
        device="cpu",
        precision="fp32",
    )

    assert predictions["pred_label"].tolist() == [1, 0, 0]
    assert predictions["pred_label_maxf1"].tolist() == predictions["pred_label"].tolist()
    assert predictions["pred_label_baserate"].tolist() == predictions["pred_label"].tolist()


def test_existing_shard_rejects_changed_release_threshold(tiny_config, tmp_path) -> None:
    shard_path = tmp_path / "shard_00000.parquet"
    _prewrite_matching_metadata_shard(tiny_config, shard_path, ["A000", "A001"])
    metadata = _metadata_payload(tiny_config)
    metadata["threshold_baserate"] = 0.95

    assert not predict_mod._existing_shard_matches(
        shard_path,
        metadata,
        ["A000", "A001"],
    )


def test_stale_shards_removed_matching_shards_resumed(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 2
    tiny_config.inference.batch_size = 2
    missions = _classifier_only_missions()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_monitor(tiny_registry, missions["EIN2"].tolist())
    _prewrite_matching_metadata_shard(
        tiny_config,
        tiny_registry.predictions_dir / "shards" / "shard_00000.parquet",
        ["A000", "A001"],
    )
    stale_path = tiny_registry.predictions_dir / "shards" / "shard_00002.parquet"
    _prewrite_matching_metadata_shard(tiny_config, stale_path, ["Z000"])
    predictor = _CountingPredictor()

    run_inference(tiny_config, tiny_registry, predictor=predictor)

    assert not stale_path.exists()
    assert predictor.n_scored == 2


def test_limited_run_does_not_delete_stale_shards(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 2
    tiny_config.inference.batch_size = 2
    missions = _classifier_only_missions()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_monitor(tiny_registry, ["A000", "A001"])
    stale_path = tiny_registry.predictions_dir / "shards" / "shard_00002.parquet"
    _prewrite_matching_metadata_shard(tiny_config, stale_path, ["Z000"])

    run_inference(tiny_config, tiny_registry, predictor=_CountingPredictor(), limit=2)

    assert stale_path.exists()


def test_run_inference_rewrites_existing_shards_with_stale_metadata(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 2
    tiny_config.inference.batch_size = 2
    missions = _classifier_only_missions()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_monitor(tiny_registry, missions["EIN2"].tolist())

    shard_path = tiny_registry.predictions_dir / "shards" / "shard_00000.parquet"
    _prewrite_first_shard(shard_path)
    predictor = _CountingPredictor()

    run_inference(tiny_config, tiny_registry, predictor=predictor)

    predictions = pd.read_parquet(tiny_registry.predictions_parquet)
    assert set(predictions["EIN2"]) == set(missions["EIN2"])
    assert len(predictions) == len(missions)
    assert predictions["EIN2"].duplicated().sum() == 0
    assert predictor.n_scored == 4
    rewritten = predictions[predictions["EIN2"].isin(["A000", "A001"])]
    assert rewritten["model_id"].eq("unknown").all()
    assert rewritten["checkpoint_sha256"].eq("unknown").all()


def test_run_inference_rewrites_existing_shards_with_wrong_ein2_slice(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 2
    tiny_config.inference.batch_size = 2
    missions = _classifier_only_missions()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_monitor(tiny_registry, missions["EIN2"].tolist())

    shard_path = tiny_registry.predictions_dir / "shards" / "shard_00000.parquet"
    _prewrite_matching_metadata_shard(tiny_config, shard_path, ["A002", "A003"])
    predictor = _CountingPredictor()

    run_inference(tiny_config, tiny_registry, predictor=predictor)

    predictions = pd.read_parquet(tiny_registry.predictions_parquet)
    assert predictions["EIN2"].tolist() == ["A000", "A001", "A002", "A003"]
    assert predictor.n_scored == 4


def test_run_inference_rewrites_existing_shards_with_wrong_row_count(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 2
    tiny_config.inference.batch_size = 2
    missions = _classifier_only_missions()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_monitor(tiny_registry, missions["EIN2"].tolist())

    shard_path = tiny_registry.predictions_dir / "shards" / "shard_00000.parquet"
    _prewrite_matching_metadata_shard(tiny_config, shard_path, ["A000"])
    predictor = _CountingPredictor()

    run_inference(tiny_config, tiny_registry, predictor=predictor)

    predictions = pd.read_parquet(tiny_registry.predictions_parquet)
    assert predictions["EIN2"].tolist() == ["A000", "A001", "A002", "A003"]
    assert predictor.n_scored == 4


def test_run_inference_rewrites_existing_shards_with_stale_pipeline_version(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 2
    tiny_config.inference.batch_size = 2
    missions = _classifier_only_missions()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_monitor(tiny_registry, missions["EIN2"].tolist())

    shard_path = tiny_registry.predictions_dir / "shards" / "shard_00000.parquet"
    _prewrite_matching_metadata_shard(tiny_config, shard_path, ["A000", "A001"])
    stale = pd.read_parquet(shard_path)
    stale["pipeline_version"] = "stale-version"
    stale.to_parquet(shard_path, index=False)
    predictor = _CountingPredictor()

    run_inference(tiny_config, tiny_registry, predictor=predictor)

    assert predictor.n_scored == 4


def test_run_inference_rewrites_existing_shards_with_extra_columns(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 2
    tiny_config.inference.batch_size = 2
    missions = _classifier_only_missions()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_monitor(tiny_registry, missions["EIN2"].tolist())

    shard_path = tiny_registry.predictions_dir / "shards" / "shard_00000.parquet"
    _prewrite_matching_metadata_shard(tiny_config, shard_path, ["A000", "A001"])
    extra = pd.read_parquet(shard_path)
    extra["unexpected"] = "extra"
    extra.to_parquet(shard_path, index=False)
    predictor = _CountingPredictor()

    run_inference(tiny_config, tiny_registry, predictor=predictor)

    assert predictor.n_scored == 4


def test_run_inference_rewrites_existing_shards_with_stale_calibrator_params(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    tiny_config.inference.shard_size = 2
    tiny_config.inference.batch_size = 2
    missions = _classifier_only_missions()
    monkeypatch.setattr(predict_mod, "load_missions", lambda cfg, **kwargs: missions)
    _write_calibrator(tiny_registry)
    _write_monitor(tiny_registry, missions["EIN2"].tolist())

    shard_path = tiny_registry.predictions_dir / "shards" / "shard_00000.parquet"
    _prewrite_matching_metadata_shard(tiny_config, shard_path, ["A000", "A001"])
    stale = pd.read_parquet(shard_path)
    stale["calibrator_params_hash"] = "0" * 64
    stale.to_parquet(shard_path, index=False)
    predictor = _CountingPredictor()

    run_inference(tiny_config, tiny_registry, predictor=predictor)

    assert predictor.n_scored == 4


def _missions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "EIN2": "E0004",
                "mission_text": "to worship and teach children through ministry, training, and support programs",
                "ntee_major_group": "X",
                "is_truncated": False,
                "NTEE_IRS": "X20",
                "data_source": "synthetic",
            },
            {
                "EIN2": "E0001",
                "mission_text": "church",
                "ntee_major_group": "X",
                "is_truncated": False,
                "NTEE_IRS": "X20",
                "data_source": "synthetic",
            },
            {
                "EIN2": "E0003",
                "mission_text": "food",
                "ntee_major_group": "K",
                "is_truncated": False,
                "NTEE_IRS": "K20",
                "data_source": "synthetic",
            },
            {
                "EIN2": "E0002",
                "mission_text": "alpha beta gamma delta epsilon zeta",
                "ntee_major_group": "A",
                "is_truncated": False,
                "NTEE_IRS": "A20",
                "data_source": "synthetic",
            },
            {
                "EIN2": "E0005",
                "mission_text": "to provide food education and housing for families through training and services",
                "ntee_major_group": "P",
                "is_truncated": False,
                "NTEE_IRS": "P20",
                "data_source": "synthetic",
            },
        ]
    )


def _classifier_only_missions() -> pd.DataFrame:
    rows = []
    for i in range(4):
        rows.append(
            {
                "EIN2": f"A{i:03d}",
                "mission_text": (
                    "to provide food education and housing for families "
                    f"through training and services {i}"
                ),
                "ntee_major_group": "P",
                "is_truncated": False,
                "NTEE_IRS": "P20",
                "data_source": "synthetic",
            }
        )
    return pd.DataFrame(rows)


def _write_calibrator(registry) -> None:
    registry.calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    registry.calibrator_path.write_text(
        json.dumps(_calibrator_payload())
    )
    registry.base_rate_precision.write_text(json.dumps({"threshold": 0.9}))


def _calibrator_payload() -> dict:
    return {
        "method": "platt",
        "params": {"a": 1.0, "b": 0.0},
        "threshold": 0.5,
        "max_f1_threshold": 0.7,
        "threshold_baserate": 0.9,
    }


def _metadata_payload(tiny_config) -> dict:
    return {
        "model_id": "unknown",
        "checkpoint_sha256": "unknown",
        "calibrator_method": "platt",
        "calibrator_params_hash": predict_mod._calibrator_params_hash(
            {"params": {"a": 1.0, "b": 0.0}},
        ),
        "threshold": 0.5,
        "threshold_maxf1": 0.7,
        "threshold_baserate": 0.9,
        "inference_date": "2026-01-01T00:00:00+00:00",
        "pipeline_version": predict_mod._pipeline_version(),
        "config_hash": predict_mod._config_hash(tiny_config),
    }


def _write_selected_model(registry) -> None:
    registry.selected_model.parent.mkdir(parents=True, exist_ok=True)
    registry.selected_model.write_text(
        json.dumps(
            {
                "encoder_id": "stub-model",
                "tokenizer_id": "stub-model",
                "checkpoint_sha256": "stub-sha",
                "checkpoint_relpath": "unused/model.safetensors",
            }
        )
    )


def _write_monitor(registry, ein2s: list[str]) -> None:
    registry.monitor_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"EIN2": ein2s}).to_csv(registry.monitor_manifest, index=False)


def _prewrite_first_shard(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            _prediction_row("A000", 0.1, 0),
            _prediction_row("A001", 0.9, 1),
        ],
        columns=_prediction_columns(),
    ).to_parquet(path, index=False)


def _prewrite_matching_metadata_shard(tiny_config, path, ein2s: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ein2 in ein2s:
        row = _prediction_row(ein2, 0.8, 1)
        row["model_id"] = "unknown"
        row["checkpoint_sha256"] = "unknown"
        row["calibrator_params_hash"] = predict_mod._calibrator_params_hash(
            {"params": {"a": 1.0, "b": 0.0}},
        )
        row["pipeline_version"] = predict_mod._pipeline_version()
        row["config_hash"] = predict_mod._config_hash(tiny_config)
        rows.append(row)
    pd.DataFrame(rows, columns=_prediction_columns()).to_parquet(path, index=False)


def _prediction_row(ein2: str, prob: float, label: int) -> dict:
    return {
        "EIN2": ein2,
        "pred_label": label,
        "pred_label_maxf1": label,
        "pred_label_baserate": label,
        "prob_raw": prob,
        "prob_calibrated": prob,
        "decision_source": "classifier",
        "tier": "HIGH",
        "Q": 5.0,
        "ntee_major_group": "P",
        "model_id": "preexisting",
        "checkpoint_sha256": "preexisting-sha",
        "calibrator_method": "platt",
        "calibrator_params_hash": "preexisting-params-sha",
        "threshold": 0.5,
        "threshold_maxf1": 0.7,
        "threshold_baserate": 0.9,
        "inference_date": "2026-01-01T00:00:00+00:00",
        "pipeline_version": "test",
        "config_hash": "0" * 64,
    }


def _prediction_columns() -> list[str]:
    return [
        "EIN2",
        "pred_label",
        "pred_label_maxf1",
        "pred_label_baserate",
        "prob_raw",
        "prob_calibrated",
        "decision_source",
        "tier",
        "Q",
        "ntee_major_group",
        "model_id",
        "checkpoint_sha256",
        "calibrator_method",
        "calibrator_params_hash",
        "threshold",
        "threshold_maxf1",
        "threshold_baserate",
        "inference_date",
        "pipeline_version",
        "config_hash",
    ]


class _StubPredictor:
    def predict_proba(self, texts):
        scores = [0.85 if "worship" in text or "provide" in text else 0.25 for text in texts]
        return [[1.0 - score, score] for score in scores]


class _CountingPredictor:
    def __init__(self) -> None:
        self.n_scored = 0

    def predict_proba(self, texts):
        self.n_scored += len(texts)
        return [[0.2, 0.8] for _ in texts]


class _FixedPredictor:
    def __init__(self, scores) -> None:
        self.scores = list(scores)

    def predict_proba(self, texts):
        assert len(texts) == len(self.scores)
        return [[1.0 - score, score] for score in self.scores]


def _classifier_row(ein2: str, text: str) -> dict:
    return {
        "EIN2": ein2,
        "mission_text": text,
        "decision_source": "classifier",
        "rule_label": None,
        "tier": "HIGH",
        "Q": 5.0,
        "ntee_major_group": "P",
    }


def _rule_row(ein2: str, source: str, label: int | None) -> dict:
    row = _classifier_row(ein2, "rule text")
    row["decision_source"] = source
    row["rule_label"] = label
    return row
