"""Tests for encoder fine-tuning argument and loss plumbing."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from binary_classifier.train import encoder as encoder_mod


@pytest.mark.parametrize(
    ("cuda", "bf16", "mps", "expected_device", "expected_precision"),
    [
        (True, True, False, "cuda", "bf16"),
        (True, False, False, "cuda", "fp32"),
        (False, False, True, "mps", "fp32"),
        (False, False, False, "cpu", "fp32"),
    ],
)
def test_device_precision_resolution_matrix(
    monkeypatch,
    cuda,
    bf16,
    mps,
    expected_device,
    expected_precision,
) -> None:
    """Auto device and precision use capability checks only."""
    monkeypatch.setattr(encoder_mod.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(encoder_mod.torch.cuda, "is_bf16_supported", lambda: bf16)
    monkeypatch.setattr(encoder_mod.torch.backends.mps, "is_available", lambda: mps)

    device = encoder_mod.resolve_device("auto")

    assert device == expected_device
    assert encoder_mod.resolve_precision("auto", device) == expected_precision
    assert encoder_mod.resolve_device("cpu") == "cpu"
    assert encoder_mod.resolve_precision("bf16", "cpu") == "bf16"


def test_soft_ce_matches_hand_computation_with_num_items_normalization() -> None:
    """Soft CE divides by num_items_in_batch, not by tensor length implicitly."""
    logits = torch.tensor([[1.5, -0.5], [0.25, 1.25]], dtype=torch.float32)
    labels = torch.tensor([0.25, 0.75], dtype=torch.float32)
    outputs = SimpleNamespace(logits=logits)

    loss = encoder_mod.soft_ce(outputs, labels, num_items_in_batch=4)

    logp = torch.log_softmax(logits, dim=-1)
    expected = -(
        labels * logp[:, 1] + (1 - labels) * logp[:, 0]
    ).sum() / 4
    assert torch.isclose(loss, expected)


def test_weighted_soft_ce_closure_applies_inverse_frequency_terms() -> None:
    """The class-weighted closure uses the same soft-CE path with w0/w1 terms."""
    logits = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    outputs = {"logits": logits}
    w0, w1 = 1.5, 0.75

    loss_fn = encoder_mod._build_loss_func(class_weights=(w0, w1))
    loss = loss_fn(outputs, labels, 2)

    logp = torch.log_softmax(logits, dim=-1)
    expected = -(w1 * labels * logp[:, 1] + w0 * (1 - labels) * logp[:, 0]).sum() / 2
    assert torch.isclose(loss, expected)


def test_finetune_builds_training_args_and_run_layout(
    monkeypatch,
    tmp_path,
    tiny_config,
    tiny_registry,
) -> None:
    """Fine-tune wiring is offline-testable and writes per-run logs/metrics."""
    train_df = _frame(6)
    dev_df = _frame(4, start=10)
    validation_df = pd.DataFrame(
        {
            "EIN2": ["v0", "v1", "v2", "v3"],
            "text": ["validation zero", "validation one", "validation two", "validation three"],
            "human_label": [0, 1, 0, 1],
        },
    )
    captured = {}

    class FakeTokenizer:
        cls_token_id = 101
        sep_token_id = 102

        def encode(self, text, add_special_tokens=True):
            del text
            return [101, 7, 102] if add_special_tokens else [7]

        def __call__(self, texts, truncation=True, max_length=256):
            del truncation, max_length
            return {
                "input_ids": [[101, 5, 102] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

    class FakeModel:
        def to(self, device):
            captured["device"] = device
            return self

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class FakeEarlyStoppingCallback:
        def __init__(self, early_stopping_patience):
            captured["patience"] = early_stopping_patience

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def train(self):
            captured["trained"] = True

        def predict(self, dataset):
            labels = np.asarray(dataset["labels"], dtype=int)
            logits = np.column_stack([1 - labels, labels]).astype(float)
            return SimpleNamespace(predictions=logits)

    monkeypatch.setattr(
        encoder_mod.AutoTokenizer,
        "from_pretrained",
        lambda model_id: FakeTokenizer(),
    )
    monkeypatch.setattr(
        encoder_mod.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda model_id, num_labels: FakeModel(),
    )
    monkeypatch.setattr(encoder_mod, "DataCollatorWithPadding", lambda tokenizer: tokenizer)
    monkeypatch.setattr(encoder_mod, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(encoder_mod, "EarlyStoppingCallback", FakeEarlyStoppingCallback)
    monkeypatch.setattr(encoder_mod, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        encoder_mod,
        "load_human_split",
        lambda cfg, registry, split: validation_df,
    )
    tiny_config.training.device = "cpu"
    tiny_config.training.precision = "fp32"

    row = encoder_mod.finetune(
        tiny_config,
        tiny_registry,
        tiny_config.training.encoders[0].model_copy(update={"id": "fake/encoder"}),
        train_df,
        dev_df,
        targets="soft",
        arm="default",
        train_fraction=0.5,
        seed=123,
        run_root=tmp_path / "runs",
    )

    args = captured["training_args"]
    assert args["output_dir"] == str(tiny_registry.checkpoints_dir / "fake__encoder" / "default" / "s123")
    assert args["eval_strategy"] == "epoch"
    assert args["save_strategy"] == "epoch"
    assert args["warmup_ratio"] == tiny_config.training.warmup_fraction
    assert args["metric_for_best_model"] == "pr_auc"
    assert args["greater_is_better"] is True
    assert args["bf16"] is False
    assert args["report_to"] == "none"
    assert captured["trainer_kwargs"]["processing_class"].cls_token_id == 101
    assert callable(captured["trainer_kwargs"]["compute_loss_func"])
    assert callable(captured["trainer_kwargs"]["compute_metrics"])
    assert captured["patience"] == tiny_config.training.early_stopping_patience
    assert captured["trained"] is True
    assert row["run_id"] == "fake__encoder-soft-default-f0p5-s123"
    assert row["dev"]["pr_auc"] is not None
    assert row["validation"]["pr_auc"] is not None
    run_dir = tmp_path / "runs" / row["run_id"]
    assert (run_dir / "train.log").exists()
    assert (run_dir / "metrics.json").exists()


def test_finetune_predictor_uses_configured_warmup(
    monkeypatch,
    tmp_path,
    tiny_config,
    tiny_registry,
) -> None:
    """OOF fold training keeps scheduler warmup aligned with normal fine-tuning."""
    captured = {}

    class FakeTokenizer:
        cls_token_id = 101
        sep_token_id = 102

        def encode(self, text, add_special_tokens=True):
            del text
            return [101, 7, 102] if add_special_tokens else [7]

        def __call__(self, texts, truncation=True, max_length=256):
            del truncation, max_length
            return {
                "input_ids": [[101, 5, 102] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

    class FakeModel:
        def to(self, device):
            captured["device"] = device
            return self

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def train(self):
            captured["trained"] = True

    monkeypatch.setattr(
        encoder_mod.AutoTokenizer,
        "from_pretrained",
        lambda model_id: FakeTokenizer(),
    )
    monkeypatch.setattr(
        encoder_mod.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda model_id, num_labels: FakeModel(),
    )
    monkeypatch.setattr(encoder_mod, "DataCollatorWithPadding", lambda tokenizer: tokenizer)
    monkeypatch.setattr(encoder_mod, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(encoder_mod, "Trainer", FakeTrainer)
    tiny_config.training.device = "cpu"
    tiny_config.training.precision = "fp32"
    tiny_config.training.warmup_fraction = 0.17

    predictor = encoder_mod.finetune_predictor(
        tiny_config,
        tiny_registry,
        tiny_config.training.encoders[0].model_copy(update={"id": "fake/encoder"}),
        _frame(6),
        _frame(2, start=10),
        targets="soft",
        arm="default",
        train_fraction=1.0,
        seed=123,
        run_root=tmp_path / "runs",
    )

    args = captured["training_args"]
    assert args["output_dir"] == str(
        tmp_path / "runs" / "oof" / "fake__encoder-default-s123"
    )
    assert args["warmup_ratio"] == tiny_config.training.warmup_fraction
    assert args["save_strategy"] == "no"
    assert args["eval_strategy"] == "no"
    assert args["bf16"] is False
    assert captured["trainer_kwargs"]["processing_class"].cls_token_id == 101
    assert callable(captured["trainer_kwargs"]["compute_loss_func"])
    assert captured["trained"] is True
    assert isinstance(predictor, encoder_mod._FoldPredictor)


def _frame(n_rows: int, start: int = 0) -> pd.DataFrame:
    """Build a balanced tiny training/dev frame."""
    rows = []
    for i in range(start, start + n_rows):
        label = i % 2
        rows.append(
            {
                "EIN2": f"e{i}",
                "text": f"text {i}",
                "p_pos": float(label) * 0.8 + 0.1,
                "hard_label": label,
            },
        )
    return pd.DataFrame(rows)
