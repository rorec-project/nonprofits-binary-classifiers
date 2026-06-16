"""Encoder fine-tuning utilities for stage 06 training runs."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from binary_classifier import metrics
from binary_classifier.train.data import load_human_split

if TYPE_CHECKING:
    from collections.abc import Iterator

    from binary_classifier.config import BinaryClassifierConfig, EncoderArm
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)


def resolve_device(knob: str) -> str:
    """Resolve the training device from the configured runtime knob.

    Args:
        knob: ``"auto"`` or an explicit device name.

    Returns:
        ``"cuda"``, ``"mps"``, ``"cpu"``, or the explicit device name.

    """
    if knob != "auto":
        return knob
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def resolve_precision(knob: str, device: str) -> str:
    """Resolve the numeric precision from capability checks only.

    Args:
        knob: ``"auto"`` or an explicit precision name.
        device: Already-resolved device name.

    Returns:
        ``"bf16"`` for CUDA devices with bf16 support, otherwise ``"fp32"``
        when the knob is ``"auto"``; explicit knobs are returned unchanged.

    """
    if knob != "auto":
        return knob
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def soft_ce(
    outputs: Any, labels: torch.Tensor, num_items_in_batch: Any
) -> torch.Tensor:
    """Compute soft-label cross entropy normalized by batch item count.

    Args:
        outputs: Model output with a ``logits`` attribute or key.
        labels: Float tensor containing positive-class probabilities.
        num_items_in_batch: Number of items contributing to the loss.

    Returns:
        Scalar loss tensor.

    """
    return _soft_ce_weighted(
        outputs,
        labels,
        num_items_in_batch,
        class_weights=None,
    )


def finetune(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    encoder: EncoderArm,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    *,
    targets: str,
    arm: str,
    train_fraction: float,
    seed: int,
    run_root: Path | str | None = None,
) -> dict[str, Any]:
    """Fine-tune an encoder arm and return one ``results.jsonl`` row.

    Args:
        cfg: Validated task configuration.
        registry: Path registry that owns model run/checkpoint directories.
        encoder: Hugging Face encoder arm to fine-tune.
        train_df: Silver training frame with text and target columns.
        dev_df: Silver development frame used for checkpoint selection.
        targets: ``"soft"`` for ``p_pos`` targets or ``"hard"`` for labels.
        arm: Training-data arm name (``default``, ``hard``,
            ``class_weighted``, etc.).
        train_fraction: Fraction of available training rows used in this run.
        seed: Explicit run seed.
        run_root: Optional override for the run-artifact root, primarily for
            tests; defaults to ``registry.runs_dir``.

    Returns:
        Dict conforming to the stage-06 ``results.jsonl`` row schema.

    """
    start = perf_counter()
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = resolve_device(cfg.training.device)
    precision = resolve_precision(cfg.training.precision, device)

    slug = _model_slug(encoder.id)
    run_id = _run_id(slug, targets, arm, train_fraction, seed)
    runs_base = Path(run_root) if run_root is not None else registry.runs_dir
    run_dir = runs_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = registry.checkpoints_dir / slug / arm / f"s{seed}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with _file_logging(run_dir / "train.log"):
        logger.info("Starting encoder fine-tune run %s", run_id)
        tokenizer: Any = AutoTokenizer.from_pretrained(encoder.id)
        _assert_cls_sep_startup(tokenizer)
        model: Any = AutoModelForSequenceClassification.from_pretrained(
            encoder.id,
            num_labels=2,
        )
        model.to(device)

        train_dataset = _dataset_from_frame(
            train_df,
            tokenizer,
            max_length=encoder.max_length,
            targets=targets,
            arm=arm,
        )
        dev_dataset = _dataset_from_frame(
            dev_df,
            tokenizer,
            max_length=encoder.max_length,
            targets=targets,
            arm=arm,
        )
        data_collator = DataCollatorWithPadding(tokenizer=cast(Any, tokenizer))
        class_weights = _class_weights(train_df) if arm == "class_weighted" else None
        compute_loss = _build_loss_func(class_weights=class_weights)

        args = TrainingArguments(
            output_dir=str(checkpoints_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=cfg.training.metric_for_best_model,
            greater_is_better=cfg.training.greater_is_better,
            warmup_steps=cfg.training.warmup_fraction,
            learning_rate=cfg.training.learning_rate,
            num_train_epochs=cfg.training.epochs,
            per_device_train_batch_size=cfg.training.batch_size,
            weight_decay=cfg.training.weight_decay,
            save_total_limit=cfg.training.save_total_limit,
            bf16=(precision == "bf16"),
            report_to=cfg.training.report_to or "none",
            disable_tqdm=True,
            logging_strategy="steps",
            logging_steps=50,
            seed=seed,
            data_seed=seed,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_loss_func=compute_loss,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.training.early_stopping_patience,
                ),
            ],
        )
        trainer.train()

        dev_scores = _positive_probs(
            trainer.predict(cast(Any, dev_dataset)).predictions
        )
        dev_metrics = _score_frame(dev_df, dev_scores, seed)

        validation_df = load_human_split(cfg, registry, "validation")
        validation_dataset = _dataset_from_human_frame(
            validation_df,
            tokenizer,
            max_length=encoder.max_length,
        )
        validation_scores = _positive_probs(
            trainer.predict(cast(Any, validation_dataset)).predictions,
        )
        validation_metrics = _score_frame(validation_df, validation_scores, seed)

        wall_seconds = perf_counter() - start
        row: dict[str, Any] = {
            "run_id": run_id,
            "model": encoder.id,
            "targets": targets,
            "arm": arm,
            "train_fraction": float(train_fraction),
            "n_train": int(len(train_df)),
            "seed": int(seed),
            "dev": dev_metrics,
            "validation": validation_metrics,
            "wall_seconds": float(wall_seconds),
            "precision": precision,
            "device": device,
            "git_sha": "unknown",
            "config_hash": "unknown",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        (run_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        logger.info("Finished encoder fine-tune run %s", run_id)
        return row


def compute_metrics(eval_pred: transformers.EvalPrediction) -> dict[str, float]:
    """Compute checkpoint-selection metrics for Hugging Face evaluation.

    Args:
        eval_pred: Evaluation prediction tuple whose ``predictions`` are logits
            and whose ``label_ids`` are float labels.

    Returns:
        Dict containing ``pr_auc``, ``minority_f1``, and ``mcc``.

    """
    y_score = _positive_probs(eval_pred.predictions)
    y_true = _binarize(eval_pred.label_ids)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "minority_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _build_loss_func(
    *,
    class_weights: tuple[float, float] | None,
) -> Callable[[Any, torch.Tensor, Any], torch.Tensor]:
    """Return the loss closure passed to ``Trainer``."""

    def compute_loss_func(
        outputs: Any,
        labels: torch.Tensor,
        num_items_in_batch: Any,
    ) -> torch.Tensor:
        return _soft_ce_weighted(outputs, labels, num_items_in_batch, class_weights)

    return compute_loss_func


def _soft_ce_weighted(
    outputs: Any,
    labels: torch.Tensor,
    num_items_in_batch: Any,
    class_weights: tuple[float, float] | None,
) -> torch.Tensor:
    """Shared soft-CE implementation with optional class weights."""
    logits = _logits(outputs)
    labels = labels.to(device=logits.device, dtype=logits.dtype).view(-1)
    logp = torch.log_softmax(logits, dim=-1)
    positive = labels * logp[:, 1]
    negative = (1 - labels) * logp[:, 0]
    if class_weights is not None:
        w0, w1 = class_weights
        positive = positive * torch.as_tensor(
            w1, device=logits.device, dtype=logits.dtype
        )
        negative = negative * torch.as_tensor(
            w0, device=logits.device, dtype=logits.dtype
        )
    divisor = torch.as_tensor(
        num_items_in_batch, device=logits.device, dtype=logits.dtype
    )
    return -(positive + negative).sum() / divisor


def _logits(outputs: Any) -> torch.Tensor:
    """Return logits from a transformers output or dict-like stub."""
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


def _dataset_from_frame(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    max_length: int,
    targets: str,
    arm: str,
) -> Dataset:
    """Build a tokenized HF dataset from a silver training/dev frame."""
    _require_columns(frame, {"text", "p_pos", "hard_label"}, "training frame")
    if targets == "hard" or arm == "hard":
        labels = pd.to_numeric(frame["hard_label"], errors="coerce").astype(float)
    elif targets == "soft":
        labels = pd.to_numeric(frame["p_pos"], errors="coerce").astype(float)
    else:
        raise ValueError("targets must be 'soft' or 'hard'.")
    dataset = Dataset.from_dict(
        {
            "text": frame["text"].fillna("").astype(str).tolist(),
            "labels": labels.tolist(),
        },
    )
    return _tokenize_dataset(dataset, tokenizer, max_length=max_length)


def _dataset_from_human_frame(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    max_length: int,
) -> Dataset:
    """Build a tokenized HF dataset from a human validation frame."""
    _require_columns(frame, {"text", "human_label"}, "human validation frame")
    labels = pd.to_numeric(frame["human_label"], errors="coerce").astype(float)
    dataset = Dataset.from_dict(
        {
            "text": frame["text"].fillna("").astype(str).tolist(),
            "labels": labels.tolist(),
        },
    )
    return _tokenize_dataset(dataset, tokenizer, max_length=max_length)


def _tokenize_dataset(dataset: Dataset, tokenizer: Any, *, max_length: int) -> Dataset:
    """Apply tokenizer with truncation and remove raw text."""

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def _score_frame(frame: pd.DataFrame, y_score: np.ndarray, seed: int) -> dict[str, Any]:
    """Score positive-class probabilities against hard or human labels."""
    if "hard_label" in frame.columns:
        label_column = "hard_label"
    elif "human_label" in frame.columns:
        label_column = "human_label"
    else:
        raise ValueError("frame must contain hard_label or human_label.")
    y_true = pd.to_numeric(frame[label_column], errors="coerce").astype(int).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    return metrics.compute_metric_bundle(y_true, y_pred, y_score=y_score, seed=seed)


def _positive_probs(predictions: Any) -> np.ndarray:
    """Convert logits to positive-class probabilities."""
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    logits_arr = np.asarray(logits, dtype=float)
    logits_arr = logits_arr - logits_arr.max(axis=1, keepdims=True)
    exp = np.exp(logits_arr)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1]


def _binarize(labels: Any) -> np.ndarray:
    """Binarize soft labels at 0.5 for validation metrics."""
    return (np.asarray(labels, dtype=float) >= 0.5).astype(int)


def _class_weights(frame: pd.DataFrame) -> tuple[float, float]:
    """Compute inverse-frequency weights ``(w0, w1)`` from hard labels."""
    _require_columns(frame, {"hard_label"}, "training frame")
    labels = pd.to_numeric(frame["hard_label"], errors="coerce").astype(int)
    counts = labels.value_counts().to_dict()
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))
    if n0 == 0 or n1 == 0:
        raise ValueError("class_weighted arm requires both hard-label classes.")
    total = n0 + n1
    return total / (2 * n0), total / (2 * n1)


def _assert_cls_sep_startup(tokenizer: Any) -> None:
    """Assert that the tokenizer wraps text with CLS and SEP special tokens."""
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if cls_id is None or sep_id is None:
        raise ValueError("Encoder tokenizer must define CLS and SEP token ids.")
    token_ids = tokenizer.encode("startup sanity", add_special_tokens=True)
    if not token_ids or token_ids[0] != cls_id or token_ids[-1] != sep_id:
        raise ValueError("Encoder tokenizer must add CLS at start and SEP at end.")


@contextmanager
def _file_logging(path: Path) -> Iterator[None]:
    """Attach a per-run file handler to package and transformers loggers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )
    run_loggers = [
        logging.getLogger("transformers"),
        logging.getLogger("binary_classifier"),
    ]
    old_state = [(log, log.level, log.propagate) for log in run_loggers]
    try:
        for log in run_loggers:
            log.addHandler(handler)
            log.setLevel(logging.INFO)
            log.propagate = False
        yield
    finally:
        for log, old_level, old_propagate in old_state:
            log.removeHandler(handler)
            log.setLevel(old_level)
            log.propagate = old_propagate
        handler.close()


def _run_id(slug: str, targets: str, arm: str, train_fraction: float, seed: int) -> str:
    """Build a stable run identifier for an encoder fine-tune."""
    fraction = str(train_fraction).replace(".", "p")
    return f"{slug}-{targets}-{arm}-f{fraction}-s{seed}"


def _model_slug(model_id: str) -> str:
    """Return a filesystem-safe slug for a Hugging Face model id."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id).strip("_")


def _require_columns(df: pd.DataFrame, required: set[str], frame_name: str) -> None:
    """Raise when a DataFrame lacks required columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {sorted(missing)}.")
