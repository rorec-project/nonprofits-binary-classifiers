"""Classical and frozen-embedding baselines for training stage comparisons."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import AutoModel, AutoTokenizer, set_seed

from binary_classifier import metrics

if TYPE_CHECKING:
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

_DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EmbeddingResult:
    """Text embedding matrix plus runtime metadata.

    Args:
        embeddings: Two-dimensional matrix aligned to the input text order.
        device: Runtime device used to produce embeddings.
        precision: Numeric precision used for model inference.

    """

    embeddings: np.ndarray
    device: str
    precision: str


def tfidf_logreg(
    train_df: pd.DataFrame,
    eval_dfs: Mapping[str, pd.DataFrame],
    seed: int,
) -> dict[str, Any]:
    """Train and score a TF-IDF + logistic-regression baseline.

    Args:
        train_df: Training frame containing ``text`` and ``hard_label``.
        eval_dfs: Named evaluation frames with ``text`` and either
            ``hard_label`` or ``human_label``.
        seed: Random seed passed to the logistic-regression solver and metrics.

    Returns:
        Results row conforming to the training ``results.jsonl`` schema.

    """
    start = perf_counter()
    train_text, y_train = _training_xy(train_df)
    clf = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(analyzer="word", ngram_range=(1, 2)),
                        ),
                        (
                            "char_wb",
                            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
                        ),
                    ],
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ],
    )
    clf.fit(train_text, y_train)

    eval_metrics = {
        name: _score_predict_proba(
            frame,
            np.asarray(clf.predict_proba(_text_series(frame))[:, 1], dtype=float),
            seed,
        )
        for name, frame in eval_dfs.items()
    }
    wall_seconds = perf_counter() - start
    logger.info("Trained TF-IDF logistic baseline on %d rows", len(train_df))
    return _result_row(
        model="baseline:tfidf_logreg",
        n_train=len(train_df),
        seed=seed,
        eval_metrics=eval_metrics,
        wall_seconds=wall_seconds,
        device="cpu",
        precision="fp32",
    )


def minilm_logreg(
    train_df: pd.DataFrame,
    eval_dfs: Mapping[str, pd.DataFrame],
    seed: int,
    registry: PathRegistry,
    model_name: str = _DEFAULT_MINILM_MODEL,
    device: str = "auto",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Train and score a MiniLM-embedding + logistic-regression baseline.

    Embeddings are generated with plain ``transformers`` and cached under
    ``registry.embeddings_dir``. Cache hits are accepted only when the saved EIN2
    index exactly matches the requested ordered EIN2s.

    Args:
        train_df: Training frame containing ``EIN2``, ``text``, and
            ``hard_label``.
        eval_dfs: Named evaluation frames containing ``EIN2``, ``text``, and
            either ``hard_label`` or ``human_label``.
        seed: Random seed passed to transformers, logistic regression, and
            metrics.
        registry: Path registry whose ``embeddings_dir`` stores the cache.
        model_name: Hugging Face model identifier for the MiniLM encoder.
        device: ``"auto"`` or an explicit torch device string.
        batch_size: Inference batch size for transformer embedding.

    Returns:
        Results row conforming to the training ``results.jsonl`` schema.

    """
    start = perf_counter()
    set_seed(seed)
    _, y_train = _training_xy(train_df)
    embedding_frame = _collect_embedding_frame(train_df, eval_dfs)
    result = _load_or_compute_embeddings(
        embedding_frame=embedding_frame,
        registry=registry,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    embedding_lookup = {
        ein2: result.embeddings[i]
        for i, ein2 in enumerate(embedding_frame["EIN2"].astype(str).tolist())
    }

    train_embeddings = _embeddings_for_frame(train_df, embedding_lookup)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(train_embeddings, y_train)

    eval_metrics = {}
    for name, frame in eval_dfs.items():
        eval_embeddings = _embeddings_for_frame(frame, embedding_lookup)
        y_score = np.asarray(clf.predict_proba(eval_embeddings)[:, 1], dtype=float)
        eval_metrics[name] = _score_predict_proba(frame, y_score, seed)

    wall_seconds = perf_counter() - start
    logger.info("Trained MiniLM logistic baseline on %d rows", len(train_df))
    return _result_row(
        model="baseline:minilm_logreg",
        n_train=len(train_df),
        seed=seed,
        eval_metrics=eval_metrics,
        wall_seconds=wall_seconds,
        device=result.device,
        precision=result.precision,
    )


def compute_minilm_embeddings(
    texts: Sequence[str],
    model_name: str = _DEFAULT_MINILM_MODEL,
    device: str = "auto",
    batch_size: int = 32,
) -> EmbeddingResult:
    """Embed texts with a Hugging Face AutoModel and mean pooling.

    Args:
        texts: Input texts in embedding order.
        model_name: Hugging Face model identifier.
        device: ``"auto"`` or an explicit torch device string. ``"auto"``
            resolves by runtime capability checks: CUDA, then MPS, then CPU.
        batch_size: Number of texts per transformer inference batch.

    Returns:
        Embedding matrix and runtime metadata.

    """
    resolved_device = _resolve_device(device)
    tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
    model: Any = AutoModel.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tokenize: Any = getattr(tokenizer, "__call__")
        encoded = tokenize(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        batches.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))

    if batches:
        embeddings = np.concatenate(batches, axis=0)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)
    return EmbeddingResult(
        embeddings=embeddings,
        device=resolved_device,
        precision="fp32",
    )


def _training_xy(train_df: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Return validated training text and hard labels."""
    _require_columns(train_df, {"text", "hard_label"}, "train_df")
    text = _text_series(train_df)
    labels = _labels(train_df, require_hard=True)
    if len(np.unique(labels)) != 2:
        raise ValueError("train_df hard_label must contain both binary classes.")
    return text, labels


def _score_predict_proba(
    eval_df: pd.DataFrame,
    y_score: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Compute the shared metric bundle from positive-class scores."""
    y_true = _labels(eval_df, require_hard=False)
    y_pred = (y_score >= 0.5).astype(int)
    return metrics.compute_metric_bundle(
        y_true,
        y_pred,
        y_score=y_score,
        seed=seed,
    )


def _text_series(df: pd.DataFrame) -> pd.Series:
    """Return text column as non-null strings."""
    _require_columns(df, {"text"}, "frame")
    return df["text"].fillna("").astype(str)


def _labels(df: pd.DataFrame, *, require_hard: bool) -> np.ndarray:
    """Return binary labels, preferring hard labels over human labels."""
    if "hard_label" in df.columns:
        column = "hard_label"
    elif not require_hard and "human_label" in df.columns:
        column = "human_label"
    else:
        expected = "hard_label" if require_hard else "hard_label or human_label"
        raise ValueError(f"frame must contain {expected}.")

    labels = pd.to_numeric(df[column], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{column} values must be coded 0/1.")
    return labels.astype(int).to_numpy()


def _collect_embedding_frame(
    train_df: pd.DataFrame,
    eval_dfs: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Collect ordered unique EIN2/text rows needed for embedding."""
    frames = [train_df, *eval_dfs.values()]
    for i, frame in enumerate(frames):
        _require_columns(frame, {"EIN2", "text"}, f"embedding frame {i}")

    combined = pd.concat(
        [frame[["EIN2", "text"]].copy() for frame in frames],
        ignore_index=True,
    )
    combined["EIN2"] = combined["EIN2"].astype("string").str.strip()
    combined["text"] = combined["text"].fillna("").astype(str)
    if combined["EIN2"].isna().any() or (combined["EIN2"] == "").any():
        raise ValueError("EIN2 values must be non-empty for embedding cache alignment.")

    text_counts = combined.groupby("EIN2", sort=False)["text"].nunique()
    conflicts = text_counts[text_counts > 1]
    if not conflicts.empty:
        raise ValueError("EIN2 values map to conflicting texts in embedding inputs.")
    return combined.drop_duplicates(subset=["EIN2"], keep="first").reset_index(
        drop=True,
    )


def _load_or_compute_embeddings(
    embedding_frame: pd.DataFrame,
    registry: PathRegistry,
    model_name: str,
    device: str,
    batch_size: int,
) -> EmbeddingResult:
    """Load matching cached embeddings or recompute and save them."""
    model_slug = _model_slug(model_name)
    embedding_path = registry.embeddings_dir / f"{model_slug}.npy"
    index_path = registry.embeddings_dir / f"{model_slug}.ein2_index.csv"
    requested_ein2s = embedding_frame["EIN2"].astype(str).tolist()

    cached = _read_embedding_cache(embedding_path, index_path, requested_ein2s)
    if cached is not None:
        logger.info("Loaded cached embeddings from %s", embedding_path)
        return EmbeddingResult(
            embeddings=cached, device=_resolve_device(device), precision="fp32"
        )

    logger.info("Computing embeddings for %d EIN2s", len(requested_ein2s))
    result = compute_minilm_embeddings(
        embedding_frame["text"].astype(str).tolist(),
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    if result.embeddings.shape[0] != len(requested_ein2s):
        raise ValueError("Embedding row count does not match requested EIN2 index.")

    registry.embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, result.embeddings)
    pd.DataFrame({"EIN2": requested_ein2s}).to_csv(index_path, index=False)
    return result


def _read_embedding_cache(
    embedding_path: Any,
    index_path: Any,
    requested_ein2s: list[str],
) -> np.ndarray | None:
    """Return cached embeddings only when the ordered EIN2 index matches."""
    if not embedding_path.exists() or not index_path.exists():
        return None

    index = pd.read_csv(index_path, dtype={"EIN2": "string"})
    if "EIN2" not in index.columns:
        logger.warning("Ignoring embedding cache with no EIN2 index: %s", index_path)
        return None
    cached_ein2s = index["EIN2"].astype("string").str.strip().astype(str).tolist()
    if cached_ein2s != requested_ein2s:
        logger.info("Ignoring embedding cache because EIN2 order changed")
        return None

    embeddings = np.load(embedding_path, allow_pickle=False)
    if embeddings.shape[0] != len(requested_ein2s):
        logger.warning("Ignoring embedding cache with mismatched row count")
        return None
    return np.asarray(embeddings, dtype=np.float32)


def _embeddings_for_frame(
    frame: pd.DataFrame,
    embedding_lookup: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Gather embeddings aligned to a frame's EIN2 order."""
    _require_columns(frame, {"EIN2"}, "embedding frame")
    rows = []
    for ein2 in frame["EIN2"].astype("string").str.strip().astype(str):
        try:
            rows.append(embedding_lookup[ein2])
        except KeyError as exc:
            raise ValueError(f"No cached embedding for EIN2 {ein2!r}.") from exc
    return np.vstack(rows).astype(np.float32, copy=False)


def _result_row(
    *,
    model: str,
    n_train: int,
    seed: int,
    eval_metrics: Mapping[str, dict[str, Any]],
    wall_seconds: float,
    device: str,
    precision: str,
) -> dict[str, Any]:
    """Build a baseline result row matching the PR-2 schema."""
    row: dict[str, Any] = {
        "run_id": f"{model.replace(':', '-')}-seed-{seed}",
        "model": model,
        "targets": "hard",
        "arm": "default",
        "train_fraction": 1.0,
        "n_train": int(n_train),
        "seed": int(seed),
        "wall_seconds": float(wall_seconds),
        "precision": precision,
        "device": device,
        "git_sha": "unknown",
        "config_hash": "unknown",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    row.update(eval_metrics)
    return row


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` according to CUDA, MPS, then CPU capability checks."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _model_slug(model_name: str) -> str:
    """Return a filesystem-safe cache slug for a model identifier."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")


def _require_columns(df: pd.DataFrame, required: set[str], frame_name: str) -> None:
    """Raise if a frame lacks required columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {sorted(missing)}.")
