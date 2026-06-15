"""Tests for baseline model helpers."""

import numpy as np
import pandas as pd

from binary_classifier.train import baselines


def test_tfidf_logreg_scores_synthetic_rows() -> None:
    """TF-IDF baseline learns obvious lexical signal on synthetic data."""
    frame = _synthetic_text_frame(60)
    train_df = frame.iloc[:42].reset_index(drop=True)
    dev_df = frame.iloc[42:51].reset_index(drop=True)
    validation_df = frame.iloc[51:].drop(columns=["hard_label"]).rename(
        columns={"human_label_source": "human_label"},
    )

    row = baselines.tfidf_logreg(
        train_df,
        {"dev": dev_df, "validation": validation_df},
        seed=123,
    )

    assert row["model"] == "baseline:tfidf_logreg"
    assert row["targets"] == "hard"
    assert row["arm"] == "default"
    assert row["n_train"] == len(train_df)
    assert row["device"] == "cpu"
    assert row["precision"] == "fp32"
    assert row["dev"]["pr_auc"] > dev_df["hard_label"].mean()
    assert row["validation"]["pr_auc"] > validation_df["human_label"].mean()


def test_minilm_logreg_uses_seeded_embedding_cache(monkeypatch, tiny_registry) -> None:
    """MiniLM baseline avoids downloads in tests and reuses aligned cache hits."""
    frame = _synthetic_text_frame(24)
    train_df = frame.iloc[:18].reset_index(drop=True)
    eval_df = train_df.copy()
    calls = []

    def fake_embeddings(
        texts,
        model_name="fake/minilm",
        device="auto",
        batch_size=32,
    ):
        del model_name, device, batch_size
        calls.append(list(texts))
        rng = np.random.default_rng(99)
        embeddings = rng.normal(size=(len(texts), 12)).astype(np.float32)
        return baselines.EmbeddingResult(
            embeddings=embeddings,
            device="cpu",
            precision="fp32",
        )

    monkeypatch.setattr(baselines, "compute_minilm_embeddings", fake_embeddings)

    row = baselines.minilm_logreg(
        train_df,
        {"dev": eval_df, "validation": eval_df},
        seed=99,
        registry=tiny_registry,
        model_name="fake/minilm",
    )
    cached_row = baselines.minilm_logreg(
        train_df,
        {"dev": eval_df, "validation": eval_df},
        seed=99,
        registry=tiny_registry,
        model_name="fake/minilm",
    )

    assert row["model"] == "baseline:minilm_logreg"
    assert row["dev"]["pr_auc"] is not None
    assert cached_row["validation"]["roc_auc"] is not None
    assert len(calls) == 1
    assert (tiny_registry.embeddings_dir / "fake__minilm.npy").exists()
    assert (tiny_registry.embeddings_dir / "fake__minilm.ein2_index.csv").exists()


def _synthetic_text_frame(n_rows: int) -> pd.DataFrame:
    """Build deterministic rows with separable religious/non-religious tokens."""
    rows = []
    for i in range(n_rows):
        label = i % 2
        if label == 1:
            text = f"church worship prayer ministry congregation service {i}"
        else:
            text = f"animal shelter soccer arts food bank community {i}"
        rows.append(
            {
                "EIN2": f"{i:09d}",
                "text": text,
                "hard_label": label,
                "human_label_source": label,
            },
        )
    return pd.DataFrame(rows)
