"""Regression tests for tokenizer special-token wrapping."""

import pytest
from transformers import AutoTokenizer


pytestmark = pytest.mark.network


@pytest.mark.parametrize(
    "model_id",
    ["microsoft/deberta-v3-base", "answerdotai/ModernBERT-base"],
)
def test_tokenizer_adds_cls_and_sep_tokens(model_id: str) -> None:
    tok = AutoTokenizer.from_pretrained(model_id)

    input_ids = tok("hello world")["input_ids"]

    assert tok.cls_token_id is not None, f"{model_id} has no CLS token id"
    assert tok.sep_token_id is not None, f"{model_id} has no SEP token id"
    assert input_ids[0] == tok.cls_token_id, (
        f"{model_id} did not prepend CLS token: "
        f"got {input_ids[0]}, expected {tok.cls_token_id}"
    )
    assert input_ids[-1] == tok.sep_token_id, (
        f"{model_id} did not append SEP token: "
        f"got {input_ids[-1]}, expected {tok.sep_token_id}"
    )
