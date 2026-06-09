"""Tests for T2.A: annotation hardening (schema + annotators)."""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from binary_classifier.annotate.annotators.openai_annotator import OpenAIAnnotator
from binary_classifier.annotate.annotators.vllm_annotator import VLLMAnnotator
from binary_classifier.annotate.schema import (
    AnnotationStore,
    BinaryLabel,
    LabelRecord,
    SourceType,
    build_json_schema,
)


# ── Schema tests ─────────────────────────────────────────────────────────────


def test_build_json_schema_is_strict_compatible() -> None:
    """The schema has all fields required, nullable types use anyOf, and
    additionalProperties is false."""
    schema = build_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    required = set(schema["required"])
    assert required == {
        "binary_label",
        "confidence",
        "domains_present",
        "evidence_spans",
        "boundary_notes",
        "reason",
    }

    # Nullable fields must use anyOf with null
    for field in ("domains_present", "evidence_spans", "boundary_notes", "reason"):
        prop = schema["properties"][field]
        assert "anyOf" in prop
        assert any(sub.get("type") == "null" for sub in prop["anyOf"])

    # Non-nullable fields should not have anyOf
    assert "anyOf" not in schema["properties"]["binary_label"]
    assert "anyOf" not in schema["properties"]["confidence"]


# ── OpenAI annotator tests ───────────────────────────────────────────────────


def _make_mock_openai_response(content: dict, fingerprint: str = "fp_abc123"):
    """Build a mock OpenAI ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(content)
    response.system_fingerprint = fingerprint
    return response


def test_openai_annotator_emits_schema_valid_json_and_fingerprint() -> None:
    """Mocked OpenAI path returns a parsed LabelRecord with system_fingerprint."""
    mock_client = MagicMock()
    mock_response = _make_mock_openai_response(
        {
            "binary_label": "religious",
            "confidence": 0.95,
            "domains_present": None,
            "evidence_spans": None,
            "boundary_notes": None,
            "reason": "Mentions church and worship.",
        },
        fingerprint="fp_test_42",
    )
    mock_client.chat.completions.create.return_value = mock_response

    annotator = OpenAIAnnotator(
        model_id="gpt-4o-mini",
        prompt_id="v1",
        prompt_text="Classify the text.",
        api_key="test-key",
    )
    annotator.client = mock_client

    record = annotator.annotate("We are a church.", ein2="00-1")

    assert record.binary_label == BinaryLabel.RELIGIOUS
    assert record.confidence == 0.95
    assert record.system_fingerprint == "fp_test_42"
    assert record.reason == "Mentions church and worship."

    # Verify the API was called with json_schema strict mode
    call_args = mock_client.chat.completions.create.call_args.kwargs
    assert call_args["response_format"]["type"] == "json_schema"
    assert call_args["response_format"]["json_schema"]["strict"] is True
    assert call_args["response_format"]["json_schema"]["schema"] == build_json_schema()


def test_openai_annotator_forwards_reasoning_effort() -> None:
    """When reasoning_effort is set, it is passed to the API call."""
    mock_client = MagicMock()
    mock_response = _make_mock_openai_response(
        {
            "binary_label": "nonreligious",
            "confidence": 0.8,
            "domains_present": None,
            "evidence_spans": None,
            "boundary_notes": None,
            "reason": "No religious language.",
        }
    )
    mock_client.chat.completions.create.return_value = mock_response

    annotator = OpenAIAnnotator(
        model_id="gpt-5-mini",
        prompt_id="v1",
        prompt_text="Classify.",
        api_key="test",
        reasoning_effort="minimal",
    )
    annotator.client = mock_client

    annotator.annotate("We help animals.", ein2="00-2")

    call_args = mock_client.chat.completions.create.call_args.kwargs
    assert call_args.get("reasoning_effort") == "minimal"


def test_openai_annotator_no_reasoning_effort_when_none() -> None:
    """When reasoning_effort is None, it is not included in the API call."""
    mock_client = MagicMock()
    mock_response = _make_mock_openai_response(
        {
            "binary_label": "nonreligious",
            "confidence": 0.8,
            "domains_present": None,
            "evidence_spans": None,
            "boundary_notes": None,
            "reason": "No religious language.",
        }
    )
    mock_client.chat.completions.create.return_value = mock_response

    annotator = OpenAIAnnotator(
        model_id="gpt-4o-mini",
        prompt_id="v1",
        prompt_text="Classify.",
        api_key="test",
    )
    annotator.client = mock_client

    annotator.annotate("We help animals.", ein2="00-2")

    call_args = mock_client.chat.completions.create.call_args.kwargs
    assert call_args.get("reasoning_effort") is None


# ── vLLM annotator tests ─────────────────────────────────────────────────────


def test_vllm_annotator_uses_shared_schema_no_inline_dict() -> None:
    """vLLM annotator passes the schema from schema.py to guided_json and has
    no inline properties dict."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {
            "binary_label": "religious",
            "confidence": 0.9,
            "domains_present": None,
            "evidence_spans": None,
            "boundary_notes": None,
            "reason": "test",
        }
    )
    mock_client.chat.completions.create.return_value = mock_response

    annotator = VLLMAnnotator(
        model_id="gemma-3-27b-it",
        prompt_id="v1",
        prompt_text="Classify.",
    )
    annotator.client = mock_client

    annotator.annotate("test text", ein2="00-3")

    call_args = mock_client.chat.completions.create.call_args.kwargs
    assert "extra_body" in call_args
    guided = call_args["extra_body"]["guided_json"]
    assert guided == build_json_schema()


# ── AnnotationStore tests ────────────────────────────────────────────────────


def _make_record(ein2: str, source_id: str, fingerprint: str | None = None) -> LabelRecord:
    return LabelRecord(
        EIN2=ein2,
        source_id=source_id,
        source_type=SourceType.LLM_PROMPT,
        model_id="m1",
        prompt_id="v1",
        temperature=0.0,
        system_fingerprint=fingerprint,
    )


def test_store_append_only_no_full_rewrite(tmp_path) -> None:
    """Writing records appends to the CSV file without rewriting the entire file."""
    path = tmp_path / "store.csv"
    store = AnnotationStore(path)

    # Append 2 records
    store.append(_make_record("00-1", "m1__v1", "fp1"))
    store.append(_make_record("00-2", "m1__v1", "fp2"))

    # Verify file has header + 2 data rows
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3

    # Append 2 more records
    store.append(_make_record("00-3", "m1__v1", "fp3"))
    store.append(_make_record("00-4", "m1__v1", "fp4"))

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 5

    # All 4 records present
    df = store.to_frame()
    assert len(df) == 4
    assert set(df["system_fingerprint"]) == {"fp1", "fp2", "fp3", "fp4"}


def test_store_append_many_also_appends(tmp_path) -> None:
    """append_many writes directly without loading the full store."""
    path = tmp_path / "store.csv"
    store = AnnotationStore(path)

    batch1 = [_make_record(f"00-{i}", "m1__v1") for i in range(50)]
    store.append_many(batch1)

    batch2 = [_make_record(f"01-{i}", "m1__v1") for i in range(50)]
    store.append_many(batch2)

    df = store.to_frame()
    assert len(df) == 100


def test_system_fingerprint_persisted_in_store(tmp_path) -> None:
    """system_fingerprint column survives a write + read round-trip."""
    path = tmp_path / "store.csv"
    store = AnnotationStore(path)
    store.append(_make_record("00-1", "m1__v1", "fp_xyz"))

    df = pd.read_csv(path)
    assert "system_fingerprint" in df.columns
    assert df.loc[0, "system_fingerprint"] == "fp_xyz"

    # Rehydrate from flat dict
    record = LabelRecord.from_flat_dict(df.to_dict("records")[0])
    assert record.system_fingerprint == "fp_xyz"


def test_already_done_uses_cached_set(tmp_path) -> None:
    """already_done should use an internal set for O(1) lookups."""
    path = tmp_path / "store.csv"
    store = AnnotationStore(path)

    records = [_make_record(f"00-{i:03d}", "m1__v1") for i in range(100)]
    store.append_many(records)

    done_set = store._build_done_set()
    assert isinstance(done_set, set)
    assert len(done_set) == 100
    assert ("00-050", "m1__v1") in done_set
    assert store.already_done("00-050", "m1__v1")
    assert not store.already_done("99-999", "m1__v1")


def test_done_pairs_reads_only_two_columns(tmp_path) -> None:
    """done_pairs should not require loading the full dataframe."""
    path = tmp_path / "store.csv"
    store = AnnotationStore(path)

    records = [_make_record(f"existing-{i}", "m1__v1") for i in range(100)]
    store.append_many(records)

    pairs = store.done_pairs()
    assert len(pairs) == 100
    assert ("existing-50", "m1__v1") in pairs

    # _df should remain None if it was never loaded
    assert store._df is None


def test_resume_with_existing_and_new_records(tmp_path) -> None:
    """Simulate a resume scenario: 100 existing records, 100 new ones.
    Only the new ones should remain in the work list."""
    path = tmp_path / "store.csv"
    store = AnnotationStore(path)

    existing = [_make_record(f"E-{i}", "m1__v1") for i in range(100)]
    store.append_many(existing)

    # Simulate building a work matrix with 100 existing + 100 new
    work_items = [(f"E-{i}", "text", "m1__v1") for i in range(100)]
    work_items += [(f"N-{i}", "text", "m1__v1") for i in range(100)]

    existing_pairs = store.done_pairs()
    remaining = [w for w in work_items if (w[0], w[2]) not in existing_pairs]

    assert len(remaining) == 100
    assert all(w[0].startswith("N-") for w in remaining)


def test_store_backward_compatible_with_old_csv(tmp_path) -> None:
    """A CSV written before system_fingerprint existed loads without error."""
    path = tmp_path / "store.csv"
    # Write an old-style CSV manually (no system_fingerprint column)
    old_df = pd.DataFrame(
        [
            {
                "EIN2": "00-1",
                "source_id": "m1__v1",
                "source_type": "llm_prompt",
                "label": 1.0,
                "confidence": 0.9,
                "model_id": "m1",
                "prompt_id": "v1",
                "temperature": 0.0,
                "seed": 42,
                "run_timestamp": "2024-01-01T00:00:00+00:00",
                "raw_response": "{}",
                "reason": "test",
                "domains_present": None,
                "evidence_spans": None,
                "boundary_notes": None,
                "binary_label": "religious",
            }
        ]
    )
    old_df.to_csv(path, index=False)

    store = AnnotationStore(path)
    df = store.to_frame()
    assert "system_fingerprint" in df.columns
    assert pd.isna(df.loc[0, "system_fingerprint"])

    record = store.records_for_ein2("00-1")[0]
    assert record.system_fingerprint is None


def test_label_record_round_trip_with_none_fields() -> None:
    """from_flat_dict handles NaN/None fields gracefully."""
    record = LabelRecord(
        EIN2="00-1",
        source_id="m1__v1",
        source_type=SourceType.LLM_PROMPT,
        model_id="m1",
        prompt_id="v1",
        temperature=0.0,
        system_fingerprint=None,
        reason=None,
        domains_present=None,
        evidence_spans=None,
        boundary_notes=None,
    )
    flat = record.to_flat_dict()
    restored = LabelRecord.from_flat_dict(flat)
    assert restored.system_fingerprint is None
    assert restored.reason is None
    assert restored.domains_present is None
