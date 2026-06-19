"""Tests for Stage 03 OpenAI Batch API request and response handling."""

import json
from types import SimpleNamespace

import pandas as pd

from binary_classifier.annotate.annotators.openai_annotator import OpenAIAnnotator
from binary_classifier.annotate.run_annotation import (
    _run_openai_batch_group,
    run_annotation_matrix,
)
from binary_classifier.annotate.schema import BinaryLabel, LabelRecord, SourceType
from binary_classifier.config import BakeoffCandidate


class _LiveStubAnnotator:
    """No-network live annotator for non-OpenAI Stage 03 groups."""

    def __init__(self, spec: BakeoffCandidate, prompt_id: str) -> None:
        self.spec = spec
        self.prompt_id = prompt_id

    def annotate(self, text: str, ein2: str = "") -> LabelRecord:
        return LabelRecord(
            EIN2=ein2,
            source_id=f"{self.spec.id}__{self.prompt_id}",
            source_type=SourceType.LLM_PROMPT,
            model_id=self.spec.id,
            prompt_id=self.prompt_id,
            temperature=0.0,
            binary_label=BinaryLabel.NONRELIGIOUS,
        )


def test_openai_batch_request_uses_live_chat_body_shape() -> None:
    """Batch JSONL uses the same strict Chat Completions body as live calls."""
    annotator = OpenAIAnnotator(
        model_id="gpt-5-mini-2025-08-07",
        prompt_id="v1",
        prompt_text="system prompt",
        seed=123,
        api_key="test-key",
        reasoning_effort="minimal",
        guided_json=True,
    )

    request = annotator.build_batch_request("mission text", "custom-1")

    assert request["custom_id"] == "custom-1"
    assert request["method"] == "POST"
    assert request["url"] == "/v1/chat/completions"
    body = request["body"]
    assert body["model"] == "gpt-5-mini-2025-08-07"
    assert body["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "mission text"},
    ]
    assert body["seed"] == 123
    assert body["reasoning_effort"] == "minimal"
    assert "temperature" not in body
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True


def test_openai_batch_response_success_and_error_parse_to_label_records() -> None:
    """Batch output success rows parse normally; failed rows become abstains."""
    annotator = OpenAIAnnotator(
        model_id="gpt-4o-mini-2024-07-18",
        prompt_id="v1",
        prompt_text="system prompt",
        api_key="test-key",
    )
    raw_label = json.dumps(
        {
            "binary_label": "religious",
            "confidence": 0.91,
            "domains_present": ["faith_tradition"],
            "evidence_spans": ["church"],
            "boundary_notes": None,
            "reason": "mentions a church",
        }
    )
    success_line = json.dumps(
        {
            "custom_id": "gpt-4o-mini-2024-07-18__v1::00-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": raw_label}}],
                    "system_fingerprint": "fp-test",
                },
            },
        }
    )
    error_line = json.dumps(
        {
            "custom_id": "gpt-4o-mini-2024-07-18__v1::00-2",
            "response": {
                "status_code": 429,
                "body": {
                    "error": {
                        "message": "rate limit",
                        "type": "rate_limit_error",
                    }
                },
            },
        }
    )

    success = annotator.parse_batch_response_line(success_line, "00-1")
    error = annotator.parse_batch_response_line(error_line, "00-2")

    assert success.binary_label == BinaryLabel.RELIGIOUS
    assert success.label == 1.0
    assert success.system_fingerprint == "fp-test"
    assert success.raw_response == raw_label
    assert error.binary_label is None
    assert error.label is None
    assert "batch_status_429" in str(error.reason)
    assert "rate limit" in str(error.reason)


def test_stage03_batches_only_openai_groups_when_enabled(
    tiny_config,
    tmp_path,
    monkeypatch,
) -> None:
    """Stage 03 can mix OpenAI batch routing with live vLLM annotation."""
    tiny_config.annotation.openai_batch = True
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_path = prompts_dir / "v1.txt"
    prompt_path.write_text("prompt")
    batch_calls = []

    def factory(spec, prompt_id, prompt_text):
        return _LiveStubAnnotator(spec, prompt_id)

    def fake_batch_group(**kwargs):
        spec = kwargs["spec"]
        prompt_id = kwargs["prompt_id"]
        rows = kwargs["rows"]
        batch_calls.append((spec.id, prompt_id, len(rows)))
        return [
            LabelRecord(
                EIN2=ein2,
                source_id=f"{spec.id}__{prompt_id}",
                source_type=SourceType.LLM_PROMPT,
                model_id=spec.id,
                prompt_id=prompt_id,
                temperature=0.0,
                binary_label=BinaryLabel.RELIGIOUS,
            )
            for ein2, _text in rows
        ]

    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation._run_openai_batch_group",
        fake_batch_group,
    )

    store = run_annotation_matrix(
        df=pd.DataFrame(
            {
                "EIN2": ["00-1", "00-2"],
                "text": ["church mission", "food pantry"],
            }
        ),
        specs=[
            BakeoffCandidate(id="gpt-4o-mini-2024-07-18", provider="openai"),
            BakeoffCandidate(id="google/gemma-3-27b-it", provider="vllm"),
        ],
        prompt_paths=[prompt_path],
        store_path=tmp_path / "store.csv",
        annotator_factory=factory,
        cfg=tiny_config,
    )

    frame = store.to_frame()
    assert batch_calls == [("gpt-4o-mini-2024-07-18", "v1", 2)]
    assert len(frame) == 4
    assert set(frame["source_id"]) == {
        "gpt-4o-mini-2024-07-18__v1",
        "google/gemma-3-27b-it__v1",
    }


def test_stage03_openai_batch_can_be_disabled_for_smoke_runs(
    tiny_config,
    tmp_path,
    monkeypatch,
) -> None:
    """Limited/canary callers can force live OpenAI calls despite config default."""
    tiny_config.annotation.openai_batch = True
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_path = prompts_dir / "v1.txt"
    prompt_path.write_text("prompt")

    def fail_batch_group(**kwargs):
        raise AssertionError("batch path should not run")

    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation._run_openai_batch_group",
        fail_batch_group,
    )

    store = run_annotation_matrix(
        df=pd.DataFrame({"EIN2": ["00-1"], "text": ["church mission"]}),
        specs=[BakeoffCandidate(id="gpt-4o-mini-2024-07-18", provider="openai")],
        prompt_paths=[prompt_path],
        store_path=tmp_path / "store.csv",
        annotator_factory=lambda spec, prompt_id, prompt_text: _LiveStubAnnotator(
            spec, prompt_id
        ),
        cfg=tiny_config,
        use_openai_batch=False,
    )

    assert len(store.to_frame()) == 1


def test_openai_batch_reuses_complete_existing_output(tmp_path, monkeypatch) -> None:
    """A rerun after output download does not submit duplicate batch work."""
    prompt_id = "v1"
    spec = BakeoffCandidate(id="gpt-4o-mini-2024-07-18", provider="openai")
    source_id = f"{spec.id}__{prompt_id}"
    safe_source = "gpt-4o-mini-2024-07-18__v1"
    output_dir = tmp_path / "openai_batch"
    output_dir.mkdir()
    raw_label = json.dumps(
        {
            "binary_label": "religious",
            "confidence": 0.91,
            "domains_present": None,
            "evidence_spans": None,
            "boundary_notes": None,
            "reason": "mentions church",
        }
    )
    output_line = json.dumps(
        {
            "custom_id": f"{source_id}::00-1",
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": raw_label}}]},
            },
        }
    )
    (output_dir / f"{safe_source}.output.jsonl").write_text(output_line + "\n")

    def fail_submit(**kwargs):
        raise AssertionError("existing output should be reused")

    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation._submit_and_download_openai_batch",
        fail_submit,
    )

    records = _run_openai_batch_group(
        spec=spec,
        prompt_id=prompt_id,
        prompt_text="prompt",
        rows=[("00-1", "church mission")],
        store_path=tmp_path / "store.csv",
        annotator_factory=lambda spec, prompt_id, prompt_text: OpenAIAnnotator(
            model_id=spec.id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            api_key="test-key",
        ),
        poll_seconds=1,
        completion_window="24h",
    )

    assert len(records) == 1
    assert records[0].binary_label == BinaryLabel.RELIGIOUS


def test_openai_batch_error_file_rows_are_terminal(tmp_path, monkeypatch) -> None:
    """Existing error JSONL rows prevent duplicate batch submission on rerun."""
    prompt_id = "v1"
    spec = BakeoffCandidate(id="gpt-4o-mini-2024-07-18", provider="openai")
    source_id = f"{spec.id}__{prompt_id}"
    safe_source = "gpt-4o-mini-2024-07-18__v1"
    output_dir = tmp_path / "openai_batch"
    output_dir.mkdir()
    error_line = json.dumps(
        {
            "custom_id": f"{source_id}::00-1",
            "error": {
                "message": "invalid request",
                "type": "invalid_request_error",
            },
        }
    )
    (output_dir / f"{safe_source}.errors.jsonl").write_text(error_line + "\n")

    def fail_submit(**kwargs):
        raise AssertionError("existing error output should be reused")

    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation._submit_and_download_openai_batch",
        fail_submit,
    )

    records = _run_openai_batch_group(
        spec=spec,
        prompt_id=prompt_id,
        prompt_text="prompt",
        rows=[("00-1", "church mission")],
        store_path=tmp_path / "store.csv",
        annotator_factory=lambda spec, prompt_id, prompt_text: OpenAIAnnotator(
            model_id=spec.id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            api_key="test-key",
        ),
        poll_seconds=1,
        completion_window="24h",
    )

    assert len(records) == 1
    assert records[0].binary_label is None
    assert "invalid request" in str(records[0].reason)


def test_submit_openai_batch_uses_mock_client_no_network(tmp_path) -> None:
    """The OpenAI batch submit path talks only to the injected client in tests."""
    from binary_classifier.annotate.run_annotation import _submit_and_download_openai_batch

    requests_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "output.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    metadata_path = tmp_path / "batch.json"
    requests_path.write_text("{}\n")

    class _Files:
        def create(self, file, purpose):
            assert purpose == "batch"
            assert file.read() == b"{}\n"
            return SimpleNamespace(id="file-in")

        def content(self, file_id):
            assert file_id == "file-out"
            return SimpleNamespace(
                write_to_file=lambda path: path.write_text('{"custom_id":"c"}\n')
            )

    class _Batches:
        def create(self, input_file_id, endpoint, completion_window):
            assert input_file_id == "file-in"
            assert endpoint == "/v1/chat/completions"
            assert completion_window == "24h"
            return SimpleNamespace(
                id="batch-1",
                status="completed",
                output_file_id="file-out",
                error_file_id=None,
            )

    annotator = OpenAIAnnotator(
        model_id="gpt-4o-mini-2024-07-18",
        prompt_id="v1",
        prompt_text="prompt",
        api_key="test-key",
    )
    annotator.client = SimpleNamespace(files=_Files(), batches=_Batches())

    _submit_and_download_openai_batch(
        annotator=annotator,
        requests_path=requests_path,
        output_path=output_path,
        errors_path=errors_path,
        metadata_path=metadata_path,
        poll_seconds=1,
        completion_window="24h",
    )

    assert output_path.read_text() == '{"custom_id":"c"}\n'
    metadata = json.loads(metadata_path.read_text())
    assert metadata["batch_id"] == "batch-1"
    assert metadata["request_sha256"]


def test_submit_openai_batch_resumes_existing_metadata(tmp_path) -> None:
    """A persisted matching batch id is retrieved instead of resubmitted."""
    from binary_classifier.annotate.run_annotation import _submit_and_download_openai_batch

    requests_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "output.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    metadata_path = tmp_path / "batch.json"
    requests_path.write_text("{}\n")
    import hashlib

    metadata_path.write_text(
        json.dumps(
            {
                "batch_id": "batch-existing",
                "request_sha256": hashlib.sha256(b"{}\n").hexdigest(),
            }
        )
    )

    class _Files:
        def create(self, file, purpose):
            raise AssertionError("should not upload a duplicate batch input")

        def content(self, file_id):
            assert file_id == "file-out"
            return SimpleNamespace(
                write_to_file=lambda path: path.write_text('{"custom_id":"c"}\n')
            )

    class _Batches:
        def create(self, input_file_id, endpoint, completion_window):
            raise AssertionError("should not create a duplicate batch")

        def retrieve(self, batch_id):
            assert batch_id == "batch-existing"
            return SimpleNamespace(
                id="batch-existing",
                status="completed",
                output_file_id="file-out",
                error_file_id=None,
            )

    annotator = OpenAIAnnotator(
        model_id="gpt-4o-mini-2024-07-18",
        prompt_id="v1",
        prompt_text="prompt",
        api_key="test-key",
    )
    annotator.client = SimpleNamespace(files=_Files(), batches=_Batches())

    _submit_and_download_openai_batch(
        annotator=annotator,
        requests_path=requests_path,
        output_path=output_path,
        errors_path=errors_path,
        metadata_path=metadata_path,
        poll_seconds=1,
        completion_window="24h",
    )

    assert output_path.read_text() == '{"custom_id":"c"}\n'
