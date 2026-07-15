"""OpenAI API annotator.

Uses the ``openai`` Python client with optional structured JSON output (guided
JSON) to produce schema-valid ``LabelRecord`` objects. Temperature is fixed at 0
for models that support custom sampling, and omitted for reasoning-effort models.
"""

import json
import logging
import os
from typing import Any, Literal, cast

from openai import BadRequestError, OpenAI

from binary_classifier.annotate.annotators.base import Annotator
from binary_classifier.annotate.schema import (
    BinaryLabel,
    LabelRecord,
    SourceType,
    build_json_schema,
    conformance_error,
)

logger = logging.getLogger(__name__)


class OpenAIAnnotator(Annotator):
    """Annotator backed by the OpenAI API.

    Args:
        model_id: OpenAI model snapshot (e.g. ``gpt-4o-mini``).
        prompt_id: Prompt version tag.
        prompt_text: Full prompt text.
        temperature: Fixed at 0.0 for best-effort reproducible output when the
            selected model supports custom sampling.
        seed: Random seed passed to the API.
        max_retries: Retries on rate-limit or transient errors.
        api_key: Optional explicit API key (falls back to ``OPENAI_API_KEY``).
        reasoning_effort: Optional reasoning-effort knob for GPT-5-class models.
        guided_json: Whether to ask OpenAI for strict schema-constrained output.

    """

    def __init__(
        self,
        model_id: str,
        prompt_id: str,
        prompt_text: str,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 5,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        guided_json: bool = True,
    ) -> None:
        """Initialize the API client and annotation decoding controls.

        ``guided_json`` mirrors the annotation config flag so the same parser can
        evaluate strict schema-mode output and prompt-only JSON output under a
        single downstream record schema.
        """
        super().__init__(
            model_id=model_id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            temperature=temperature,
            seed=seed,
            max_retries=max_retries,
            source_type=SourceType.LLM_PROMPT,
            reasoning_effort=reasoning_effort,
            guided_json=guided_json,
        )
        self.client: OpenAI = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    # ── Core annotation method ───────────────────────────────────────────

    def build_request_body(self, text: str) -> dict[str, Any]:
        """Return the Chat Completions body used by live and batch calls."""
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.prompt_text},
                {"role": "user", "content": text},
            ],
            "seed": self.seed,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = cast(
                Literal["none", "minimal", "low", "medium", "high", "xhigh"],
                self.reasoning_effort,
            )
        else:
            kwargs["temperature"] = self.temperature
        # cfg.annotation.guided_json controls OpenAI strict schema mode; false
        # leaves only prompt-level JSON guidance.
        if self.guided_json:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "label_record",
                    "schema": build_json_schema(),
                    "strict": True,
                },
            }
        return kwargs

    def build_batch_request(self, text: str, custom_id: str) -> dict[str, Any]:
        """Return one OpenAI Batch API JSONL request object."""
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": self.build_request_body(text),
        }

    def annotate(self, text: str, ein2: str = "") -> LabelRecord:
        """Call the OpenAI API and return a parsed ``LabelRecord``.

        The guided JSON flag only changes provider-side structured output. Raw
        parsing remains unchanged so malformed prompt-only responses become
        downstream abstain/error records instead of silent labels.

        Args:
            text: Mission text to classify.
            ein2: IRS identifier for the record.

        Returns:
            A ``LabelRecord`` with ``raw_response`` preserved.

        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    **self.build_request_body(text),
                )
                return self.parse_chat_completion_body(response.model_dump(), ein2)
            except BadRequestError as exc:
                return self._error_record(ein2, str(exc))
            except Exception as exc:
                if attempt >= self.max_retries:
                    return self._error_record(ein2, str(exc))
                self._retry_sleep(attempt)
        # Defensive fallback (should never be reached)
        return self._error_record(ein2, "unknown_failure")

    # ── Parsing helpers ──────────────────────────────────────────────────

    def parse_chat_completion_body(
        self,
        body: dict[str, Any],
        ein2: str,
    ) -> LabelRecord:
        """Parse an OpenAI Chat Completions response body into a record."""
        choices = body.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        raw = message.get("content")
        if raw is None:
            return self._error_record(ein2, "empty_content")
        parsed = self._parse_raw(raw, ein2)
        parsed.raw_response = raw
        parsed.system_fingerprint = body.get("system_fingerprint")
        return parsed

    def parse_batch_response_line(self, line: str, ein2: str) -> LabelRecord:
        """Parse one OpenAI Batch API output JSONL line into a record."""
        item = json.loads(line)
        if item.get("error"):
            return self._error_record(ein2, _format_batch_error(item["error"]))

        response = item.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body") or {}
        if status_code != 200:
            message = _format_batch_error(body.get("error") or body)
            return self._error_record(ein2, f"batch_status_{status_code}: {message}")
        return self.parse_chat_completion_body(body, ein2)

    def _parse_raw(self, raw: str, ein2: str) -> LabelRecord:
        """Parse the raw JSON string into a ``LabelRecord``.

        If the JSON is malformed or fields are missing, returns an error-style
        record with ``binary_label=None`` so that downstream aggregation can
        treat it as an abstain.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._error_record(ein2, "JSONDecodeError")

        err = conformance_error(data)

        if err is not None:
            logger.warning(
                "Non-conformant LLM response from %s for EIN2=%s: %s",
                self._build_source_id(),
                ein2,
                err,
            )

        binary_label_str = data.get("binary_label")
        try:
            binary_label = BinaryLabel(binary_label_str) if binary_label_str else None
        except ValueError:
            binary_label = None

        return LabelRecord(
            EIN2=ein2,
            source_id=self._build_source_id(),
            source_type=self.source_type,
            model_id=self.model_id,
            prompt_id=self.prompt_id,
            temperature=self.temperature,
            seed=self.seed,
            binary_label=binary_label,
            confidence=data.get("confidence"),
            reason=data.get("reason"),
            domains_present=data.get("domains_present"),
            evidence_spans=data.get("evidence_spans"),
            boundary_notes=data.get("boundary_notes"),
            raw_response=raw,
            error=err,
        )

    def _error_record(self, ein2: str, error: str) -> LabelRecord:
        """Return a record representing a failed annotation."""
        return LabelRecord(
            EIN2=ein2,
            source_id=self._build_source_id(),
            source_type=self.source_type,
            model_id=self.model_id,
            prompt_id=self.prompt_id,
            temperature=self.temperature,
            seed=self.seed,
            binary_label=None,
            confidence=None,
            reason=None,
            raw_response=None,
            error=error,
        )


def _format_batch_error(error: Any) -> str:
    """Return a compact error message from OpenAI batch response payloads."""
    if isinstance(error, dict):
        message = error.get("message") or error.get("error") or json.dumps(error)
        code = error.get("code") or error.get("type")
        return f"{code}: {message}" if code else str(message)
    return str(error)
