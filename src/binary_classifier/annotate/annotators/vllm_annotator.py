"""vLLM / local OpenAI-compatible endpoint annotator.

Calls a local vLLM server at ``http://127.0.0.1:8000/v1`` using the OpenAI
client library. When guided JSON is enabled, requests vLLM's native
``extra_body={"guided_json": ...}`` (xgrammar, ``disable_any_whitespace``)
structured output — chosen over OpenAI ``response_format=json_schema`` because
Gemma-3 runs away emitting whitespace under the latter; see ``annotate``.
"""

import json
import logging
import re
from typing import Any

from openai import OpenAI

from binary_classifier.annotate.annotators.base import Annotator
from binary_classifier.annotate.schema import (
    BinaryLabel,
    LabelRecord,
    SourceType,
    build_json_schema,
    conformance_error,
)

logger = logging.getLogger(__name__)

# Matches a leading ```json / ``` fence and the trailing ``` fence that some
# instruction-tuned models (e.g. Gemma-3) wrap around structured JSON output
# even under guided decoding.
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _strip_code_fence(raw: str) -> str:
    """Return ``raw`` with a surrounding markdown code fence removed, if present.

    Guided JSON constrains the *content* but not whether the model wraps it in a
    ```json fence; stripping the fence keeps parsing robust to either form.
    """
    match = _CODE_FENCE_RE.match(raw)
    return match.group("body") if match else raw


class VLLMAnnotator(Annotator):
    """Annotator backed by a local vLLM endpoint.

    Args:
        model_id: HuggingFace model identifier (e.g.
            ``Qwen/Qwen3-235B-A22B-Instruct-2507``). Must match the model
            loaded by the vLLM server.
        prompt_id: Prompt version tag.
        prompt_text: Full prompt text.
        temperature: Fixed at 0.0 for best-effort reproducible output.
        seed: Random seed.
        max_retries: Retries on connection errors.
        base_url: vLLM server URL (default ``http://127.0.0.1:8000/v1``).
        api_key: Dummy key for OpenAI-compatible clients (default ``EMPTY``).
        guided_json: Whether to ask vLLM for schema-constrained decoding.

    """

    def __init__(
        self,
        model_id: str,
        prompt_id: str,
        prompt_text: str,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 5,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        reasoning_effort: str | None = None,
        guided_json: bool = True,
    ) -> None:
        """Initialize the local endpoint client and decoding controls.

        ``guided_json`` mirrors the annotation config flag so experiments can
        explicitly compare constrained decoding against prompt-only JSON output
        without changing the parsing contract downstream.
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
            base_url=base_url,
            api_key=api_key,
        )

    # ── Core annotation method ───────────────────────────────────────────

    def annotate(self, text: str, ein2: str = "") -> LabelRecord:
        """Call the local vLLM endpoint and return a parsed ``LabelRecord``.

        The guided JSON flag only changes provider-side decoding. Parsing still
        validates the raw response so disabled structured decoding becomes an
        abstain/error record if the model ignores the JSON instruction.

        Args:
            text: Mission text to classify.
            ein2: IRS identifier for the record.

        Returns:
            A ``LabelRecord`` with ``raw_response`` preserved.

        """
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": self.prompt_text},
                        {"role": "user", "content": text},
                    ],
                    "temperature": self.temperature,
                    "seed": self.seed,
                }
                if self.guided_json:
                    # Use vLLM's native guided_json (xgrammar) rather than the
                    # OpenAI response_format=json_schema path: on Gemma-3 the
                    # latter lets the model run away emitting thousands of
                    # whitespace tokens until it hits the length limit, leaving
                    # truncated, unparseable JSON. The guided path with
                    # disable_any_whitespace constrains decoding so the model
                    # closes the object and stops cleanly in ~90 tokens.
                    kwargs["extra_body"] = {
                        "guided_json": build_json_schema(),
                        "guided_decoding_backend": "xgrammar",
                        "disable_any_whitespace": True,
                    }
                response = self.client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content
                if raw is None:
                    return self._error_record(ein2, "empty_content")
                parsed = self._parse_raw(raw, ein2)
                parsed.raw_response = raw
                return parsed
            except Exception as exc:
                if attempt >= self.max_retries:
                    return self._error_record(ein2, str(exc))
                self._retry_sleep(attempt)
        # Defensive fallback (should never be reached)
        return self._error_record(ein2, "unknown_failure")

    # ── Parsing helpers ──────────────────────────────────────────────────

    def _parse_raw(self, raw: str, ein2: str) -> LabelRecord:
        """Parse the raw JSON string into a ``LabelRecord``.

        If the JSON is malformed or fields are missing, returns an error-style
        record with ``binary_label=None`` so that downstream aggregation can
        treat it as an abstain.
        """
        try:
            data = json.loads(_strip_code_fence(raw))
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
