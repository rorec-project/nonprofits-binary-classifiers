"""Abstract annotator interface.

All provider-specific annotators (OpenAI, vLLM, etc.) implement ``Annotator``
and return a ``LabelRecord`` with a consistent schema. The interface is
intentionally thin: one ``annotate(text)`` method that is provider-agnostic.
"""

import time
from abc import ABC, abstractmethod

from binary_classifier.annotate.schema import LabelRecord, SourceType


class Annotator(ABC):
    """Provider-agnostic annotator interface.

    Args:
        model_id: Model identifier (e.g. ``gpt-4o-mini`` or
            ``Qwen/Qwen3-235B-A22B-Instruct-2507``).
        prompt_id: Prompt version (e.g. ``v1``, ``v2``, ``v3``).
        prompt_text: The full prompt text (including instructions and JSON
            schema) that will be sent to the model.
        temperature: Sampling temperature (0.0 for low-variance annotations).
        seed: Random seed for reproducibility.
        max_retries: Number of retries on transient failures.
        source_type: Origin of the label (defaults to ``llm_prompt``).
        reasoning_effort: Optional reasoning-effort knob for GPT-5-class
            models (e.g. ``minimal``). Stored on the instance; wiring it into
            the API request is handled by T2.A (provider-specific).
        guided_json: Whether provider calls should request schema-constrained
            JSON decoding. Disabling it leaves JSON compliance to the prompt.

    """

    def __init__(
        self,
        model_id: str,
        prompt_id: str,
        prompt_text: str,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 5,
        source_type: SourceType = SourceType.LLM_PROMPT,
        reasoning_effort: str | None = None,
        guided_json: bool = True,
    ) -> None:
        """Store provider-neutral annotation settings for downstream calls.

        The base class keeps runtime knobs in one place so OpenAI and vLLM
        annotators apply the same prompt, seed, retry, and structured-decoding
        choices during bake-off and production annotation.
        """
        self.model_id: str = model_id
        self.prompt_id: str = prompt_id
        self.prompt_text: str = prompt_text
        self.temperature: float = temperature
        self.seed: int = seed
        self.max_retries: int = max_retries
        self.source_type: SourceType = source_type
        self.reasoning_effort: str | None = reasoning_effort
        self.guided_json: bool = guided_json

    # ── Abstract method ──────────────────────────────────────────────────

    @abstractmethod
    def annotate(self, text: str, ein2: str = "") -> LabelRecord:
        """Annotate a single text record.

        Args:
            text: The mission or activity text to classify.
            ein2: IRS identifier for the record (used as join key).

        Returns:
            A validated ``LabelRecord`` with all fields populated.

        """
        ...

    # ── Shared helpers ───────────────────────────────────────────────────

    def _build_source_id(self) -> str:
        """Build a canonical source_id from model_id + prompt_id."""
        return f"{self.model_id}__{self.prompt_id}"

    def _retry_sleep(self, attempt: int) -> None:
        """Exponential backoff between retries."""
        time.sleep(min(2**attempt, 60))
