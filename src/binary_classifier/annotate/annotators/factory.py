"""Annotator factory.

Routes a config-driven model spec to the correct provider-specific annotator.
This is the single construction point used by both the bake-off (stage 02) and
the full annotation run (stage 03), so provider routing and hyperparameter
wiring live in one place.
"""

from typing import TYPE_CHECKING

from binary_classifier.annotate.annotators.base import Annotator
from binary_classifier.annotate.annotators.openai_annotator import OpenAIAnnotator
from binary_classifier.annotate.annotators.vllm_annotator import VLLMAnnotator

if TYPE_CHECKING:
    from binary_classifier.config import BakeoffCandidate, BinaryClassifierConfig


def make_annotator(
    cfg: "BinaryClassifierConfig",
    spec: "BakeoffCandidate",
    prompt_id: str,
    prompt_text: str,
) -> Annotator:
    """Build the annotator for a single model spec and prompt.

    Routing is by ``spec.provider``: ``openai`` → :class:`OpenAIAnnotator`,
    ``vllm`` → :class:`VLLMAnnotator`. ``reasoning_effort`` is forwarded only
    when set on the spec, so providers/models that do not support it are never
    passed an unexpected value. ``guided_json`` is forwarded from the global
    annotation config so bake-off and production runs honor the same
    structured-decoding switch.

    Args:
        cfg: Validated configuration object (supplies temperature/seed/retries).
        spec: The candidate model spec (id, provider, optional reasoning_effort).
        prompt_id: Prompt version stem (e.g. ``v1``) — the real provenance key.
        prompt_text: Full prompt text sent to the model.

    Returns:
        A provider-specific :class:`Annotator` instance.

    Raises:
        ValueError: If ``spec.provider`` is not a known provider.

    """
    # Forward reasoning_effort only when set, so providers/models that do not
    # support it are never passed a value.
    extra: dict[str, str] = {}
    if spec.reasoning_effort is not None:
        extra["reasoning_effort"] = spec.reasoning_effort

    if spec.provider == "openai":
        # The guided-json switch is a global annotation knob, not a per-model
        # slate choice, so every provider construction path must receive it.
        return OpenAIAnnotator(
            model_id=spec.id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            temperature=cfg.annotation.temperature,
            seed=cfg.annotation.seed,
            max_retries=cfg.annotation.max_retries,
            guided_json=cfg.annotation.guided_json,
            **extra,
        )

    if spec.provider == "vllm":
        # The guided-json switch is a global annotation knob, not a per-model
        # slate choice, so every provider construction path must receive it.
        return VLLMAnnotator(
            model_id=spec.id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            temperature=cfg.annotation.temperature,
            seed=cfg.annotation.seed,
            max_retries=cfg.annotation.max_retries,
            guided_json=cfg.annotation.guided_json,
            **extra,
        )

    raise ValueError(f"Unknown provider: {spec.provider!r} (spec id={spec.id!r})")
