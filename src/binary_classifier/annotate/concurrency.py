"""Provider-level concurrency controls for annotation calls.

Stages 02 and 03 both run one worker per model x prompt arm. The semaphores in
this module cap the number of simultaneous provider API calls across those arms
without changing resume keys, checkpointing, or model/prompt grouping.
"""

import threading

from binary_classifier.annotate.annotators.base import Annotator
from binary_classifier.annotate.schema import LabelRecord
from binary_classifier.config import BinaryClassifierConfig

ProviderLimiters = dict[str, threading.Semaphore]


def build_provider_limiters(cfg: BinaryClassifierConfig) -> ProviderLimiters:
    """Build provider semaphores from annotation config."""
    return {
        "openai": threading.Semaphore(cfg.annotation.openai_max_concurrency),
        "vllm": threading.Semaphore(cfg.annotation.vllm_max_concurrency),
    }


def annotate_with_provider_limit(
    annotator: Annotator,
    provider: str,
    text: str,
    ein2: str,
    limiters: ProviderLimiters | None,
) -> LabelRecord:
    """Run one annotation call under the provider semaphore when configured."""
    limiter = limiters.get(provider) if limiters is not None else None
    if limiter is None:
        return annotator.annotate(text, ein2=ein2)
    with limiter:
        return annotator.annotate(text, ein2=ein2)
