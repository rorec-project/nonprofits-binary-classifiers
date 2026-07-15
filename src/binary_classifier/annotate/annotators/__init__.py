"""Annotator sub-package.

Public exports:
    * ``Annotator`` - abstract base class.
    * ``OpenAIAnnotator`` - OpenAI API implementation.
    * ``VLLMAnnotator`` - local vLLM endpoint implementation.
"""

from binary_classifier.annotate.annotators.base import Annotator
from binary_classifier.annotate.annotators.openai_annotator import OpenAIAnnotator
from binary_classifier.annotate.annotators.vllm_annotator import VLLMAnnotator

__all__ = [
    "Annotator",
    "OpenAIAnnotator",
    "VLLMAnnotator",
]
