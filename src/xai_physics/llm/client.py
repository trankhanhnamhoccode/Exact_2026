from __future__ import annotations

from typing import Protocol


class SchemaLLMClient(Protocol):
    """
    Minimal interface for schema-extraction LLM clients.

    Implementations can be:
    - fake test client
    - OpenAI client
    - local model client
    - Kaggle/local transformers client
    """

    def generate(self, prompt: str) -> str:
        ...
