"""Summarizer interface shared by both backends.

The API and the local-HF backends are kept behind one protocol so the report
can compare them on identical inputs, and so the demo can fall back to the
local model when no API key is present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from csne.config import SummarizationConfig
from csne.data.loader import CodeExample


class Summarizer(ABC):
    """Generates a short natural-language explanation of a function."""

    @abstractmethod
    def summarize(self, example: CodeExample) -> str:
        """Return a one-to-three sentence summary of what the function does."""

    @abstractmethod
    def summarize_batch(self, examples: list[CodeExample]) -> list[str]:
        """Summarize many functions. Backends batch or parallelize as they can."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded in results, e.g. `anthropic:claude-sonnet-5`."""


def build_summarizer(config: SummarizationConfig) -> Summarizer:
    """Construct the backend named by `config.backend`.

    Backend modules are imported lazily so the optional `anthropic` and `hf`
    extras are only required by the backend actually in use.
    """
    raise NotImplementedError


def load_prompt_template(path: str) -> str:
    """Read the summarization prompt template from disk.

    Kept as a file, not a string literal, so prompt changes show up as diffs
    and can be tied to a specific experiment run.
    """
    raise NotImplementedError
