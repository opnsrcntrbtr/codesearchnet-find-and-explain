"""Claude-backed summarizer (prompting, no fine-tuning).

Requires the `anthropic` extra and an `ANTHROPIC_API_KEY`. Costs API tokens
per function, so evaluation runs should be capped via `EvalConfig`.
"""

from __future__ import annotations

from csne.config import SummarizationConfig
from csne.data.loader import CodeExample
from csne.summarization.base import Summarizer

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicSummarizer(Summarizer):
    """Generates function summaries through the Anthropic Messages API."""

    def __init__(self, config: SummarizationConfig) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    def summarize(self, example: CodeExample) -> str:
        raise NotImplementedError

    def summarize_batch(self, examples: list[CodeExample]) -> list[str]:
        """Concurrent single-message calls with retry on rate limits."""
        raise NotImplementedError
