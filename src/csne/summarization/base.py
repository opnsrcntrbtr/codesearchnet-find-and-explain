"""Summarizer interface shared by both backends.

The API and the local-HF backends are kept behind one protocol so the report
can compare them on identical inputs, and so the demo can fall back to the
local model when no API key is present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from csne.config import SummarizationConfig
from csne.data.loader import CodeExample

_BACKENDS = {"anthropic", "hf"}


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
    if config.backend not in _BACKENDS:
        raise ValueError(
            f"Unknown summarization backend {config.backend!r}. Valid: {sorted(_BACKENDS)}"
        )

    if config.backend == "anthropic":
        from csne.summarization.anthropic_backend import AnthropicSummarizer

        return AnthropicSummarizer(config)

    from csne.summarization.hf_backend import HFSummarizer

    return HFSummarizer(config)


def load_prompt_template(path: str) -> str:
    """Read the summarization prompt template from disk.

    Kept as a file, not a string literal, so prompt changes show up as diffs
    and can be tied to a specific experiment run.
    """
    text = Path(path).read_text()
    if not text.strip():
        raise ValueError(f"{path}: prompt template is empty")
    return text


def render_prompt(template: str, example: CodeExample) -> str:
    """Fill the prompt template's placeholders for one function.

    `str.format` only parses the template string for `{field}` markers — the
    substituted code is inserted verbatim, not rescanned — so code containing
    `{`/`}` (dict literals, f-strings) is safe here.
    """
    return template.format(
        func_name=example.func_name,
        repo=example.repo,
        path=example.path,
        code=example.code,
    )
