"""Local Hugging Face summarizer (seq2seq, fine-tunable).

Requires the `hf` extra. Free and fully reproducible — no API key, no network
at inference — which makes it the backend of record for the report's
reproducibility claims.
"""

from __future__ import annotations

from csne.config import DataConfig, SummarizationConfig
from csne.data.loader import CodeExample
from csne.summarization.base import Summarizer

DEFAULT_MODEL = "Salesforce/codet5p-220m"


class HFSummarizer(Summarizer):
    """Generates function summaries with a local seq2seq model."""

    def __init__(self, config: SummarizationConfig) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    def summarize(self, example: CodeExample) -> str:
        raise NotImplementedError

    def summarize_batch(self, examples: list[CodeExample]) -> list[str]:
        """True batched generation on GPU."""
        raise NotImplementedError


def finetune_summarizer(
    summarization_config: SummarizationConfig,
    data_config: DataConfig,
    run_name: str | None = None,
) -> HFSummarizer:
    """Fine-tune the local model on (code, cleaned docstring) pairs.

    Optional: the zero-shot backend is the baseline, and this is the
    fine-tuned comparison point in the report's summarization experiments.
    """
    raise NotImplementedError
