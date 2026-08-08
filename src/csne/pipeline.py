"""The find-and-explain pipeline: retrieve, then summarize.

This is the object the demo notebook and the HF Space both drive.
"""

from __future__ import annotations

from dataclasses import dataclass

from csne.config import ExperimentConfig
from csne.retrieval.encoder import CodeSearchEncoder
from csne.retrieval.index import EmbeddingIndex, SearchHit
from csne.summarization.base import Summarizer


@dataclass
class Explanation:
    """A retrieved function plus its generated explanation."""

    hit: SearchHit
    summary: str


class FindAndExplain:
    """Retrieves candidate functions for a query and explains each one."""

    def __init__(
        self,
        encoder: CodeSearchEncoder,
        index: EmbeddingIndex,
        summarizer: Summarizer,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> FindAndExplain:
        """Assemble the full pipeline from a config, loading a saved index if present."""
        raise NotImplementedError

    def find(self, query: str, k: int = 5) -> list[SearchHit]:
        """Retrieval only — no summarization cost."""
        raise NotImplementedError

    def explain(self, query: str, k: int = 5) -> list[Explanation]:
        """Retrieve top-k and summarize each result."""
        raise NotImplementedError
