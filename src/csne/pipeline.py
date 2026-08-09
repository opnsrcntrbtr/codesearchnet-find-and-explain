"""The find-and-explain pipeline: retrieve, then summarize.

This is the object the demo notebook and the HF Space both drive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from csne.config import ExperimentConfig
from csne.data.loader import load_prepared
from csne.retrieval.encoder import CodeSearchEncoder
from csne.retrieval.index import EmbeddingIndex, SearchHit
from csne.summarization.base import Summarizer, build_summarizer


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
        self.encoder = encoder
        self.index = index
        self.summarizer = summarizer

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> FindAndExplain:
        """Assemble the full pipeline from a config, loading a saved index if present.

        Loading the saved index (rather than re-encoding the corpus) is what
        makes `search`/`explain` fast enough for interactive use in a demo —
        re-encoding a large corpus on every process start would not be.
        """
        encoder = CodeSearchEncoder.from_config(config.retrieval)
        summarizer = build_summarizer(config.summarization)

        index_path = Path(config.retrieval.output_dir) / "index"
        if index_path.exists():
            index = EmbeddingIndex.load(str(index_path))
        else:
            examples = load_prepared("test", config.data)
            index = EmbeddingIndex.build(examples, encoder)

        return cls(encoder, index, summarizer)

    def find(self, query: str, k: int = 5) -> list[SearchHit]:
        """Retrieval only — no summarization cost."""
        return self.index.search(query, self.encoder, k=k)

    def explain(self, query: str, k: int = 5) -> list[Explanation]:
        """Retrieve top-k and summarize each result.

        Summarizes in one batch call rather than per-hit, so an API backend
        gets the concurrency in `summarize_batch` instead of k sequential
        round trips.
        """
        hits = self.find(query, k=k)
        summaries = self.summarizer.summarize_batch([hit.example for hit in hits])
        return [Explanation(hit=hit, summary=summary) for hit, summary in zip(hits, summaries)]
