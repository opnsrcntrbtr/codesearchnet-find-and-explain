"""In-memory vector index over the encoded corpus.

Exact search over normalized vectors: the CodeSearchNet eval corpus is small
enough that a brute-force matmul beats the complexity of an ANN index, and it
avoids approximation error confounding the retrieval metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from csne.data.loader import CodeExample
from csne.retrieval.encoder import CodeSearchEncoder


@dataclass(frozen=True)
class SearchHit:
    """One retrieved function and its similarity to the query."""

    example: CodeExample
    score: float
    rank: int


class EmbeddingIndex:
    """Holds corpus embeddings and answers top-k similarity queries."""

    def __init__(self, examples: list[CodeExample], embeddings: np.ndarray) -> None:
        if len(examples) != len(embeddings):
            raise ValueError(
                f"corpus/embedding mismatch: {len(examples)} examples, "
                f"{len(embeddings)} vectors"
            )
        self.examples = examples
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def build(
        cls,
        examples: list[CodeExample],
        encoder: CodeSearchEncoder,
        batch_size: int = 64,
    ) -> EmbeddingIndex:
        """Encode a corpus and wrap it in an index."""
        return cls(examples, encoder.encode_code(examples, batch_size=batch_size))

    def _rank(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) of the top-k per query row.

        `argpartition` finds the top k in O(n) and only the k survivors get
        sorted — full argsort over the whole corpus per query is the obvious
        bottleneck at eval time.
        """
        scores = query_vectors @ self.embeddings.T
        k = min(k, self.embeddings.shape[0])
        if k == 0:
            empty = np.zeros((len(query_vectors), 0), dtype=int)
            return empty, empty.astype(np.float32)

        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        return np.take_along_axis(top, order, axis=1), np.take_along_axis(
            top_scores, order, axis=1
        )

    def _to_hits(self, indices: np.ndarray, scores: np.ndarray) -> list[SearchHit]:
        return [
            SearchHit(example=self.examples[int(idx)], score=float(score), rank=rank)
            for rank, (idx, score) in enumerate(zip(indices, scores), start=1)
        ]

    def search(self, query: str, encoder: CodeSearchEncoder, k: int = 10) -> list[SearchHit]:
        """Return the top-k most similar functions for one query."""
        return self.search_batch([query], encoder, k=k)[0]

    def search_batch(
        self, queries: list[str], encoder: CodeSearchEncoder, k: int = 10
    ) -> list[list[SearchHit]]:
        """Batched `search`, used by evaluation over the full query set."""
        if not queries:
            return []
        vectors = encoder.encode_queries(queries)
        indices, scores = self._rank(vectors, k)
        return [self._to_hits(indices[i], scores[i]) for i in range(len(queries))]

    def gold_ranks(self, queries: list[str], gold_ids: list[str], encoder: CodeSearchEncoder) -> list[int]:
        """1-indexed rank of each gold document; 0 when it is not ranked at all.

        Computed over the full corpus rather than a top-k slice, because MRR
        needs the true rank even when it falls outside any cutoff.
        """
        if len(queries) != len(gold_ids):
            raise ValueError("queries and gold_ids must be the same length")
        position = {ex.id: i for i, ex in enumerate(self.examples)}
        vectors = encoder.encode_queries(queries)
        scores = vectors @ self.embeddings.T
        order = np.argsort(-scores, axis=1)

        ranks: list[int] = []
        for row, gold_id in zip(order, gold_ids):
            gold_idx = position.get(gold_id)
            if gold_idx is None:
                ranks.append(0)
                continue
            ranks.append(int(np.where(row == gold_idx)[0][0]) + 1)
        return ranks

    def save(self, path: str) -> None:
        """Persist embeddings + corpus metadata so eval need not re-encode."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "embeddings.npy", self.embeddings)
        with open(target / "corpus.jsonl", "w") as fh:
            fh.writelines(json.dumps(asdict(ex)) + "\n" for ex in self.examples)

    @classmethod
    def load(cls, path: str) -> EmbeddingIndex:
        """Load an index written by `save`."""
        target = Path(path)
        embeddings = np.load(target / "embeddings.npy")
        with open(target / "corpus.jsonl") as fh:
            examples = [CodeExample(**json.loads(line)) for line in fh if line.strip()]
        return cls(examples, embeddings)
