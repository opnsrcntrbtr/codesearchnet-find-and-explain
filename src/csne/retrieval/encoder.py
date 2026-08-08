"""The dual-purpose encoder that embeds both queries and code.

A single shared encoder (rather than separate query/code towers) keeps the
model small enough to fine-tune on a free Colab T4 and matches the
sentence-transformers training recipe used in `train.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from csne.config import RetrievalConfig
from csne.data.loader import CodeExample

if TYPE_CHECKING:  # pragma: no cover - import cost is only paid at runtime
    from sentence_transformers import SentenceTransformer


def format_code_for_encoding(example: CodeExample) -> str:
    """Render one function as the string the encoder sees.

    The function name is prepended because it is often the single most
    query-like token in the whole snippet (`parse_json_file` vs. its body),
    and MiniLM truncates at 256 tokens — so the most informative signal has
    to come first, before any risk of truncation.
    """
    return f"{example.func_name} {example.code}".strip()


class CodeSearchEncoder:
    """Wraps a sentence-transformers model with code-aware input formatting."""

    def __init__(self, model_name_or_path: str, max_seq_length: int = 256) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name_or_path
        self.model: SentenceTransformer = SentenceTransformer(model_name_or_path)
        self.model.max_seq_length = max_seq_length

    @classmethod
    def from_config(cls, config: RetrievalConfig) -> CodeSearchEncoder:
        """Build an untrained encoder from `config.base_model`."""
        return cls(config.base_model, max_seq_length=config.max_seq_length)

    @classmethod
    def from_checkpoint(cls, path: str, max_seq_length: int = 256) -> CodeSearchEncoder:
        """Load a fine-tuned encoder saved by `train_encoder`."""
        return cls(path, max_seq_length=max_seq_length)

    @property
    def dimension(self) -> int:
        """Embedding width, needed to allocate the index.

        The accessor was renamed in sentence-transformers 5.x; fall back so
        the package still works on the 3.x/4.x versions Colab may pin.
        """
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode raw strings to L2-normalized float32 vectors.

        Normalization is done here, once, so the index can treat a dot
        product as cosine similarity.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed natural-language queries. Returns L2-normalized vectors."""
        return self._encode(queries, batch_size)

    def encode_code(self, examples: list[CodeExample], batch_size: int = 64) -> np.ndarray:
        """Embed code snippets. Returns L2-normalized vectors."""
        return self._encode([format_code_for_encoding(ex) for ex in examples], batch_size)

    def save(self, path: str) -> None:
        """Persist the encoder so a Colab-trained model can be pulled locally."""
        self.model.save(path)
