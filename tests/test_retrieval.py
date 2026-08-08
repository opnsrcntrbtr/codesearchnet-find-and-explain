"""Retrieval tests: embedding shape/normalization and index correctness.

The encoder tests load the real MiniLM checkpoint (~90MB, cached after the
first run) because the properties under test — normalization, dimensionality,
ranking — are properties of the actual model, and a mock would assert nothing.
They are marked `slow` so the fast suite stays offline.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_example

from csne.config import RetrievalConfig
from csne.retrieval.encoder import CodeSearchEncoder, format_code_for_encoding
from csne.retrieval.index import EmbeddingIndex

pytestmark = pytest.mark.slow

CORPUS = [
    make_example(
        id="json", repo="r/json", func_name="parse_json_file",
        docstring="Parse a json file from disk and return a dict.",
        code="def parse_json_file(path):\n    with open(path) as fh:\n        return json.load(fh)",
    ),
    make_example(
        id="email", repo="r/email", func_name="send_email",
        docstring="Send an email message to a recipient over SMTP.",
        code="def send_email(to, body):\n    server = smtplib.SMTP()\n    server.sendmail(to, body)",
    ),
    make_example(
        id="matrix", repo="r/matrix", func_name="rotate_matrix",
        docstring="Rotate a square matrix by ninety degrees clockwise.",
        code="def rotate_matrix(m):\n    return [list(r) for r in zip(*m[::-1])]",
    ),
]


@pytest.fixture(scope="module")
def encoder() -> CodeSearchEncoder:
    return CodeSearchEncoder.from_config(RetrievalConfig(max_seq_length=128))


@pytest.fixture(scope="module")
def index(encoder: CodeSearchEncoder) -> EmbeddingIndex:
    return EmbeddingIndex.build(CORPUS, encoder)


def test_encoder_outputs_are_l2_normalized(encoder: CodeSearchEncoder):
    """The index treats a dot product as cosine — normalization is load-bearing."""
    vectors = encoder.encode_queries(["parse a json file", "send an email"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_code_embeddings_are_l2_normalized(encoder: CodeSearchEncoder):
    vectors = encoder.encode_code(CORPUS)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_query_and_code_embeddings_share_dimensionality(encoder: CodeSearchEncoder):
    """Both sides must land in the same space to be comparable."""
    q = encoder.encode_queries(["parse a json file"])
    c = encoder.encode_code(CORPUS[:1])
    assert q.shape[1] == c.shape[1] == encoder.dimension


def test_encoding_empty_input_returns_empty_matrix(encoder: CodeSearchEncoder):
    """An empty eval slice must not crash the matmul in the index."""
    assert encoder.encode_queries([]).shape == (0, encoder.dimension)


def test_format_code_leads_with_func_name():
    """MiniLM truncates at 256 tokens, so the most query-like signal goes first."""
    assert format_code_for_encoding(CORPUS[0]).startswith("parse_json_file")


def test_index_returns_k_hits_in_descending_score_order(
    index: EmbeddingIndex, encoder: CodeSearchEncoder
):
    """Ranks are 1-indexed and sorted; metrics depend on this ordering."""
    hits = index.search("read a json file from disk", encoder, k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_index_k_larger_than_corpus_is_clamped(
    index: EmbeddingIndex, encoder: CodeSearchEncoder
):
    """Asking for more results than exist returns everything, not an error."""
    assert len(index.search("anything", encoder, k=99)) == len(CORPUS)


def test_index_finds_semantic_match_first(index: EmbeddingIndex, encoder: CodeSearchEncoder):
    """A paraphrased query — no shared content words — still ranks its function first."""
    hits = index.search("load configuration from a json document", encoder, k=1)
    assert hits[0].example.id == "json"


def test_index_distinguishes_unrelated_queries(
    index: EmbeddingIndex, encoder: CodeSearchEncoder
):
    """Different queries must not all collapse onto the same nearest neighbor."""
    top = [index.search(q, encoder, k=1)[0].example.id for q in
           ("send a mail message over smtp", "turn a grid ninety degrees")]
    assert top == ["email", "matrix"]


def test_gold_ranks_are_one_indexed(index: EmbeddingIndex, encoder: CodeSearchEncoder):
    """Rank 1 is the best possible; 0 is reserved for 'not ranked'."""
    ranks = index.gold_ranks([ex.query for ex in CORPUS], [ex.id for ex in CORPUS], encoder)
    assert all(1 <= r <= len(CORPUS) for r in ranks)


def test_gold_ranks_report_zero_for_unknown_id(
    index: EmbeddingIndex, encoder: CodeSearchEncoder
):
    """An id absent from the corpus is a miss, not an exception."""
    assert index.gold_ranks(["anything"], ["not-in-corpus"], encoder) == [0]


def test_search_batch_matches_single_search(
    index: EmbeddingIndex, encoder: CodeSearchEncoder
):
    """Batching must not change results — evaluation uses the batched path."""
    queries = ["parse a json file", "send an email"]
    batched = index.search_batch(queries, encoder, k=2)
    single = [index.search(q, encoder, k=2) for q in queries]
    assert [[h.example.id for h in row] for row in batched] == [
        [h.example.id for h in row] for row in single
    ]


def test_index_roundtrips_through_save_and_load(
    index: EmbeddingIndex, encoder: CodeSearchEncoder, tmp_path
):
    """A reloaded index returns the same hits as the one that wrote it."""
    index.save(str(tmp_path / "idx"))
    reloaded = EmbeddingIndex.load(str(tmp_path / "idx"))
    assert reloaded.examples == index.examples
    assert np.allclose(reloaded.embeddings, index.embeddings)
    assert [h.example.id for h in reloaded.search("json", encoder, k=3)] == [
        h.example.id for h in index.search("json", encoder, k=3)
    ]


def test_index_rejects_corpus_embedding_mismatch():
    """A silent off-by-one here would misattribute every retrieved result."""
    with pytest.raises(ValueError, match="mismatch"):
        EmbeddingIndex(CORPUS, np.zeros((2, 384), dtype=np.float32))
