"""Retrieval metrics, following the CodeSearchNet challenge protocol.

Each query is scored against a corpus containing its true function plus
distractors, so numbers stay comparable across runs and against published
baselines.

Rank convention throughout: 1-indexed, and 0 means "the gold document was not
ranked at all". Every metric here treats 0 as a miss rather than a rank.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from csne.config import EvalConfig
from csne.data.loader import CodeExample
from csne.retrieval.encoder import CodeSearchEncoder, format_code_for_encoding
from csne.retrieval.index import EmbeddingIndex


@dataclass
class RetrievalResults:
    """Metrics for one retrieval run."""

    run_name: str
    mrr: float
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    n_queries: int = 0

    def as_row(self) -> dict[str, float | str | int]:
        """Flatten to a single CSV row."""
        row: dict[str, float | str | int] = {
            "run_name": self.run_name,
            "mrr": round(self.mrr, 4),
            "n_queries": self.n_queries,
        }
        for k, v in sorted(self.recall_at_k.items()):
            row[f"recall@{k}"] = round(v, 4)
        for k, v in sorted(self.ndcg_at_k.items()):
            row[f"ndcg@{k}"] = round(v, 4)
        return row


def mean_reciprocal_rank(ranks: list[int]) -> float:
    """MRR over 1-indexed gold ranks. A missing gold contributes 0."""
    if not ranks:
        return 0.0
    return sum(1.0 / r if r > 0 else 0.0 for r in ranks) / len(ranks)


def recall_at_k(ranks: list[int], k: int) -> float:
    """Fraction of queries whose gold function appears in the top k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if 0 < r <= k) / len(ranks)


def ndcg_at_k(ranks: list[int], k: int) -> float:
    """NDCG@k for the single-relevant-document case.

    With exactly one relevant document the ideal DCG is 1, so this reduces to
    the mean of 1/log2(rank + 1) for golds inside the cutoff.
    """
    if not ranks:
        return 0.0
    return sum(1.0 / math.log2(r + 1) if 0 < r <= k else 0.0 for r in ranks) / len(ranks)


def _summarize(run_name: str, ranks: list[int], config: EvalConfig) -> RetrievalResults:
    return RetrievalResults(
        run_name=run_name,
        mrr=mean_reciprocal_rank(ranks),
        recall_at_k={k: recall_at_k(ranks, k) for k in config.retrieval_k},
        ndcg_at_k={k: ndcg_at_k(ranks, k) for k in config.retrieval_k},
        n_queries=len(ranks),
    )


def _eval_corpus(examples: list[CodeExample], config: EvalConfig) -> list[CodeExample]:
    """Cap the corpus at the configured pool size.

    CSN scores each query against its gold plus ~999 distractors. Using the
    entire eval split instead would make the task strictly harder and the
    numbers incomparable to published results.
    """
    limit = config.distractor_pool_size + 1
    return examples[:limit] if limit < len(examples) else examples


def evaluate_retrieval(
    examples: list[CodeExample],
    encoder: CodeSearchEncoder,
    config: EvalConfig,
    run_name: str,
    index: EmbeddingIndex | None = None,
) -> RetrievalResults:
    """Score an encoder over the eval split and return all retrieval metrics."""
    corpus = _eval_corpus(examples, config)
    search_index = index if index is not None else EmbeddingIndex.build(corpus, encoder)
    ranks = search_index.gold_ranks(
        [ex.query for ex in corpus], [ex.id for ex in corpus], encoder
    )
    return _summarize(run_name, ranks, config)


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    """Split into lowercase word tokens, also splitting snake_case identifiers.

    `parse_json_file` has to match the query "parse json file" or BM25 is an
    unfairly weak baseline — the comparison would prove nothing.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text):
        parts = match.lower().split("_")
        tokens.extend(p for p in parts if p)
    return tokens


def evaluate_bm25(
    examples: list[CodeExample],
    config: EvalConfig,
    run_name: str = "bm25",
    k1: float = 1.2,
    b: float = 0.75,
) -> RetrievalResults:
    """Lexical BM25 baseline, evaluated on the identical query/corpus setup.

    Implemented here rather than pulled in as a dependency: it is ~30 lines,
    and it must tokenize identically to the comparison above to be fair.
    """
    corpus = _eval_corpus(examples, config)
    docs = [_tokenize(format_code_for_encoding(ex)) for ex in corpus]
    lengths = [len(d) for d in docs]
    avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0
    n_docs = len(docs)

    doc_freq: Counter[str] = Counter()
    term_freqs: list[Counter[str]] = []
    for doc in docs:
        tf = Counter(doc)
        term_freqs.append(tf)
        doc_freq.update(tf.keys())

    idf = {
        term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
    }

    ranks: list[int] = []
    for gold_idx, ex in enumerate(corpus):
        query_terms = _tokenize(ex.query)
        scores = [0.0] * n_docs
        for i, tf in enumerate(term_freqs):
            length_norm = k1 * (1 - b + b * (lengths[i] / avgdl if avgdl else 0.0))
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq:
                    score += idf[term] * (freq * (k1 + 1)) / (freq + length_norm)
            scores[i] = score
        # Ties are broken toward the lower index; without the tiebreak, a
        # query matching nothing would report rank 1 whenever the gold
        # happened to sit first.
        order = sorted(range(n_docs), key=lambda i: (-scores[i], i))
        ranks.append(order.index(gold_idx) + 1)

    return _summarize(run_name, ranks, config)
