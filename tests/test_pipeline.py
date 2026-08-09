"""Pipeline wiring tests.

`FindAndExplain` is glue: retrieve, then summarize. Fakes stand in for the
encoder/index/summarizer so these test the wiring, not the models.
"""

from __future__ import annotations

from helpers import make_example

from csne.pipeline import Explanation, FindAndExplain
from csne.retrieval.index import SearchHit


class FakeIndex:
    def __init__(self, hits: list[SearchHit]):
        self.hits = hits
        self.searched_with: list[tuple[str, int]] = []

    def search(self, query, encoder, k=10):
        self.searched_with.append((query, k))
        return self.hits[:k]


class FakeSummarizer:
    def __init__(self):
        self.batches: list[list] = []

    def summarize(self, example):
        return f"summary of {example.id}"

    def summarize_batch(self, examples):
        self.batches.append(examples)
        return [f"summary of {ex.id}" for ex in examples]

    @property
    def name(self):
        return "fake"


def _hits(n: int) -> list[SearchHit]:
    return [
        SearchHit(example=make_example(id=str(i), repo=f"r/{i}"), score=1.0 - i * 0.1, rank=i + 1)
        for i in range(n)
    ]


def test_find_delegates_to_index_search():
    """Retrieval-only path costs nothing extra — no summarizer call."""
    index = FakeIndex(_hits(3))
    summarizer = FakeSummarizer()
    pipeline = FindAndExplain(encoder=None, index=index, summarizer=summarizer)

    hits = pipeline.find("a query", k=2)

    assert index.searched_with == [("a query", 2)]
    assert [h.example.id for h in hits] == ["0", "1"]
    assert summarizer.batches == []


def test_explain_zips_hits_and_summaries_in_order():
    index = FakeIndex(_hits(3))
    summarizer = FakeSummarizer()
    pipeline = FindAndExplain(encoder=None, index=index, summarizer=summarizer)

    explanations = pipeline.explain("a query", k=3)

    assert all(isinstance(e, Explanation) for e in explanations)
    assert [e.hit.example.id for e in explanations] == ["0", "1", "2"]
    assert [e.summary for e in explanations] == [
        "summary of 0",
        "summary of 1",
        "summary of 2",
    ]


def test_explain_summarizes_in_one_batch_call():
    """One batch call, not k sequential ones — that's where an API backend's
    concurrency comes from."""
    index = FakeIndex(_hits(5))
    summarizer = FakeSummarizer()
    pipeline = FindAndExplain(encoder=None, index=index, summarizer=summarizer)

    pipeline.explain("q", k=5)

    assert len(summarizer.batches) == 1
    assert len(summarizer.batches[0]) == 5


def test_explain_with_no_hits_returns_empty():
    index = FakeIndex([])
    summarizer = FakeSummarizer()
    pipeline = FindAndExplain(encoder=None, index=index, summarizer=summarizer)

    assert pipeline.explain("q", k=5) == []
    assert summarizer.batches == [[]]
