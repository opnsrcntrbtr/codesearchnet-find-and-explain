"""Metric tests. These are pure functions over rank lists — test them hard,
since every number in the report flows through here.
"""

from __future__ import annotations

import csv
import math

import pytest
from helpers import make_example

from csne.config import EvalConfig
from csne.evaluation.reporting import append_result, config_hash, results_path
from csne.evaluation.retrieval_metrics import (
    _tokenize,
    evaluate_bm25,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


def test_mrr_known_values():
    """Ranks [1, 2, 4] -> (1 + 0.5 + 0.25) / 3."""
    assert mean_reciprocal_rank([1, 2, 4]) == pytest.approx((1 + 0.5 + 0.25) / 3)


def test_mrr_counts_missing_gold_as_zero():
    """A gold outside the retrieved list contributes 0, not a crash or 1/0."""
    assert mean_reciprocal_rank([1, 0]) == pytest.approx(0.5)


def test_mrr_of_empty_is_zero():
    assert mean_reciprocal_rank([]) == 0.0


def test_recall_at_k_boundary():
    """A gold at exactly rank k counts as a hit; k+1 does not."""
    assert recall_at_k([5], 5) == 1.0
    assert recall_at_k([6], 5) == 0.0


def test_recall_at_k_ignores_missing_gold():
    """Rank 0 means unranked, and must never count as a hit."""
    assert recall_at_k([0], 5) == 0.0


def test_recall_at_k_fraction():
    assert recall_at_k([1, 3, 20, 0], 5) == pytest.approx(0.5)


def test_ndcg_at_k_single_relevant_document():
    """With one relevant doc, NDCG@k = 1 / log2(rank + 1)."""
    assert ndcg_at_k([3], 10) == pytest.approx(1 / math.log2(4))


def test_ndcg_at_rank_one_is_perfect():
    assert ndcg_at_k([1], 10) == pytest.approx(1.0)


def test_ndcg_zero_outside_cutoff():
    assert ndcg_at_k([11], 10) == 0.0


def test_ndcg_never_exceeds_one():
    """A broken ideal-DCG would show up as a score above 1."""
    assert ndcg_at_k([1, 1, 1], 10) <= 1.0


def test_bm25_ranks_exact_lexical_match_first():
    """A query lifted from a function's own docstring should retrieve it."""
    examples = [
        make_example(
            id="a", repo="r/a", func_name="parse_json_file",
            docstring="Parse a json file from disk.",
            code="def parse_json_file(path):\n    return json.load(open(path))",
        ),
        make_example(
            id="b", repo="r/b", func_name="send_email",
            docstring="Send an email message to a recipient.",
            code="def send_email(to, body):\n    smtp.send(to, body)",
        ),
        make_example(
            id="c", repo="r/c", func_name="rotate_matrix",
            docstring="Rotate a square matrix ninety degrees.",
            code="def rotate_matrix(m):\n    return list(zip(*m[::-1]))",
        ),
    ]
    results = evaluate_bm25(examples, EvalConfig(retrieval_k=[1], distractor_pool_size=10))
    assert results.recall_at_k[1] == 1.0


def test_tokenize_splits_snake_case_identifiers():
    """`parse_json_file` must yield "parse json file", or BM25 is unfairly weak
    against the embedding model and the comparison proves nothing."""
    assert _tokenize("def parse_json_file(path):") == ["def", "parse", "json", "file", "path"]


def test_bm25_reports_every_query():
    examples = [
        make_example(id=str(i), repo=f"r/{i}", docstring=f"unique docstring number {i}")
        for i in range(5)
    ]
    results = evaluate_bm25(examples, EvalConfig(retrieval_k=[1, 5], distractor_pool_size=10))
    assert results.n_queries == 5


def test_distractor_pool_caps_corpus():
    """The corpus is capped at pool_size + 1 to match the CSN protocol."""
    examples = [
        make_example(id=str(i), repo=f"r/{i}", docstring=f"unique docstring number {i}")
        for i in range(50)
    ]
    results = evaluate_bm25(examples, EvalConfig(retrieval_k=[1], distractor_pool_size=9))
    assert results.n_queries == 10


def test_append_result_writes_header_once(tmp_path):
    """Appending a second row must not repeat the CSV header."""
    append_result(str(tmp_path), "r.csv", {"run_name": "a", "mrr": 0.5})
    append_result(str(tmp_path), "r.csv", {"run_name": "b", "mrr": 0.6})

    with open(tmp_path / "r.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["run_name"] for r in rows] == ["a", "b"]


def test_append_result_records_provenance(tmp_path):
    """Every row carries a timestamp and commit so a number is traceable."""
    append_result(str(tmp_path), "r.csv", {"run_name": "a", "mrr": 0.5})
    with open(tmp_path / "r.csv", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["timestamp"]
    assert "git_commit" in row


def test_append_result_rejects_schema_drift(tmp_path):
    """A new column would be silently dropped; fail instead."""
    append_result(str(tmp_path), "r.csv", {"run_name": "a", "mrr": 0.5})
    with pytest.raises(ValueError, match="not in the existing header"):
        append_result(str(tmp_path), "r.csv", {"run_name": "b", "brand_new": 1})


def test_config_hash_is_stable_and_sensitive(tiny_config):
    """Identical configs hash alike; a changed hyperparameter does not."""
    import dataclasses

    assert config_hash(tiny_config) == config_hash(tiny_config)
    changed = dataclasses.replace(
        tiny_config, retrieval=dataclasses.replace(tiny_config.retrieval, epochs=9)
    )
    assert config_hash(tiny_config) != config_hash(changed)


def test_results_path_creates_directory(tmp_path):
    path = results_path(str(tmp_path / "nested"), "r.csv")
    assert path.parent.is_dir()
