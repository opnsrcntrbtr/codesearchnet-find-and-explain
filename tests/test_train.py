"""Training-input tests.

`train_encoder` itself needs a GPU to be worth testing, but the data it is
handed is pure logic — and getting it wrong means training on the wrong corpus
without any visible signal.
"""

from __future__ import annotations

import pytest
from helpers import make_example

from csne.config import DataConfig, RetrievalConfig
from csne.data.loader import save_split
from csne.retrieval.train import build_training_pairs, train_encoder


@pytest.fixture
def cached_corpus(tmp_path) -> DataConfig:
    """A cached train split of 100 examples across 100 repos."""
    config = DataConfig(cache_dir=str(tmp_path), max_train=None)
    save_split(
        "train",
        [make_example(id=str(i), repo=f"org/repo{i}", docstring=f"does thing {i}")
         for i in range(100)],
        config,
    )
    return config


def test_pairs_are_query_and_formatted_code(cached_corpus: DataConfig):
    """The code side must match what the encoder sees at search time."""
    query, code = build_training_pairs(cached_corpus)[0]
    assert query.startswith("does thing")
    assert code.startswith("f ")  # func_name leads, as in format_code_for_encoding


def test_max_train_none_uses_whole_split(cached_corpus: DataConfig):
    assert len(build_training_pairs(cached_corpus)) == 100


def test_max_train_subsets_the_cache(cached_corpus: DataConfig):
    """The cache holds the full split; a subset run must not train on all of it."""
    config = DataConfig(cache_dir=cached_corpus.cache_dir, max_train=10)
    assert len(build_training_pairs(config)) == 10


def test_max_train_larger_than_corpus_is_harmless(cached_corpus: DataConfig):
    config = DataConfig(cache_dir=cached_corpus.cache_dir, max_train=10_000)
    assert len(build_training_pairs(config)) == 100


def test_subset_is_sampled_not_sliced(cached_corpus: DataConfig):
    """Slicing would take one contiguous block of the corpus, which is
    repo-ordered — the sample must not be the first N in order."""
    config = DataConfig(cache_dir=cached_corpus.cache_dir, max_train=10)
    queries = [q for q, _ in build_training_pairs(config)]
    first_ten = [f"does thing {i}" for i in range(10)]
    assert queries != first_ten


def test_subset_is_deterministic_given_seed(cached_corpus: DataConfig):
    """Two runs of the same config must train on the same data."""
    config = DataConfig(cache_dir=cached_corpus.cache_dir, max_train=10)
    assert build_training_pairs(config) == build_training_pairs(config)


def test_subset_differs_across_seeds(cached_corpus: DataConfig):
    a = DataConfig(cache_dir=cached_corpus.cache_dir, max_train=10, seed=1)
    b = DataConfig(cache_dir=cached_corpus.cache_dir, max_train=10, seed=2)
    assert build_training_pairs(a) != build_training_pairs(b)


def test_unsupported_loss_is_rejected(cached_corpus: DataConfig):
    """Fails before downloading a model and burning GPU time."""
    with pytest.raises(ValueError, match="Unsupported loss"):
        train_encoder(RetrievalConfig(loss="TripletLoss"), cached_corpus)
