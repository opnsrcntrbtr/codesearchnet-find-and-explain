"""Shared fixtures. Tests must never hit the network or a GPU."""

from __future__ import annotations

import pytest
from helpers import make_example

from csne.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    RetrievalConfig,
    SummarizationConfig,
)
from csne.data.loader import CodeExample


@pytest.fixture
def sample_examples() -> list[CodeExample]:
    """Hand-written examples covering the cases preprocessing must get right.

    A reST docstring, a Google-style docstring, a clean one-liner, pure
    boilerplate, a too-short docstring, and a non-English docstring.
    """
    return [
        make_example(
            id="u1",
            repo="acme/rest",
            func_name="connect",
            docstring=(
                "Open a database connection.\n\n"
                ":param host: the host to connect to\n"
                ":param port: the port\n"
                ":returns: a live connection\n"
            ),
            code='def connect(host, port):\n    """Open a database connection."""\n    return Conn(host, port)',
        ),
        make_example(
            id="u2",
            repo="acme/google",
            func_name="retry",
            docstring=(
                "Retry a callable with exponential backoff.\n\n"
                "Args:\n    fn: the callable to retry\n"
                "Returns:\n    Whatever fn returns\n"
            ),
            code='def retry(fn):\n    """Retry a callable."""\n    return fn()',
        ),
        make_example(
            id="u3",
            repo="acme/clean",
            func_name="slugify",
            docstring="Convert a string into a URL-safe slug.",
        ),
        make_example(id="u4", repo="acme/boiler", func_name="__init__", docstring="TODO"),
        make_example(id="u5", repo="acme/short", func_name="g", docstring="Does."),
        make_example(
            id="u6",
            repo="acme/cjk",
            func_name="h",
            docstring="将字符串转换为安全的网址片段以便使用",
        ),
    ]


@pytest.fixture
def data_config() -> DataConfig:
    """A `DataConfig` with tiny limits; never used to hit the network in tests."""
    return DataConfig(max_train=10, max_eval=10, cache_dir="/tmp/csne-test-cache")


@pytest.fixture
def tiny_config(data_config: DataConfig) -> ExperimentConfig:
    """A full `ExperimentConfig` pointed at local fixtures, not Hugging Face."""
    return ExperimentConfig(
        name="test",
        data=data_config,
        retrieval=RetrievalConfig(epochs=0, batch_size=2),
        summarization=SummarizationConfig(backend="hf"),
        evaluation=EvalConfig(retrieval_k=[1, 5], distractor_pool_size=10),
    )
