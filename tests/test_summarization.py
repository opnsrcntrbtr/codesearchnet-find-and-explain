"""Summarization tests.

The Anthropic backend is exercised structurally (dispatch, interface,
concurrency ordering) with the client's `summarize` mocked — there is no
`ANTHROPIC_API_KEY` in this environment, and these tests must not spend real
API credits. The HF backend additionally gets `slow`-marked tests that load
the real local model, since "does it produce a summary" is a property of the
actual model and a mock would assert nothing.
"""

from __future__ import annotations

import time

import pytest
from helpers import make_example

from csne.config import SummarizationConfig
from csne.summarization.anthropic_backend import AnthropicSummarizer
from csne.summarization.base import (
    Summarizer,
    build_summarizer,
    load_prompt_template,
    render_prompt,
)
from csne.summarization.hf_backend import HFSummarizer


def test_build_summarizer_dispatches_anthropic():
    summarizer = build_summarizer(SummarizationConfig(backend="anthropic", model="claude-sonnet-5"))
    assert isinstance(summarizer, AnthropicSummarizer)
    assert summarizer.name == "anthropic:claude-sonnet-5"


def test_build_summarizer_dispatches_hf(monkeypatch):
    """The real HF backend loads model weights; stub construction for this
    fast dispatch-only check and leave real generation to the slow tests."""

    def fake_init(self, config):
        self.config = config
        self.model_name = config.model or "stub-model"

    monkeypatch.setattr(HFSummarizer, "__init__", fake_init)
    summarizer = build_summarizer(SummarizationConfig(backend="hf"))
    assert isinstance(summarizer, HFSummarizer)


def test_build_summarizer_rejects_unknown_backend():
    """An unknown backend fails loudly at config load, not mid-experiment."""
    with pytest.raises(ValueError, match="Unknown summarization backend"):
        build_summarizer(SummarizationConfig(backend="openai"))


def test_prompt_template_renders_all_fields():
    """Every `{placeholder}` in the template is filled; none leak into the prompt."""
    template = load_prompt_template("prompts/function_summary.txt")
    example = make_example(func_name="slugify", repo="acme/util", path="lib/text.py")
    rendered = render_prompt(template, example)
    assert "slugify" in rendered
    assert "acme/util" in rendered
    assert "lib/text.py" in rendered
    assert "{func_name}" not in rendered
    assert "{code}" not in rendered


def test_render_prompt_survives_braces_in_code():
    """Code containing dict literals/f-strings must not break template.format."""
    template = load_prompt_template("prompts/function_summary.txt")
    example = make_example(code="def f():\n    return {1: 2, 'a': f'{1+1}'}")
    rendered = render_prompt(template, example)
    assert "{1: 2, 'a':" in rendered


def test_load_prompt_template_rejects_empty(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_prompt_template(str(empty))


def test_summarize_batch_preserves_input_order(monkeypatch):
    """Concurrent API calls must not reorder results relative to inputs.

    Sleeps are staggered in reverse so the *last* input would finish *first*
    if ordering depended on completion time rather than `pool.map`'s
    input-order guarantee.
    """
    summarizer = build_summarizer(SummarizationConfig(backend="anthropic"))
    examples = [make_example(id=str(i), repo=f"r/{i}") for i in range(5)]

    def fake_summarize(ex):
        time.sleep(0.01 * (len(examples) - int(ex.id)))
        return f"summary-{ex.id}"

    monkeypatch.setattr(summarizer, "summarize", fake_summarize)
    results = summarizer.summarize_batch(examples)
    assert results == [f"summary-{i}" for i in range(5)]


def test_summarize_batch_empty_input():
    summarizer = build_summarizer(SummarizationConfig(backend="anthropic"))
    assert summarizer.summarize_batch([]) == []


def test_both_backends_satisfy_the_summarizer_interface():
    """Backends stay swappable — the report compares them on identical inputs."""
    assert issubclass(AnthropicSummarizer, Summarizer)
    assert issubclass(HFSummarizer, Summarizer)


# --- Slow: loads the real local model -------------------------------------

@pytest.fixture(scope="module")
def hf_summarizer() -> Summarizer:
    return build_summarizer(SummarizationConfig(backend="hf", max_output_tokens=32))


class TestHFSummarizerReal:
    """Loads the real local model — property tests, not mocks."""

    pytestmark = pytest.mark.slow

    def test_generates_nonempty_text(self, hf_summarizer: Summarizer):
        example = make_example(
            func_name="factorial",
            code="def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
        )
        summary = hf_summarizer.summarize(example)
        assert isinstance(summary, str)
        assert summary.strip() != ""

    def test_summarize_batch_matches_example_count(self, hf_summarizer: Summarizer):
        examples = [
            make_example(id="a", func_name="add", code="def add(a, b):\n    return a + b"),
            make_example(id="b", func_name="sub", code="def sub(a, b):\n    return a - b"),
        ]
        summaries = hf_summarizer.summarize_batch(examples)
        assert len(summaries) == 2
        assert all(isinstance(s, str) for s in summaries)
