"""Summarization tests. Both backends are mocked — no API calls, no downloads."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Phase 3: implement alongside csne.summarization")


def test_build_summarizer_dispatches_on_backend_name():
    """`backend: anthropic` and `backend: hf` yield their respective classes."""


def test_build_summarizer_rejects_unknown_backend():
    """An unknown backend fails loudly at config load, not mid-experiment."""


def test_prompt_template_renders_all_fields():
    """Every `{placeholder}` in the template is filled; none leak into the prompt."""


def test_summarize_batch_preserves_input_order():
    """Concurrent API calls must not reorder results relative to inputs."""


def test_both_backends_satisfy_the_summarizer_interface():
    """Backends stay swappable — the report compares them on identical inputs."""
