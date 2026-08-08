"""Summarization layer: explain a retrieved function in natural language.

Two interchangeable backends behind one `Summarizer` interface — an
Anthropic API backend and a local Hugging Face backend.
"""

from csne.summarization.base import Summarizer, build_summarizer

__all__ = ["Summarizer", "build_summarizer"]
