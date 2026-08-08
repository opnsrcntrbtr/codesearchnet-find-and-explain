"""Summarization metrics against reference docstrings.

BLEU is reported for comparability with prior code-summarization work;
BERTScore is reported because BLEU penalizes valid paraphrases heavily at
the one-sentence lengths this task produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from csne.config import EvalConfig
from csne.data.loader import CodeExample
from csne.summarization.base import Summarizer


@dataclass
class SummarizationResults:
    """Metrics for one summarization run."""

    run_name: str
    backend: str
    bleu: float | None = None
    bertscore_f1: float | None = None
    llm_judge_score: float | None = None
    n_examples: int = 0
    samples: list[dict[str, str]] = field(default_factory=list)


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU (sacrebleu) over generated vs. reference summaries."""
    raise NotImplementedError


def bertscore_f1(predictions: list[str], references: list[str]) -> float:
    """Mean BERTScore F1 over generated vs. reference summaries."""
    raise NotImplementedError


def llm_judge(
    predictions: list[str], examples: list[CodeExample], model: str = "claude-sonnet-5"
) -> float:
    """Optional LLM-as-judge score for faithfulness to the code.

    Judges the summary against the function body, not the reference docstring:
    CodeSearchNet docstrings are themselves noisy and sometimes stale.
    """
    raise NotImplementedError


def evaluate_summarization(
    examples: list[CodeExample],
    summarizer: Summarizer,
    config: EvalConfig,
    run_name: str,
) -> SummarizationResults:
    """Generate summaries for the eval split and score them."""
    raise NotImplementedError
