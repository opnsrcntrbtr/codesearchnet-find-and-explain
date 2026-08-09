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

# BERTScore's default (roberta-large) is ~1.4GB and slow on CPU. This project
# runs summarization eval locally, not on Colab, so a small model keeps a
# report-quality run (summarization_sample_size ~50-100) practical; absolute
# BERTScore values are model-dependent anyway, so what matters is using the
# same model_type across the runs being compared.
_BERTSCORE_MODEL = "distilbert-base-uncased"

_LLM_JUDGE_PROMPT = """You are grading whether a summary accurately describes \
what a Python function does. Judge against the function body — not the \
reference docstring, which may itself be stale or noisy.

Function:
```python
{code}
```

Summary to grade: {summary}

Respond with a single integer from 1 to 5 (1 = inaccurate or misleading, \
5 = accurate and complete) and nothing else."""


@dataclass
class SummarizationResults:
    """Metrics for one summarization run.

    `bleu` is corpus-level sacrebleu, 0-100. `bertscore_f1` is mean F1,
    0-1. `llm_judge_score` is the mean 1-5 rating, unnormalized — the three
    metrics are reported on their own conventional scales, not rescaled to
    match each other.
    """

    run_name: str
    backend: str
    bleu: float | None = None
    bertscore_f1: float | None = None
    llm_judge_score: float | None = None
    n_examples: int = 0
    samples: list[dict[str, str]] = field(default_factory=list)

    def as_row(self) -> dict[str, float | str | int]:
        """Flatten to a single CSV row."""
        row: dict[str, float | str | int] = {
            "run_name": self.run_name,
            "backend": self.backend,
            "n_examples": self.n_examples,
        }
        if self.bleu is not None:
            row["bleu"] = round(self.bleu, 4)
        if self.bertscore_f1 is not None:
            row["bertscore_f1"] = round(self.bertscore_f1, 4)
        if self.llm_judge_score is not None:
            row["llm_judge_score"] = round(self.llm_judge_score, 4)
        return row


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU (sacrebleu) over generated vs. reference summaries.

    0-100 scale, sacrebleu's convention. Empty predictions/references still
    score (as 0), rather than raising, so one degenerate generation doesn't
    crash a whole evaluation run.
    """
    import sacrebleu

    if not predictions:
        raise ValueError("corpus_bleu: no predictions to score")
    if len(predictions) != len(references):
        raise ValueError(
            f"corpus_bleu: {len(predictions)} predictions vs {len(references)} references"
        )
    # sacrebleu takes references as a list of reference sets (one list per
    # reference "column"); we have exactly one reference per prediction.
    return sacrebleu.corpus_bleu(predictions, [references]).score


_EMPTY_PREDICTION_SENTINEL = "[no summary generated]"


def bertscore_f1(predictions: list[str], references: list[str]) -> float:
    """Mean BERTScore F1 over generated vs. reference summaries.

    An empty prediction (a weak summarizer sometimes generates nothing) is
    replaced with a sentinel before scoring, not dropped: `bert_score`'s
    internal empty-string handling crashes the whole batch on a real
    `AttributeError` under current `transformers` (reproduced against a real
    flan-t5-base run — confirmed the empty string itself is the trigger, not
    a transformers-version issue, since a bare tokenizer handles the same
    call fine in isolation). The sentinel is unrelated to any real summary,
    so it scores low against the reference — an empty prediction should be
    penalized, not silently excluded from the average.
    """
    import bert_score

    if not predictions:
        raise ValueError("bertscore_f1: no predictions to score")
    if len(predictions) != len(references):
        raise ValueError(
            f"bertscore_f1: {len(predictions)} predictions vs {len(references)} references"
        )
    safe_predictions = [p if p.strip() else _EMPTY_PREDICTION_SENTINEL for p in predictions]
    _, _, f1 = bert_score.score(
        safe_predictions,
        references,
        lang="en",
        model_type=_BERTSCORE_MODEL,
        verbose=False,
    )
    return float(f1.mean())


def llm_judge(
    predictions: list[str], examples: list[CodeExample], model: str = "claude-sonnet-5"
) -> float:
    """Optional LLM-as-judge score for faithfulness to the code.

    Judges the summary against the function body, not the reference docstring:
    CodeSearchNet docstrings are themselves noisy and sometimes stale.
    """
    import re

    import anthropic

    if not predictions:
        raise ValueError("llm_judge: no predictions to score")
    if len(predictions) != len(examples):
        raise ValueError(f"llm_judge: {len(predictions)} predictions vs {len(examples)} examples")

    client = anthropic.Anthropic()
    scores: list[int] = []
    for example, summary in zip(examples, predictions):
        prompt = _LLM_JUDGE_PROMPT.format(code=example.code, summary=summary)
        response = client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            continue
        text = next((b.text for b in response.content if b.type == "text"), "")
        match = re.search(r"[1-5]", text)
        if match:
            scores.append(int(match.group()))

    if not scores:
        raise RuntimeError("llm_judge: no examples produced a parseable score")
    return sum(scores) / len(scores)


def evaluate_summarization(
    examples: list[CodeExample],
    summarizer: Summarizer,
    config: EvalConfig,
    run_name: str,
) -> SummarizationResults:
    """Generate summaries for a sample of the eval split and score them.

    Capped at `config.summarization_sample_size` (default 50) — running the
    full split costs real API tokens on the anthropic backend, and BLEU/
    BERTScore over a full split adds nothing a well-sized sample doesn't
    already show.
    """
    sample = (
        examples[: config.summarization_sample_size]
        if config.summarization_sample_size is not None
        else examples
    )
    if not sample:
        raise ValueError("evaluate_summarization: no examples to evaluate")

    predictions = summarizer.summarize_batch(sample)
    references = [ex.reference_summary for ex in sample]

    results = SummarizationResults(
        run_name=run_name, backend=summarizer.name, n_examples=len(sample)
    )
    if "bleu" in config.summarization_metrics:
        results.bleu = corpus_bleu(predictions, references)
    if "bertscore" in config.summarization_metrics:
        results.bertscore_f1 = bertscore_f1(predictions, references)
    if "llm_judge" in config.summarization_metrics:
        results.llm_judge_score = llm_judge(predictions, sample)

    # A handful of predictions for spot-checking, not the whole sample --
    # keeps the results log from ballooning on a large sample_size.
    results.samples = [
        {"id": ex.id, "prediction": pred, "reference": ref}
        for ex, pred, ref in zip(sample[:10], predictions[:10], references[:10])
    ]
    return results
