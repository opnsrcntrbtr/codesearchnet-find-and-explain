"""Summarization metric tests.

`corpus_bleu` is exercised with real sacrebleu — it's a pure deterministic
function, cheap to run for real. `bertscore_f1` is `slow`-marked: it loads a
real (if small) model, so "does it return a sane number" is a property of
the actual library, not something a mock would prove. `llm_judge` is mocked
throughout — no API key here, and it must not spend real credits in tests.
"""

from __future__ import annotations

import csv

import pytest
from helpers import make_example

from csne.config import EvalConfig
from csne.evaluation.reporting import append_result
from csne.evaluation.summarization_metrics import (
    SummarizationResults,
    bertscore_f1,
    corpus_bleu,
    evaluate_summarization,
    llm_judge,
)


class FakeSummarizer:
    """Deterministic stand-in — returns each example's reference verbatim
    (or a corrupted version, via `corrupt`) so metric output is predictable."""

    def __init__(self, corrupt: bool = False):
        self.corrupt = corrupt
        self.batches: list[list] = []

    def summarize(self, example):
        return self._answer(example)

    def summarize_batch(self, examples):
        self.batches.append(examples)
        return [self._answer(ex) for ex in examples]

    def _answer(self, example):
        return "completely unrelated text" if self.corrupt else example.reference_summary

    @property
    def name(self):
        return "fake:corrupt" if self.corrupt else "fake:perfect"


def test_corpus_bleu_perfect_match_scores_100():
    refs = ["parse a json file from disk", "send an email to a recipient"]
    assert corpus_bleu(refs, refs) == pytest.approx(100.0)


def test_corpus_bleu_unrelated_text_scores_low():
    predictions = ["completely different words here"] * 2
    references = ["parse a json file from disk", "send an email to a recipient"]
    score = corpus_bleu(predictions, references)
    assert 0.0 <= score < 20.0


def test_corpus_bleu_rejects_empty():
    with pytest.raises(ValueError, match="no predictions"):
        corpus_bleu([], [])


def test_corpus_bleu_rejects_length_mismatch():
    with pytest.raises(ValueError, match="vs 2 references"):
        corpus_bleu(["one"], ["one", "two"])


def test_llm_judge_averages_parsed_scores(monkeypatch):
    """Mocked end to end: no API key in this environment, and a real call
    would spend credits."""

    class FakeBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class FakeResponse:
        def __init__(self, text):
            self.content = [FakeBlock(text)]
            self.stop_reason = "end_turn"

    class FakeMessages:
        def __init__(self, texts):
            self._texts = iter(texts)

        def create(self, **kwargs):
            return FakeResponse(next(self._texts))

    class FakeClient:
        def __init__(self, texts):
            self.messages = FakeMessages(texts)

    class FakeAnthropicModule:
        @staticmethod
        def Anthropic():
            return FakeClient(["5", "3", "1"])

    import sys

    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropicModule())

    examples = [make_example(id=str(i), repo=f"r/{i}") for i in range(3)]
    score = llm_judge(["a", "b", "c"], examples)
    assert score == pytest.approx((5 + 3 + 1) / 3)


def test_llm_judge_skips_refusals(monkeypatch):
    import sys

    class FakeBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class FakeResponse:
        def __init__(self, text, stop_reason="end_turn"):
            self.content = [FakeBlock(text)] if text else []
            self.stop_reason = stop_reason

    class FakeMessages:
        def __init__(self, responses):
            self._responses = iter(responses)

        def create(self, **kwargs):
            return next(self._responses)

    class FakeClient:
        def __init__(self, responses):
            self.messages = FakeMessages(responses)

    class FakeAnthropicModule:
        @staticmethod
        def Anthropic():
            return FakeClient([FakeResponse("4"), FakeResponse("", "refusal")])

    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropicModule())

    examples = [make_example(id=str(i), repo=f"r/{i}") for i in range(2)]
    score = llm_judge(["a", "b"], examples)
    assert score == 4.0  # only the non-refused example counted


def test_llm_judge_rejects_length_mismatch():
    with pytest.raises(ValueError, match="vs 2 examples"):
        llm_judge(["a"], [make_example(id="1"), make_example(id="2")])


def test_evaluate_summarization_caps_sample_size():
    """`summarization_sample_size` bounds cost -- llm_judge is per-example API
    spend on the anthropic backend, so an uncapped run is a real bill."""
    examples = [make_example(id=str(i), repo=f"r/{i}") for i in range(20)]
    summarizer = FakeSummarizer()
    config = EvalConfig(summarization_metrics=["bleu"], summarization_sample_size=5)

    results = evaluate_summarization(examples, summarizer, config, run_name="test")

    assert results.n_examples == 5
    assert summarizer.batches[0] == examples[:5]


def test_evaluate_summarization_perfect_summarizer_scores_well():
    examples = [
        make_example(id="1", repo="r/1", docstring="Parse a json file from disk."),
        make_example(id="2", repo="r/2", docstring="Send an email to a recipient."),
    ]
    summarizer = FakeSummarizer(corrupt=False)
    config = EvalConfig(summarization_metrics=["bleu"], summarization_sample_size=None)

    results = evaluate_summarization(examples, summarizer, config, run_name="perfect")

    assert isinstance(results, SummarizationResults)
    assert results.bleu == pytest.approx(100.0)
    assert results.backend == "fake:perfect"


def test_evaluate_summarization_only_requested_metrics_are_populated():
    examples = [make_example(id="1", repo="r/1")]
    summarizer = FakeSummarizer()
    config = EvalConfig(summarization_metrics=["bleu"], summarization_sample_size=None)

    results = evaluate_summarization(examples, summarizer, config, run_name="test")

    assert results.bleu is not None
    assert results.bertscore_f1 is None
    assert results.llm_judge_score is None


def test_evaluate_summarization_rejects_empty_examples():
    with pytest.raises(ValueError, match="no examples"):
        evaluate_summarization([], FakeSummarizer(), EvalConfig(), run_name="test")


def test_evaluate_summarization_samples_are_capped_at_ten():
    examples = [make_example(id=str(i), repo=f"r/{i}") for i in range(30)]
    summarizer = FakeSummarizer()
    config = EvalConfig(summarization_metrics=["bleu"], summarization_sample_size=30)

    results = evaluate_summarization(examples, summarizer, config, run_name="test")

    assert len(results.samples) == 10


def test_results_as_row_omits_unset_metrics():
    results = SummarizationResults(run_name="r", backend="b", bleu=42.0, n_examples=5)
    row = results.as_row()
    assert row["bleu"] == 42.0
    assert "bertscore_f1" not in row
    assert "llm_judge_score" not in row


def test_summarization_results_append_to_csv(tmp_path):
    results = SummarizationResults(run_name="r", backend="b", bleu=42.0, n_examples=5)
    append_result(str(tmp_path), "summarization_results.csv", results.as_row())
    with open(tmp_path / "summarization_results.csv", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["run_name"] == "r"
    assert row["bleu"] == "42.0"


class TestBertscoreReal:
    """Loads the real (small) BERTScore model."""

    pytestmark = pytest.mark.slow

    def test_perfect_match_scores_near_one(self):
        refs = ["parse a json file from disk", "send an email to a recipient"]
        assert bertscore_f1(refs, refs) > 0.99

    def test_unrelated_text_scores_lower_than_match(self):
        references = ["parse a json file from disk", "send an email to a recipient"]
        matched = bertscore_f1(references, references)
        unrelated = bertscore_f1(["completely unrelated words"] * 2, references)
        assert unrelated < matched

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="no predictions"):
            bertscore_f1([], [])
