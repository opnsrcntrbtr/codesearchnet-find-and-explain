"""Evaluation layer: retrieval and summarization metrics, plus results logging."""

from csne.evaluation.reporting import append_result, results_path
from csne.evaluation.retrieval_metrics import RetrievalResults, evaluate_bm25, evaluate_retrieval
from csne.evaluation.summarization_metrics import SummarizationResults, evaluate_summarization

__all__ = [
    "RetrievalResults",
    "SummarizationResults",
    "append_result",
    "evaluate_bm25",
    "evaluate_retrieval",
    "evaluate_summarization",
    "results_path",
]
