"""Command-line entry points, all config-driven.

Every subcommand takes `--config` so a Colab cell and a local shell run the
identical experiment definition:

    csne prepare-data --config configs/experiments/baseline.yaml
    csne train        --config configs/experiments/baseline.yaml
    csne evaluate     --config configs/experiments/baseline.yaml
    csne search       --config configs/experiments/baseline.yaml --query "parse a json file"
"""

from __future__ import annotations

import argparse
import logging
import sys

from csne.config import ExperimentConfig, load_experiment

log = logging.getLogger("csne")


def build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser and its subcommands."""
    parser = argparse.ArgumentParser(prog="csne", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True, help="path to an experiment YAML")
        return sub

    add("prepare-data", "download, clean, split, and cache the corpus")
    add("train", "fine-tune the retrieval encoder")

    evaluate = add("evaluate", "score retrieval and append results to CSV")
    evaluate.add_argument(
        "--checkpoint",
        help="encoder to evaluate; defaults to the config's base model",
    )
    evaluate.add_argument(
        "--bm25", action="store_true", help="also evaluate the lexical BM25 baseline"
    )
    evaluate.add_argument(
        "--summarize",
        action="store_true",
        help="also score summarization (costs API tokens on the anthropic backend; "
        "capped at evaluation.summarization_sample_size)",
    )

    search = add("search", "run one retrieval-only query (free — no summarization cost)")
    search.add_argument("--query", required=True)
    search.add_argument("-k", type=int, default=5)

    explain = add("explain", "retrieve and summarize each result (costs API tokens on the anthropic backend)")
    explain.add_argument("--query", required=True)
    explain.add_argument("-k", type=int, default=5)

    return parser


def _load(args: argparse.Namespace) -> ExperimentConfig:
    return load_experiment(args.config)


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Download, clean, split, and cache the corpus."""
    from csne.data.splits import prepare_official_splits

    config = _load(args)
    splits = prepare_official_splits(config.data)
    for name, examples in splits.items():
        with_summary = sum(1 for ex in examples if ex.summary)
        log.info(
            "%s: %d examples (%d with joined summaries) -> %s",
            name,
            len(examples),
            with_summary,
            config.data.cache_dir,
        )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Fine-tune the retrieval encoder."""
    from csne.retrieval.train import train_encoder

    config = _load(args)
    train_encoder(config.retrieval, config.data, run_name=config.name)
    log.info("saved encoder to %s", config.retrieval.output_dir)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run retrieval (and optionally summarization) evaluation, appending to
    the results CSVs."""
    from csne.data.loader import load_prepared
    from csne.evaluation.reporting import (
        RETRIEVAL_CSV,
        SUMMARIZATION_CSV,
        append_result,
        config_hash,
    )
    from csne.evaluation.retrieval_metrics import evaluate_bm25, evaluate_retrieval
    from csne.retrieval.encoder import CodeSearchEncoder

    config = _load(args)
    examples = load_prepared("test", config.data)
    digest = config_hash(config)

    results = []
    if args.bm25:
        results.append(evaluate_bm25(examples, config.evaluation))

    encoder = (
        CodeSearchEncoder.from_checkpoint(args.checkpoint, config.retrieval.max_seq_length)
        if args.checkpoint
        else CodeSearchEncoder.from_config(config.retrieval)
    )
    results.append(evaluate_retrieval(examples, encoder, config.evaluation, config.name))

    for result in results:
        append_result(
            config.evaluation.results_dir,
            RETRIEVAL_CSV,
            {"config_hash": digest, **result.as_row()},
        )
        log.info(
            "%s: MRR=%.4f %s",
            result.run_name,
            result.mrr,
            " ".join(f"R@{k}={v:.4f}" for k, v in sorted(result.recall_at_k.items())),
        )

    if args.summarize:
        from csne.evaluation.summarization_metrics import evaluate_summarization
        from csne.summarization.base import build_summarizer

        summarizer = build_summarizer(config.summarization)
        summ_result = evaluate_summarization(examples, summarizer, config.evaluation, config.name)
        append_result(
            config.evaluation.results_dir,
            SUMMARIZATION_CSV,
            {"config_hash": digest, **summ_result.as_row()},
        )
        log.info(
            "%s (%s, n=%d): %s",
            summ_result.run_name,
            summ_result.backend,
            summ_result.n_examples,
            ", ".join(
                f"{k}={v:.4f}"
                for k, v in (
                    ("bleu", summ_result.bleu),
                    ("bertscore_f1", summ_result.bertscore_f1),
                    ("llm_judge", summ_result.llm_judge_score),
                )
                if v is not None
            ),
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run one find-and-explain query against a built index."""
    from csne.data.loader import load_prepared
    from csne.retrieval.encoder import CodeSearchEncoder
    from csne.retrieval.index import EmbeddingIndex

    config = _load(args)
    examples = load_prepared("test", config.data)
    encoder = (
        CodeSearchEncoder.from_checkpoint(args.checkpoint, config.retrieval.max_seq_length)
        if getattr(args, "checkpoint", None)
        else CodeSearchEncoder.from_config(config.retrieval)
    )
    index = EmbeddingIndex.build(examples, encoder)

    for hit in index.search(args.query, encoder, k=args.k):
        print(f"\n[{hit.rank}] {hit.score:.3f}  {hit.example.repo}:{hit.example.path}")
        print(f"    {hit.example.func_name}  {hit.example.id}")
        print(f"    {hit.example.reference_summary}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Retrieve top-k and summarize each result."""
    from csne.pipeline import FindAndExplain

    config = _load(args)
    pipeline = FindAndExplain.from_config(config)

    for explanation in pipeline.explain(args.query, k=args.k):
        hit = explanation.hit
        print(f"\n[{hit.rank}] {hit.score:.3f}  {hit.example.repo}:{hit.example.path}")
        print(f"    {hit.example.func_name}  {hit.example.id}")
        print(f"    {explanation.summary}")
    return 0


_COMMANDS = {
    "prepare-data": cmd_prepare_data,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "search": cmd_search,
    "explain": cmd_explain,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
