"""Fine-tune the encoder on (query, code) pairs.

Runs on Colab GPU via Colab MCP / `claude-colab`; the same entry point runs
locally on CPU for smoke tests with a truncated corpus.

Default loss is MultipleNegativesRankingLoss: it treats other in-batch code
snippets as negatives, so no explicit negative mining is needed and larger
batches directly mean harder negatives.
"""

from __future__ import annotations

import random
from pathlib import Path

from csne.config import DataConfig, RetrievalConfig
from csne.data.loader import CodeExample, load_prepared
from csne.retrieval.encoder import CodeSearchEncoder, format_code_for_encoding

_LOSSES = {"MultipleNegativesRankingLoss", "CachedMultipleNegativesRankingLoss"}


def build_training_pairs(data_config: DataConfig) -> list[tuple[str, str]]:
    """Load the train split as `(query, formatted_code)` tuples.

    The code side is formatted exactly as `CodeSearchEncoder` formats it at
    inference; training on a different rendering than we search with would
    silently cost accuracy.

    `max_train` is applied here, not only at prepare-data time: the cache
    holds the full split, so a run configured for a subset would otherwise
    silently train on everything. The subset is sampled rather than sliced
    because the cache is in corpus order, and the first N rows are dominated
    by a handful of repos.
    """
    examples: list[CodeExample] = load_prepared("train", data_config)

    if data_config.max_train is not None and data_config.max_train < len(examples):
        rng = random.Random(data_config.seed)
        examples = rng.sample(examples, data_config.max_train)

    return [(ex.query, format_code_for_encoding(ex)) for ex in examples]


def train_encoder(
    retrieval_config: RetrievalConfig,
    data_config: DataConfig,
    run_name: str | None = None,
) -> CodeSearchEncoder:
    """Fine-tune the base encoder and write it to `retrieval_config.output_dir`.

    `epochs: 0` is the documented way to evaluate the base encoder untouched:
    it saves the model without training so the baseline run flows through the
    identical load/save path as a fine-tuned one.
    """
    if retrieval_config.loss not in _LOSSES:
        raise ValueError(
            f"Unsupported loss {retrieval_config.loss!r}. Supported: {sorted(_LOSSES)}"
        )

    encoder = CodeSearchEncoder.from_config(retrieval_config)
    output_dir = Path(retrieval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if retrieval_config.epochs == 0:
        encoder.save(str(output_dir))
        return encoder

    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, losses
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    pairs = build_training_pairs(data_config)
    if not pairs:
        raise ValueError("No training pairs. Run `csne prepare-data` first.")

    # Column names are arbitrary but order matters: MNRL treats the first
    # column as anchors and the second as positives.
    dataset = Dataset.from_dict(
        {
            "query": [q for q, _ in pairs],
            "code": [c for _, c in pairs],
        }
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=retrieval_config.epochs,
        per_device_train_batch_size=retrieval_config.batch_size,
        learning_rate=retrieval_config.learning_rate,
        warmup_ratio=retrieval_config.warmup_ratio,
        run_name=run_name or "csne-retrieval",
        report_to=[],
        # In-batch negatives: a partial final batch has fewer negatives than
        # the rest, which makes its loss scale differently from every other step.
        dataloader_drop_last=True,
    )

    trainer = SentenceTransformerTrainer(
        model=encoder.model,
        args=args,
        train_dataset=dataset,
        loss=getattr(losses, retrieval_config.loss)(encoder.model),
    )
    trainer.train()

    encoder.save(str(output_dir))
    return encoder
