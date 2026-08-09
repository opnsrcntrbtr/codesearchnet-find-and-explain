"""Local Hugging Face summarizer (seq2seq, fine-tunable).

Requires the `hf` extra. Free and fully reproducible — no API key, no network
at inference — which makes it the backend of record for the report's
reproducibility claims.
"""

from __future__ import annotations

from csne.config import DataConfig, SummarizationConfig
from csne.data.loader import CodeExample
from csne.summarization.base import Summarizer, load_prompt_template, render_prompt

# flan-t5-base, not a CodeT5 checkpoint: CodeT5's own tokenizer files are
# incompatible with current `transformers` (a slow-tokenizer AddedToken bug
# in the Roberta-BPE bridge, unrelated to this project). flan-t5-base is
# instruction-tuned, so — unlike a raw code-to-comment seq2seq model — it can
# follow the same natural-language prompt as the Anthropic backend, keeping
# both backends on literally identical inputs (see `base.py`). flan-t5-small
# was tried first: at 80M params it cannot follow this prompt at all and
# degenerates to repeated <unk> tokens; flan-t5-base (250M) produces
# coherent, if weak, output — the expected shape for a zero-shot local
# baseline (fine-tuning is the improvement path, see `finetune_summarizer`).
DEFAULT_MODEL = "google/flan-t5-base"


def _pick_device() -> str:
    """Prefer GPU when available: MPS on Apple Silicon, else CUDA, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class HFSummarizer(Summarizer):
    """Generates function summaries with a local seq2seq model."""

    def __init__(self, config: SummarizationConfig) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.config = config
        self.model_name = config.model or DEFAULT_MODEL
        self.device = _pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.template = load_prompt_template(config.prompt_template)

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    def _generate(self, prompts: list[str]) -> list[str]:
        import torch

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_output_tokens,
                num_beams=4,
                early_stopping=True,
            )

        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids
        ]

    def summarize(self, example: CodeExample) -> str:
        return self._generate([render_prompt(self.template, example)])[0]

    def summarize_batch(self, examples: list[CodeExample]) -> list[str]:
        """True batched generation on GPU."""
        if not examples:
            return []
        return self._generate([render_prompt(self.template, ex) for ex in examples])


def finetune_summarizer(
    summarization_config: SummarizationConfig,
    data_config: DataConfig,
    run_name: str | None = None,
) -> HFSummarizer:
    """Fine-tune the local model on (code, target) pairs.

    Training targets:
      - "cleaned_docstring" (default): code → cleaned docstring from
        the CodeSearchNet query column.
      - "reference_summary": code → curated one-line summary from
        Nan-Do/code-search-net-python.

    Saves the checkpoint to `checkpoints/summarization/<run_name>`
    (or `checkpoints/summarization` if run_name is None) and returns
    an HFSummarizer pointing at that checkpoint.
    """
    import random

    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from csne.data.loader import CodeExample, load_prepared
    from csne.summarization.base import render_prompt

    model_name = summarization_config.model or DEFAULT_MODEL
    device = _pick_device()

    # Load training data.
    examples: list[CodeExample] = load_prepared("train", data_config)

    if data_config.max_train is not None and data_config.max_train < len(examples):
        rng = random.Random(data_config.seed)
        examples = rng.sample(examples, data_config.max_train)

    if not examples:
        raise ValueError("No training examples. Run `csne prepare-data` first.")

    # Build (prompt, target) pairs.
    prompts: list[str] = []
    targets: list[str] = []

    for ex in examples:
        prompt = render_prompt(summarization_config.prompt_template, ex)

        if summarization_config.training_target == "reference_summary":
            target = ex.reference_summary
        else:
            # cleaned_docstring — the query column is the cleaned docstring.
            target = ex.query

        if not target or not target.strip():
            continue  # skip examples without a valid target

        prompts.append(prompt)
        targets.append(target.strip())

    if not prompts:
        raise ValueError(
            "No training examples with a valid target after filtering."
        )

    # Tokenize.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.model_max_length = 512

    tokenized_inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=512,
    )

    tokenized_targets = tokenizer(
        targets,
        padding=True,
        truncation=True,
        max_length=64,  # summaries are short
    )

    # Replace -100 in labels with the input ids (standard seq2seq convention).
    tokenized_targets["input_ids"] = [
        [(t if t != -100 else tokenizer.pad_token_id) for t in row]
        for row in tokenized_targets["input_ids"]
    ]

    dataset = Dataset.from_dict(
        {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_targets["input_ids"],
        }
    )

    # Training args — similar shape to the retrieval fine-tune.
    output_dir = f"checkpoints/summarization/{run_name or 'default'}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2.0e-5,
        num_train_epochs=3,
        warmup_ratio=0.1,
        fp16=device != "cpu",  # use mixed precision on GPU/MPS
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        report_to="none",  # no wandb/tensorboard in this project
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    # Return a summarizer pointing at the fine-tuned checkpoint.
    ft_config = SummarizationConfig(
        backend="hf",
        model=output_dir,
        max_output_tokens=summarization_config.max_output_tokens,
        prompt_template=summarization_config.prompt_template,
    )
    return HFSummarizer(ft_config)
