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
    """Fine-tune the local model on (code, cleaned docstring) pairs.

    Optional: the zero-shot backend is the baseline, and this is the
    fine-tuned comparison point in the report's summarization experiments.
    """
    raise NotImplementedError
