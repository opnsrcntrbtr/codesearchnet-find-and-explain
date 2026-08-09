"""Claude-backed summarizer (prompting, no fine-tuning).

Requires the `anthropic` extra and Anthropic credentials (`ANTHROPIC_API_KEY`
or an `ant auth login` profile — the SDK resolves either automatically).
Costs API tokens per function, so evaluation runs should be capped via
`EvalConfig`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from csne.config import SummarizationConfig
from csne.data.loader import CodeExample
from csne.summarization.base import Summarizer, load_prompt_template, render_prompt

DEFAULT_MODEL = "claude-sonnet-5"

# Concurrent request cap for summarize_batch. The SDK already retries
# 429/5xx with backoff; this just bounds how many requests are in flight at
# once so a large eval run doesn't open hundreds of connections at once.
_MAX_CONCURRENCY = 8


class AnthropicSummarizer(Summarizer):
    """Generates function summaries through the Anthropic Messages API."""

    def __init__(self, config: SummarizationConfig) -> None:
        import anthropic

        self.config = config
        self.model = config.model or DEFAULT_MODEL
        # Zero-arg client: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or
        # an `ant auth login` profile, in that order. Never hardcode a key.
        self.client = anthropic.Anthropic()
        self.template = load_prompt_template(config.prompt_template)

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def summarize(self, example: CodeExample) -> str:
        import anthropic

        prompt = render_prompt(self.template, example)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.config.max_output_tokens,
                # `temperature` is not passed: it's a removed parameter on
                # current-generation models (Opus 5/4.8/4.7, Sonnet 5) and a
                # non-default value 400s. Determinism isn't load-bearing for
                # a one-off summary; thinking is left at its adaptive default.
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"Anthropic rate limited on {example.id}: {e}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(
                f"Anthropic API error on {example.id} ({e.status_code}): {e.message}"
            ) from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"Anthropic connection error on {example.id}: {e}") from e

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"Anthropic declined to summarize {example.id} "
                f"(category={getattr(response.stop_details, 'category', None)})"
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        return text.strip()

    def summarize_batch(self, examples: list[CodeExample]) -> list[str]:
        """Concurrent single-message calls, order-preserving.

        The SDK's built-in retry (default `max_retries=2`) already handles
        transient 429/5xx per call; this only bounds how many requests are
        in flight simultaneously.
        """
        if not examples:
            return []
        with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENCY, len(examples))) as pool:
            return list(pool.map(self.summarize, examples))
