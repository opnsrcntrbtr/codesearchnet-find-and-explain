"""Typed config objects loaded from the YAML files in `configs/`.

Every experiment is defined by a config file so runs are reproducible and the
exact settings can be recorded alongside results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Where the corpus comes from and how it is filtered/split."""

    retrieval_dataset: str = "code-search-net/code_search_net"
    retrieval_config: str = "python"
    summary_dataset: str = "Nan-Do/code-search-net-python"
    language: str = "python"
    max_train: int | None = None
    max_eval: int | None = 1000
    min_docstring_tokens: int = 3
    max_code_tokens: int = 512
    deduplicate: bool = True
    cache_dir: str = "data/cache"
    seed: int = 42


@dataclass
class RetrievalConfig:
    """Base encoder plus fine-tuning hyperparameters."""

    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    loss: str = "MultipleNegativesRankingLoss"
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 256
    output_dir: str = "checkpoints/retrieval"


@dataclass
class SummarizationConfig:
    """Which summarizer backend to use and how to drive it."""

    backend: str = "anthropic"  # "anthropic" | "hf"
    # None defers to each backend's own DEFAULT_MODEL — a single default here
    # would be wrong for whichever backend isn't "anthropic" (an HF backend
    # fed "claude-sonnet-5" tries to pull that as a Hub repo id and fails).
    model: str | None = None
    max_output_tokens: int = 160
    temperature: float = 0.0
    prompt_template: str = "prompts/function_summary.txt"


@dataclass
class EvalConfig:
    """Metrics and evaluation protocol."""

    retrieval_k: list[int] = field(default_factory=lambda: [1, 5, 10])
    distractor_pool_size: int = 999
    summarization_metrics: list[str] = field(default_factory=lambda: ["bleu", "bertscore"])
    results_dir: str = "results"


@dataclass
class ExperimentConfig:
    """Top-level config: one of these fully describes a run."""

    name: str
    data: DataConfig
    retrieval: RetrievalConfig
    summarization: SummarizationConfig
    evaluation: EvalConfig


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "retrieval": RetrievalConfig,
    "summarization": SummarizationConfig,
    "evaluation": EvalConfig,
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a single YAML file into a plain dict."""
    with open(path) as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(loaded).__name__}")
    return loaded


def _build_section(cls: type, values: dict[str, Any], source: str) -> Any:
    """Instantiate a config dataclass, rejecting unknown keys.

    Unknown keys are an error rather than a warning: a typo'd hyperparameter
    that silently does nothing would invalidate an experiment without any
    visible signal.
    """
    if not is_dataclass(cls):  # pragma: no cover - guards a programming error
        raise TypeError(f"{cls!r} is not a config dataclass")
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"{source}: unknown {cls.__name__} key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )
    return cls(**values)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Build an `ExperimentConfig` from a YAML file.

    Each section may either be inlined or given as a path (relative to the
    experiment file) to a shared config in `configs/`, so common data/eval
    settings are written once. An `overrides` block patches individual keys
    after the shared files load, which is how experiments differ from each
    other without duplicating whole files.
    """
    path = Path(path)
    raw = load_yaml(path)

    if "name" not in raw:
        raise ValueError(f"{path}: experiment config must define a 'name'")

    overrides = raw.get("overrides") or {}
    unknown_sections = set(overrides) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(
            f"{path}: unknown override section(s): {', '.join(sorted(unknown_sections))}"
        )

    sections: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        spec = raw.get(key, {})
        if isinstance(spec, str):
            ref = (path.parent / spec).resolve()
            values, source = load_yaml(ref), str(ref)
        elif isinstance(spec, dict):
            values, source = dict(spec), str(path)
        else:
            raise ValueError(f"{path}: section '{key}' must be a mapping or a path string")

        values.update(overrides.get(key, {}))
        sections[key] = _build_section(cls, values, source)

    return ExperimentConfig(name=raw["name"], **sections)
