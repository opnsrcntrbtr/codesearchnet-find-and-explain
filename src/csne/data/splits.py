"""Train/validation/test splitting.

The official CodeSearchNet splits are used as-is. They are already partitioned
by repository, so no function from a training repo appears in the eval index,
and using them keeps our numbers comparable to published CSN baselines.

`make_splits` exists for corpora that arrive without official splits; it
groups by repository for the same reason — two functions from one repo are
often near-identical, and splitting per-example leaks train code into the
test index and inflates retrieval scores.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from csne.config import DataConfig
from csne.data.loader import CodeExample, load_raw, load_summary_map, save_split
from csne.data.preprocess import filter_examples

SPLITS = ("train", "valid", "test")
_DEFAULT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def _repo_bucket(repo: str, seed: int) -> float:
    """Map a repo name to a stable value in [0, 1).

    Hashing rather than shuffling keeps the assignment deterministic across
    runs and independent of corpus order or size, so adding data does not
    reshuffle everything that came before.
    """
    digest = hashlib.sha1(f"{seed}:{repo}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def make_splits(
    examples: Iterable[CodeExample],
    config: DataConfig,
    ratios: dict[str, float] | None = None,
) -> dict[str, list[CodeExample]]:
    """Partition examples into train/valid/test, grouped by repo.

    Deterministic given `config.seed`: the same corpus and seed reproduce the
    same partition.
    """
    ratios = ratios or _DEFAULT_RATIOS
    total = sum(ratios[s] for s in SPLITS)
    train_end = ratios["train"] / total
    valid_end = train_end + ratios["valid"] / total

    out: dict[str, list[CodeExample]] = {s: [] for s in SPLITS}
    for ex in examples:
        bucket = _repo_bucket(ex.repo, config.seed)
        split = "train" if bucket < train_end else "valid" if bucket < valid_end else "test"
        out[split].append(ex)
    return out


def prepare_official_splits(config: DataConfig) -> dict[str, list[CodeExample]]:
    """Load, clean, and cache the official CodeSearchNet splits.

    Summaries are joined in from the annotated dataset; a function with no
    match keeps `summary=None` and falls back to its docstring at scoring
    time, so a partial join degrades rather than fails.

    `max_train` is deliberately NOT applied here, even though `config` carries
    it: the cache must hold the full train split, because `max_train` is
    applied later by `build_training_pairs` via a repo-independent *sample*.
    Slicing here instead would cache a corpus-order-biased prefix — the
    corpus is dominated by a handful of large repos at the front — and no
    later sampling could undo that. `max_eval` truncates valid/test at cache
    time because those are used whole, capped only by `distractor_pool_size`
    at evaluation time, not resampled.
    """
    summaries = load_summary_map(config)

    prepared: dict[str, list[CodeExample]] = {}
    for split in SPLITS:
        limit = config.max_eval if split != "train" else None
        raw = load_raw(config, split=split, summaries=summaries)
        examples = list(filter_examples(raw, config))
        if limit is not None:
            examples = examples[:limit]
        prepared[split] = examples
        save_split(split, examples, config)
    return prepared


def save_splits(splits: dict[str, list[CodeExample]], config: DataConfig) -> None:
    """Persist splits under `config.cache_dir` as JSONL, one file per split."""
    for split, examples in splits.items():
        save_split(split, examples, config)
