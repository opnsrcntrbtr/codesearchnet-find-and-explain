"""Pull the raw corpus from Hugging Face and normalize it to `CodeExample`.

Two datasets back this project and they disagree on column names, so all
downstream code works against `CodeExample` rather than raw dataset rows:

    code-search-net/code_search_net  -> retrieval (official splits)
    Nan-Do/code-search-net-python    -> `summary` column for summarization

They are joined on the GitHub permalink, which pins repo + commit sha + line
range and is therefore unique per function. Joining on repo+path+func_name
would collide across same-named methods defined in one file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from csne.config import DataConfig

# Column names differ between the two mirrors; map both onto CodeExample.
_RETRIEVAL_COLUMNS = {
    "url": "func_code_url",
    "code": "func_code_string",
    "docstring": "func_documentation_string",
    "func_name": "func_name",
    "repo": "repository_name",
    "path": "func_path_in_repository",
    "language": "language",
}

SPLIT_ALIASES = {"valid": "validation", "validation": "validation"}


@dataclass(frozen=True)
class CodeExample:
    """One (query, code) pair drawn from a documented function."""

    id: str
    code: str
    docstring: str
    func_name: str
    repo: str
    path: str
    language: str = "python"
    summary: str | None = None

    @property
    def query(self) -> str:
        """The natural-language side of the pair, as a search query would read.

        The cleaned docstring, not the summary: the summary column is a
        generated target for the summarization task, and using it as a
        retrieval query would leak that model's output into retrieval scores.
        """
        return self.docstring

    @property
    def reference_summary(self) -> str:
        """The target for summarization scoring.

        Prefers the curated one-line `summary` when the join supplied one and
        falls back to the docstring, which is noisier but always present.
        """
        return self.summary if self.summary else self.docstring


def _row_to_example(row: dict, summary: str | None = None) -> CodeExample:
    """Map one raw retrieval-dataset row onto a `CodeExample`."""
    return CodeExample(
        id=row[_RETRIEVAL_COLUMNS["url"]],
        code=row[_RETRIEVAL_COLUMNS["code"]],
        docstring=row[_RETRIEVAL_COLUMNS["docstring"]],
        func_name=row[_RETRIEVAL_COLUMNS["func_name"]],
        repo=row[_RETRIEVAL_COLUMNS["repo"]],
        path=row[_RETRIEVAL_COLUMNS["path"]],
        language=row[_RETRIEVAL_COLUMNS["language"]],
        summary=summary,
    )


def load_summary_map(
    config: DataConfig, limit: int | None = None, use_cache: bool = True
) -> dict[str, str]:
    """Build `{permalink: summary}` from the summary-annotated dataset.

    Cached to disk on first build: streaming all ~455K rows takes minutes over
    the network, and every `prepare-data` run would otherwise pay it again.
    Held in memory once built — 455K short strings is cheap next to re-scanning
    the dataset once per split.
    """
    from datasets import load_dataset

    cache = Path(config.cache_dir) / "summary_map.json"
    if use_cache and limit is None and cache.exists():
        with open(cache) as fh:
            return json.load(fh)

    stream = load_dataset(config.summary_dataset, split="train", streaming=True)
    mapping: dict[str, str] = {}
    for i, row in enumerate(stream):
        if limit is not None and i >= limit:
            break
        summary = (row.get("summary") or "").strip()
        if summary:
            mapping[row["url"]] = summary

    if use_cache and limit is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(mapping, fh)
    return mapping


def load_raw(
    config: DataConfig,
    split: str = "train",
    summaries: dict[str, str] | None = None,
    limit: int | None = None,
) -> Iterator[CodeExample]:
    """Stream one official CodeSearchNet split as `CodeExample`s.

    Streams rather than materializing: the full corpus does not fit
    comfortably in a free-tier Colab runtime.

    Falls back to local JSONL cache if available (for smoke tests with
    small sample datasets pushed to the repo).
    """
    from pathlib import Path

    cache_file = Path(config.cache_dir) / f"{split}.jsonl"
    if cache_file.exists():
        # Load from local cache (small sample datasets)
        for line in open(cache_file):
            line = line.strip()
            if not line:
                continue
            yield CodeExample(**json.loads(line))
        return

    from datasets import load_dataset

    hf_split = SPLIT_ALIASES.get(split, split)
    stream = load_dataset(
        config.retrieval_dataset,
        config.retrieval_config,
        split=hf_split,
        streaming=True,
    )
    for i, row in enumerate(stream):
        if limit is not None and i >= limit:
            break
        if row[_RETRIEVAL_COLUMNS["language"]] != config.language:
            continue
        url = row[_RETRIEVAL_COLUMNS["url"]]
        yield _row_to_example(row, summary=(summaries or {}).get(url))


def _split_file(split: str, config: DataConfig) -> Path:
    return Path(config.cache_dir) / f"{split}.jsonl"


def save_split(split: str, examples: list[CodeExample], config: DataConfig) -> Path:
    """Write one prepared split to the cache as JSONL."""
    path = _split_file(split, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.writelines(json.dumps(asdict(ex)) + "\n" for ex in examples)
    return path


def load_prepared(split: str, config: DataConfig) -> list[CodeExample]:
    """Load an already preprocessed split from `config.cache_dir`.

    Raises if the cache is missing, so training never silently trains on a
    differently-preprocessed corpus than the one that was evaluated.
    """
    path = _split_file(split, config)
    if not path.exists():
        raise FileNotFoundError(
            f"No prepared split at {path}. Run `csne prepare-data` first."
        )
    with open(path) as fh:
        return [CodeExample(**json.loads(line)) for line in fh if line.strip()]
