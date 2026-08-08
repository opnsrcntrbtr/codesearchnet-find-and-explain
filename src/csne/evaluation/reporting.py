"""Experiment logging: append every run to CSV under `results/`.

Results are appended, never overwritten, and each row carries the config hash
and git commit so a number in the report can be traced back to the exact run
that produced it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from csne.config import ExperimentConfig

RETRIEVAL_CSV = "retrieval_results.csv"
SUMMARIZATION_CSV = "summarization_results.csv"


def results_path(results_dir: str, filename: str) -> Path:
    """Resolve a results file path, creating the directory if needed."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def config_hash(config: ExperimentConfig) -> str:
    """Short stable hash of a config, used to group runs of identical settings."""
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def git_commit() -> str | None:
    """Current commit SHA, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def append_result(results_dir: str, filename: str, row: dict[str, Any]) -> Path:
    """Append one result row, writing the header if the file is new.

    Runs with different metric sets would otherwise produce ragged CSVs, so a
    row whose columns do not match an existing header is padded against that
    header rather than corrupting the file.
    """
    path = results_path(results_dir, filename)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit() or "",
        **row,
    }

    existing_header: list[str] | None = None
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as fh:
            existing_header = next(csv.reader(fh), None)

    header = existing_header or list(row)
    aligned = {col: row.get(col, "") for col in header}
    # Anything the header does not know about would be dropped silently.
    for key in row:
        if key not in aligned:
            raise ValueError(
                f"{path}: column {key!r} is not in the existing header. "
                "Start a new results file rather than mixing schemas."
            )

    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if existing_header is None:
            writer.writeheader()
        writer.writerow(aligned)
    return path
