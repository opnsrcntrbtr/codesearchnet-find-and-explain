"""Data layer: load CodeSearchNet-style Python data, clean it, split it."""

from csne.data.loader import (
    CodeExample,
    load_prepared,
    load_raw,
    load_summary_map,
    save_split,
)
from csne.data.preprocess import clean_docstring, filter_examples, normalize_code
from csne.data.splits import SPLITS, make_splits, prepare_official_splits, save_splits

__all__ = [
    "SPLITS",
    "CodeExample",
    "clean_docstring",
    "filter_examples",
    "load_prepared",
    "load_raw",
    "load_summary_map",
    "make_splits",
    "normalize_code",
    "prepare_official_splits",
    "save_split",
    "save_splits",
]
