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
    "CodeExample",
    "load_raw",
    "load_prepared",
    "load_summary_map",
    "save_split",
    "clean_docstring",
    "normalize_code",
    "filter_examples",
    "SPLITS",
    "make_splits",
    "prepare_official_splits",
    "save_splits",
]
