"""Data layer tests: cleaning, filtering, and leak-free splitting."""

from __future__ import annotations

import pytest
from helpers import make_example

from csne.config import DataConfig
from csne.data.loader import load_prepared, save_split
from csne.data.preprocess import clean_docstring, filter_examples, normalize_code
from csne.data.splits import SPLITS, make_splits


def test_clean_docstring_strips_rest_param_sections():
    """reST `:param:` blocks are dropped, the summary line survives."""
    raw = (
        "Open a database connection.\n\n"
        ":param host: the host\n"
        ":returns: a connection\n"
    )
    assert clean_docstring(raw) == "Open a database connection."


def test_clean_docstring_strips_google_sections():
    """Google-style `Args:`/`Returns:` headings are dropped."""
    raw = "Retry a callable.\n\nArgs:\n    fn: the callable\nReturns:\n    result\n"
    assert clean_docstring(raw) == "Retry a callable."


def test_clean_docstring_strips_doctests():
    """A doctest block is reference material, not a query."""
    raw = "Add two numbers.\n\n>>> add(1, 2)\n3\n"
    assert clean_docstring(raw) == "Add two numbers."


def test_clean_docstring_collapses_multiline_summary():
    """A summary wrapped across lines becomes one line."""
    raw = "Convert a string\ninto a URL-safe slug."
    assert clean_docstring(raw) == "Convert a string into a URL-safe slug."


def test_clean_docstring_handles_empty():
    assert clean_docstring("") == ""


def test_normalize_code_removes_docstring():
    """The docstring must not survive into the code embedding — it is the target."""
    code = 'def f(x):\n    """Add one to x."""\n    return x + 1'
    out = normalize_code(code)
    assert "Add one to x" not in out
    assert "return x + 1" in out


def test_normalize_code_keeps_docstring_only_body_valid():
    """A function whose whole body is a docstring stays syntactically valid."""
    out = normalize_code('def f():\n    """Just docs."""')
    assert "Just docs" not in out
    assert "def f()" in out


def test_normalize_code_survives_unparseable_snippet():
    """CSN stores indented bare bodies; those must not crash the cleaner."""
    out = normalize_code('    """Docs here."""\n    return 1')
    assert "Docs here" not in out
    assert "return 1" in out


def test_filter_drops_short_docstrings(data_config: DataConfig):
    """Docstrings under `min_docstring_tokens` are not usable as queries."""
    kept = list(filter_examples([make_example(docstring="Does.")], data_config))
    assert kept == []


def test_filter_drops_boilerplate(data_config: DataConfig):
    """`TODO`-style docstrings describe nothing and collide across functions."""
    kept = list(filter_examples([make_example(docstring="TODO")], data_config))
    assert kept == []


def test_filter_drops_non_english(data_config: DataConfig):
    """The encoder is English-trained; CJK docstrings only add noise."""
    kept = list(filter_examples([make_example(docstring="将字符串转换为安全的网址片段")], data_config))
    assert kept == []


def test_filter_deduplicates_by_normalized_code(data_config: DataConfig):
    """Vendored copies of one function must not span both sides of a split."""
    a = make_example(id="a", repo="one/x", code="def f(x):\n    return x + 1")
    b = make_example(id="b", repo="two/y", code="def f(x):\n      return x  +  1")
    kept = list(filter_examples([a, b], data_config))
    assert len(kept) == 1


def test_filter_keeps_and_cleans_good_examples(data_config: DataConfig, sample_examples):
    """Survivors come back cleaned, not raw."""
    kept = list(filter_examples(sample_examples, data_config))
    ids = {ex.id for ex in kept}
    assert ids == {"u1", "u2", "u3"}
    assert all(":param" not in ex.docstring for ex in kept)
    assert all("Args:" not in ex.docstring for ex in kept)


def test_filter_respects_max_code_tokens():
    """Over-long functions blow the encoder's context and are dropped."""
    config = DataConfig(max_code_tokens=5)
    long_fn = make_example(code="def f():\n    " + " + ".join(str(i) for i in range(50)))
    assert list(filter_examples([long_fn], config)) == []


def test_splits_are_disjoint_by_repo(data_config: DataConfig):
    """No repo appears in more than one split, or eval leaks train code."""
    examples = [
        make_example(id=f"{repo}-{i}", repo=repo)
        for repo in (f"org/repo{n}" for n in range(60))
        for i in range(3)
    ]
    splits = make_splits(examples, data_config)
    repos = [{ex.repo for ex in splits[s]} for s in SPLITS]
    assert repos[0] & repos[1] == set()
    assert repos[0] & repos[2] == set()
    assert repos[1] & repos[2] == set()


def test_splits_cover_every_example(data_config: DataConfig):
    """Nothing is silently dropped by splitting."""
    examples = [make_example(id=str(i), repo=f"org/repo{i}") for i in range(50)]
    splits = make_splits(examples, data_config)
    assert sum(len(splits[s]) for s in SPLITS) == len(examples)


def test_splits_are_deterministic_given_seed(data_config: DataConfig):
    """Same seed, same corpus, same partition."""
    examples = [make_example(id=str(i), repo=f"org/repo{i}") for i in range(50)]
    first = make_splits(examples, data_config)
    second = make_splits(examples, data_config)
    assert {s: [e.id for e in first[s]] for s in SPLITS} == {
        s: [e.id for e in second[s]] for s in SPLITS
    }


def test_splits_differ_across_seeds():
    """A different seed really does repartition."""
    examples = [make_example(id=str(i), repo=f"org/repo{i}") for i in range(100)]
    a = make_splits(examples, DataConfig(seed=1))
    b = make_splits(examples, DataConfig(seed=2))
    assert [e.id for e in a["train"]] != [e.id for e in b["train"]]


def test_prepared_split_roundtrips(tmp_path, sample_examples):
    """A cached split reloads to exactly what was written."""
    config = DataConfig(cache_dir=str(tmp_path))
    save_split("train", sample_examples, config)
    assert load_prepared("train", config) == sample_examples


def test_load_prepared_raises_when_cache_missing(tmp_path):
    """Silently training on a missing cache would be worse than failing."""
    with pytest.raises(FileNotFoundError):
        load_prepared("train", DataConfig(cache_dir=str(tmp_path / "nope")))


def test_reference_summary_prefers_summary_over_docstring():
    """The curated summary is the scoring target when the join supplied one."""
    ex = make_example(docstring="Long noisy docstring.", summary="Add one.")
    assert ex.reference_summary == "Add one."


def test_reference_summary_falls_back_to_docstring():
    """A partial join degrades to docstrings rather than losing the example."""
    assert make_example(summary=None).reference_summary == "Add one to a number."


def test_query_never_uses_generated_summary():
    """Using the generated summary as a query would leak it into retrieval scores."""
    ex = make_example(docstring="Add one to a number.", summary="Increment.")
    assert ex.query == "Add one to a number."
