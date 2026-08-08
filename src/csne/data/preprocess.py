"""Cleaning and filtering applied before splitting.

Preprocessing choices are part of the experimental setup: the same functions
must run over train and eval data, or retrieval scores are not comparable.
"""

from __future__ import annotations

import ast
import hashlib
import re
import warnings
from collections.abc import Iterable, Iterator

from csne.config import DataConfig
from csne.data.loader import CodeExample

# A docstring line that starts a structured section: everything from here on
# is API reference, not the description a searcher would type.
_SECTION_RE = re.compile(
    r"""^\s*(
        [:@](param|parameter|arg|argument|type|returns?|rtype|raises?|except|
             exception|yield|yields|ytype|var|ivar|cvar|note|warning|see|
             seealso|deprecated|example|todo)\b
        |(Args|Arguments|Params|Parameters|Returns?|Yields?|Raises?|Notes?|
          Examples?|Attributes|Warnings?|See\s+Also|References|Todo)\s*:\s*$
        |-{3,}\s*$
        |={3,}\s*$
        |>>>
        |\.\.\s+\w+::
    )""",
    re.VERBOSE | re.IGNORECASE,
)

# Docstrings that describe nothing. Cheap to list, and they otherwise become
# many identical "queries" pointing at unrelated functions.
_BOILERPLATE = {
    "todo",
    "tbd",
    "n/a",
    "docstring",
    "constructor",
    "init",
    "initialize",
    "initialise",
    "getter",
    "setter",
    "helper",
    "internal",
    "private",
    "deprecated",
    "not implemented",
    "see above",
    "see below",
}

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_docstring(docstring: str) -> str:
    """Reduce a raw docstring to the query-like leading description.

    Stops at the first structured section (reST `:param:`, Google-style
    `Args:`, doctests, underlines) — a developer searching for code types the
    summary line, not a full API reference.
    """
    if not docstring:
        return ""

    kept: list[str] = []
    for line in docstring.strip().splitlines():
        if _SECTION_RE.match(line):
            break
        kept.append(line)

    text = " ".join(kept)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    # Trailing colons are left over from a heading whose body we just cut.
    return text.rstrip(":").strip()


def normalize_code(code: str) -> str:
    """Normalize a function body for embedding.

    Strips the docstring: it is the retrieval target, and leaving it in the
    code makes the task trivially lexical and inflates every metric.
    Identifiers are left intact — they carry most of the usable signal.
    """
    if not code:
        return ""

    stripped = _strip_docstring_ast(code)
    if stripped is None:
        stripped = _strip_docstring_regex(code)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _strip_docstring_ast(code: str) -> str | None:
    """Remove the docstring via the AST. Returns None if `code` will not parse.

    CodeSearchNet stores bare function bodies at their original indentation,
    so many snippets are not parseable on their own; callers fall back.
    """
    try:
        # Real corpus code is full of regex literals with unescaped
        # backslashes; each one emits a SyntaxWarning that would bury the
        # actual output of a 400K-example run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            # Replace rather than delete: a function whose entire body is a
            # docstring would otherwise become syntactically invalid.
            node.body[0] = ast.Pass()

    return ast.unparse(tree)


def _strip_docstring_regex(code: str) -> str:
    """Fallback: drop the first triple-quoted string in an unparseable snippet."""
    match = re.search(r'("""|\'\'\')', code)
    if not match:
        return code
    quote = match.group(1)
    end = code.find(quote, match.end())
    if end == -1:
        return code
    return code[: match.start()] + code[end + len(quote) :]


def _is_boilerplate(text: str) -> bool:
    return text.strip().strip(".").lower() in _BOILERPLATE


def _looks_non_english(text: str) -> bool:
    """Heuristic: reject docstrings that are mostly non-ASCII.

    The corpus contains CJK and Cyrillic docstrings. They are valid data but
    the encoder is English-trained, so they add noise to both training and
    the eval distractor pool.
    """
    if not text:
        return False
    return len(_NON_ASCII_RE.findall(text)) / len(text) > 0.25


def code_fingerprint(code: str) -> str:
    """Whitespace-insensitive hash used to detect near-duplicate functions."""
    return hashlib.sha1(_WHITESPACE_RE.sub("", code).encode()).hexdigest()


def filter_examples(
    examples: Iterable[CodeExample], config: DataConfig
) -> Iterator[CodeExample]:
    """Clean each example and drop the ones that cannot serve as supervision.

    Yields cleaned copies, so callers get exactly the text the model will see.
    Deduplication is by normalized-code fingerprint: CodeSearchNet contains
    the same vendored function copied across many repos, and those copies
    would otherwise land on both sides of a split.
    """
    seen: set[str] = set()

    for ex in examples:
        docstring = clean_docstring(ex.docstring)
        if len(docstring.split()) < config.min_docstring_tokens:
            continue
        if _is_boilerplate(docstring) or _looks_non_english(docstring):
            continue

        code = normalize_code(ex.code)
        if not code or len(code.split()) > config.max_code_tokens:
            continue

        if config.deduplicate:
            fingerprint = code_fingerprint(code)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

        yield CodeExample(
            id=ex.id,
            code=code,
            docstring=docstring,
            func_name=ex.func_name,
            repo=ex.repo,
            path=ex.path,
            language=ex.language,
            summary=ex.summary,
        )
