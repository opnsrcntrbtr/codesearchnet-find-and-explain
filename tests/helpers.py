"""Test helpers shared across modules."""

from __future__ import annotations

from csne.data.loader import CodeExample


def make_example(**kwargs) -> CodeExample:
    """A `CodeExample` with sensible defaults; override only what matters."""
    base = {
        "id": "https://github.com/acme/lib/blob/abc123/lib/util.py#L1-L10",
        "code": "def f(x):\n    return x + 1",
        "docstring": "Add one to a number.",
        "func_name": "f",
        "repo": "acme/lib",
        "path": "lib/util.py",
        "language": "python",
    }
    base.update(kwargs)
    return CodeExample(**base)
