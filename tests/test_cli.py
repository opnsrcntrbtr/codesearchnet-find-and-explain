"""CLI wiring tests.

`explain` was added to the argparse subparsers but never registered in
`_COMMANDS` — invisible until someone actually ran `csne explain` and hit a
`KeyError`. This test exists so that class of bug can't recur silently.
"""

from __future__ import annotations

import pytest

from csne.cli import _COMMANDS, build_parser


def test_subparsers_and_command_handlers_match_exactly():
    """Every subcommand argparse exposes must be runnable, and vice versa —
    `explain` was added to one side of this and not the other."""
    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if a.dest == "command"
    )
    declared = set(subparsers_action.choices.keys())
    assert declared == set(_COMMANDS.keys())


def test_config_is_required_on_every_subcommand():
    parser = build_parser()
    for name in _COMMANDS:
        with pytest.raises(SystemExit):
            parser.parse_args([name])


def test_explain_requires_query():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["explain", "--config", "x.yaml"])


def test_explain_parses_with_defaults():
    parser = build_parser()
    args = parser.parse_args(["explain", "--config", "x.yaml", "--query", "parse json"])
    assert args.command == "explain"
    assert args.query == "parse json"
    assert args.k == 5


def test_evaluate_summarize_flag_defaults_false():
    parser = build_parser()
    args = parser.parse_args(["evaluate", "--config", "x.yaml"])
    assert args.summarize is False


def test_evaluate_summarize_flag_can_be_set():
    parser = build_parser()
    args = parser.parse_args(["evaluate", "--config", "x.yaml", "--summarize"])
    assert args.summarize is True
