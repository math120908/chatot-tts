"""Tests for chatot.client module."""

from __future__ import annotations

import io
import sys
from unittest import mock

import pytest

from chatot.client import _parse_args, _resolve_text


class TestParseArgs:
    """Test argument parsing."""

    def test_positional_text(self) -> None:
        args = _parse_args(["hello world"])
        assert args.text == "hello world"

    def test_no_text_is_none(self) -> None:
        args = _parse_args([])
        assert args.text is None

    def test_lang_option(self) -> None:
        args = _parse_args(["hello", "--lang", "zh"])
        assert args.text == "hello"
        assert args.lang == "zh"

    def test_socket_option(self) -> None:
        args = _parse_args(["hello", "--socket", "/tmp/test.sock"])
        assert args.socket == "/tmp/test.sock"

    def test_defaults_are_none(self) -> None:
        args = _parse_args(["test"])
        assert args.lang is None
        assert args.socket is None


class TestResolveText:
    """Test input resolution from args and stdin."""

    def test_positional_arg(self) -> None:
        args = _parse_args(["hello from arg"])
        text = _resolve_text(args)
        assert text == "hello from arg"

    def test_stdin_reading(self) -> None:
        args = _parse_args([])
        fake_stdin = io.StringIO("hello from stdin")
        with mock.patch.object(sys, "stdin", fake_stdin):
            text = _resolve_text(args)
        assert text == "hello from stdin"

    def test_positional_takes_priority_over_stdin(self) -> None:
        args = _parse_args(["from arg"])
        fake_stdin = io.StringIO("from stdin")
        with mock.patch.object(sys, "stdin", fake_stdin):
            text = _resolve_text(args)
        assert text == "from arg"

    def test_none_when_no_input_and_tty(self) -> None:
        args = _parse_args([])
        # Simulate a TTY (isatty returns True, so stdin is not read)
        fake_stdin = io.StringIO("should not be read")
        fake_stdin.isatty = lambda: True  # type: ignore[assignment]
        with mock.patch.object(sys, "stdin", fake_stdin):
            text = _resolve_text(args)
        assert text is None

    def test_none_when_stdin_is_empty(self) -> None:
        args = _parse_args([])
        fake_stdin = io.StringIO("")
        with mock.patch.object(sys, "stdin", fake_stdin):
            text = _resolve_text(args)
        assert text is None
