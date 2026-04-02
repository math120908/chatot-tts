"""Tests for chatot.protocol module."""

from __future__ import annotations

import json

import pytest

from chatot.protocol import (
    Request,
    Response,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)


class TestEncodeRequest:
    """Test encode_request."""

    def test_produces_valid_json_with_newline(self) -> None:
        req = Request(text="hello")
        raw = encode_request(req)
        assert raw.endswith(b"\n")
        obj = json.loads(raw)
        assert obj["text"] == "hello"

    def test_lang_included_when_set(self) -> None:
        req = Request(text="test", lang="zh")
        obj = json.loads(encode_request(req))
        assert obj["lang"] == "zh"

    def test_lang_null_when_unset(self) -> None:
        req = Request(text="test")
        obj = json.loads(encode_request(req))
        assert obj["lang"] is None


class TestDecodeRequest:
    """Test decode_request."""

    def test_round_trip(self) -> None:
        original = Request(text="round trip test")
        raw = encode_request(original)
        decoded = decode_request(raw)
        assert decoded.text == original.text
        assert decoded.lang == original.lang

    def test_with_lang(self) -> None:
        original = Request(text="hello", lang="en")
        raw = encode_request(original)
        decoded = decode_request(raw)
        assert decoded.lang == "en"

    def test_raises_on_empty_text(self) -> None:
        data = json.dumps({"text": ""}).encode("utf-8")
        with pytest.raises(ValueError, match="non-empty"):
            decode_request(data)

    def test_raises_on_whitespace_only_text(self) -> None:
        data = json.dumps({"text": "   \t\n  "}).encode("utf-8")
        with pytest.raises(ValueError, match="non-empty"):
            decode_request(data)

    def test_raises_on_missing_text(self) -> None:
        data = json.dumps({"lang": "en"}).encode("utf-8")
        with pytest.raises(ValueError, match="non-empty"):
            decode_request(data)


class TestResponse:
    """Test encode_response / decode_response."""

    def test_ok_round_trip(self) -> None:
        original = Response(status="ok")
        raw = encode_response(original)
        decoded = decode_response(raw)
        assert decoded.status == "ok"
        assert decoded.message is None

    def test_error_with_message(self) -> None:
        original = Response(status="error", message="something went wrong")
        raw = encode_response(original)
        decoded = decode_response(raw)
        assert decoded.status == "error"
        assert decoded.message == "something went wrong"

    def test_produces_valid_json_with_newline(self) -> None:
        resp = Response(status="ok", message="done")
        raw = encode_response(resp)
        assert raw.endswith(b"\n")
        obj = json.loads(raw)
        assert obj["status"] == "ok"


class TestUTF8:
    """Test UTF-8 encoding for CJK text."""

    def test_chinese_text_request(self) -> None:
        req = Request(text="你好世界")
        raw = encode_request(req)
        decoded = decode_request(raw)
        assert decoded.text == "你好世界"

    def test_chinese_text_in_response_message(self) -> None:
        resp = Response(status="error", message="語音合成失敗")
        raw = encode_response(resp)
        decoded = decode_response(raw)
        assert decoded.message == "語音合成失敗"

    def test_ensure_ascii_false(self) -> None:
        req = Request(text="日本語テスト")
        raw = encode_request(req)
        # ensure_ascii=False means the actual characters appear, not \\uXXXX
        assert "日本語テスト".encode("utf-8") in raw
