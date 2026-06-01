"""Wire protocol for chatot daemon <-> client communication.

Uses newline-delimited JSON over Unix domain socket.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Request:
    """TTS request from client to daemon."""

    text: str
    lang: str | None = None
    voice: str | None = None
    save_path: str | None = None


@dataclass(frozen=True, slots=True)
class Response:
    """Response from daemon to client."""

    status: str  # "ok" or "error"
    message: str | None = None


def encode_request(req: Request) -> bytes:
    """Serialize a Request to JSON bytes with trailing newline."""
    return json.dumps(asdict(req), ensure_ascii=False).encode("utf-8") + b"\n"


def decode_request(data: bytes) -> Request:
    """Deserialize bytes to a Request. Raises ValueError if text is empty."""
    obj = json.loads(data.decode("utf-8"))
    text = obj.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Request text must be a non-empty string")
    return Request(
        text=text,
        lang=obj.get("lang"),
        voice=obj.get("voice"),
        save_path=obj.get("save_path"),
    )


def encode_response(resp: Response) -> bytes:
    """Serialize a Response to JSON bytes with trailing newline."""
    return json.dumps(asdict(resp), ensure_ascii=False).encode("utf-8") + b"\n"


def decode_response(data: bytes) -> Response:
    """Deserialize bytes to a Response."""
    obj = json.loads(data.decode("utf-8"))
    return Response(status=obj["status"], message=obj.get("message"))
