"""Chatot TTS client -- sends text to the chatotd daemon for synthesis."""

from __future__ import annotations

import argparse
import asyncio
import sys

from chatot.config import load_config
from chatot.protocol import Request, decode_response, encode_request

_READ_TIMEOUT_S = 60


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="chatot",
        description="Send text to chatotd for speech synthesis.",
    )
    parser.add_argument("text", nargs="?", default=None, help="Text to synthesize")
    parser.add_argument("--lang", default=None, help="Override language (e.g. zh, en)")
    parser.add_argument("--socket", default=None, help="Path to daemon Unix socket")
    return parser.parse_args(argv)


def _resolve_text(args: argparse.Namespace) -> str | None:
    """Determine input text from positional arg or stdin.

    Positional argument wins when both are provided.
    """
    if args.text is not None:
        return args.text

    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text

    return None


async def _send(socket_path: str, text: str, lang: str | None) -> int:
    """Connect to daemon, send request, return exit code."""
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
    except (ConnectionRefusedError, FileNotFoundError):
        print("chatotd is not running. Start it with: chatotd", file=sys.stderr)
        return 1

    try:
        payload = encode_request(Request(text=text, lang=lang))
        writer.write(payload)
        await writer.drain()

        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=_READ_TIMEOUT_S)
        except asyncio.TimeoutError:
            print("Request timed out", file=sys.stderr)
            return 1

        if not raw:
            print("Daemon closed connection without response", file=sys.stderr)
            return 1

        resp = decode_response(raw)
        if resp.status == "ok":
            return 0

        msg = resp.message or "Unknown error"
        print(f"Error: {msg}", file=sys.stderr)
        return 1
    finally:
        writer.close()
        await writer.wait_closed()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for chatot client."""
    args = _parse_args(argv)

    text = _resolve_text(args)
    if text is None:
        print("Usage: chatot \"text\" or echo \"text\" | chatot", file=sys.stderr)
        sys.exit(1)

    if args.socket:
        socket_path = args.socket
    else:
        config = load_config()
        socket_path = str(config.socket_path)

    rc = asyncio.run(_send(socket_path, text, args.lang))
    sys.exit(rc)


if __name__ == "__main__":
    main()
