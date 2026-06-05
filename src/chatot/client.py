"""Chatot TTS client -- sends text to the chatotd daemon for synthesis."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from chatot.config import load_config
from chatot.protocol import Request, decode_response, encode_request

_READ_TIMEOUT_S = 120  # generous for first-time lazy-load of secondary backend


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="chatot",
        description="Send text to chatotd for speech synthesis.",
    )
    parser.add_argument("text", nargs="?", default=None, help="Text to synthesize")
    parser.add_argument("--lang", default=None, help="Override language (e.g. zh, en)")
    parser.add_argument("--voice", default=None, help="Override voice (e.g. vivian, serena, ryan)")
    parser.add_argument("--save", default=None, metavar="PATH", help="Save audio to file (in addition to playing)")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
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


async def _send(
    socket_path: str,
    text: str,
    lang: str | None,
    voice: str | None,
    save_path: str | None,
) -> int:
    """Connect to daemon, send request, return exit code."""
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
    except (ConnectionRefusedError, FileNotFoundError):
        print("chatotd is not running. Start it with: chatotd", file=sys.stderr)
        return 1

    try:
        payload = encode_request(Request(text=text, lang=lang, voice=voice, save_path=save_path))
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


def _cmd_list_voices() -> int:
    """Print available voices and return exit code."""
    try:
        from mlx_audio.tts.utils import load_model  # type: ignore[import-untyped]
        config = load_config()
        model_id = config.get_backend_option("qwen3", "model")
        if not model_id:
            model_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
        model = load_model(model_id)
        if hasattr(model, "get_supported_speakers"):
            speakers = model.get_supported_speakers()
            default = config.default_voice
            print(f"Available voices (default: {default}):")
            for v in speakers:
                marker = " (default)" if v == default else ""
                print(f"  {v}{marker}")
        else:
            print("This model does not support custom voices.")
        return 0
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for chatot client."""
    args = _parse_args(argv)

    if args.list_voices:
        sys.exit(_cmd_list_voices())

    text = _resolve_text(args)
    if text is None:
        print("Usage: chatot \"text\" or echo \"text\" | chatot", file=sys.stderr)
        sys.exit(1)

    if args.socket:
        socket_path = args.socket
    else:
        config = load_config()
        socket_path = str(config.socket_path)

    save = str(Path(args.save).resolve()) if args.save else None
    rc = asyncio.run(_send(socket_path, text, args.lang, args.voice, save))
    sys.exit(rc)


if __name__ == "__main__":
    main()
