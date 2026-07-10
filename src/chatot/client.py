"""Chatot TTS client -- sends text to the chatotd daemon for synthesis."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from chatot.config import load_config
from chatot.protocol import Request, decode_response, encode_request

_READ_TIMEOUT_S = 300  # generous for qwen3 MLX generation (~110s) + lazy-load


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="chatot",
        description="Send text to chatotd for speech synthesis.",
    )
    parser.add_argument("text", nargs="?", default=None, help="Text to synthesize")
    parser.add_argument("--lang", default=None, help="Override language (e.g. en, zh, ko, ja, sw). Note: use 'ja' for Japanese ('jp' is rejected); 'sw' (Swahili) must be set explicitly as it cannot be auto-detected")
    parser.add_argument("--voice", default=None, help="Override voice (e.g. vivian, serena, ryan)")
    parser.add_argument("--save", default=None, metavar="PATH", help="Save audio to file (without playing)")
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


_VOICE_CACHE_PATH = Path("~/.cache/chatot/voice-cache.json").expanduser()
_VOICE_CACHE_MAX_AGE_S = 7 * 24 * 3600  # 1 week


def _load_voice_cache() -> dict | None:
    """Load cached voice list if fresh enough."""
    try:
        import time
        if not _VOICE_CACHE_PATH.exists():
            return None
        age = time.time() - _VOICE_CACHE_PATH.stat().st_mtime
        if age > _VOICE_CACHE_MAX_AGE_S:
            return None
        import json
        return json.loads(_VOICE_CACHE_PATH.read_text())
    except Exception:
        return None


def _save_voice_cache(data: dict) -> None:
    """Write voice cache to disk."""
    import json
    _VOICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _VOICE_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _fetch_edge_voices() -> list[dict[str, str]]:
    """Fetch all Edge TTS voices. Returns [{short_name, locale, gender}, ...]."""
    import asyncio
    import edge_tts  # type: ignore[import-untyped]

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        raw = asyncio.run(edge_tts.list_voices())
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            raw = pool.submit(asyncio.run, edge_tts.list_voices()).result()

    return [
        {
            "short_name": v["ShortName"],
            "locale": v["Locale"],
            "gender": v["Gender"],
        }
        for v in raw
    ]


def _fetch_qwen3_voices(config) -> list[str]:
    """Load Qwen3 model and return speaker names."""
    from mlx_audio.tts.utils import load_model  # type: ignore[import-untyped]
    model_id = config.get_backend_option("qwen3", "model")
    if not model_id:
        model_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    model = load_model(model_id)
    if hasattr(model, "get_supported_speakers"):
        return list(model.get_supported_speakers())
    return []


def _cmd_list_voices() -> int:
    """Print available voices grouped by backend, with caching."""
    config = load_config()
    cache = _load_voice_cache()

    # Fetch / refresh cache
    if cache is None:
        cache = {}
        print("Fetching voice lists (cached for 1 week)...", file=sys.stderr)

        try:
            cache["edge"] = _fetch_edge_voices()
        except Exception as e:
            print(f"  Edge: fetch failed ({e})", file=sys.stderr)
            cache["edge"] = []

        try:
            cache["qwen3"] = _fetch_qwen3_voices(config)
        except ImportError:
            print("  Qwen3: mlx-audio not installed", file=sys.stderr)
            cache["qwen3"] = []
        except Exception as e:
            print(f"  Qwen3: fetch failed ({e})", file=sys.stderr)
            cache["qwen3"] = []

        _save_voice_cache(cache)
        print(file=sys.stderr)

    # Display Edge voices grouped by locale
    configured = set(config.voices.values())
    edge_voices = cache.get("edge", [])
    if edge_voices:
        from collections import defaultdict
        by_locale: dict[str, list[str]] = defaultdict(list)
        for v in edge_voices:
            by_locale[v["locale"]].append(v["short_name"])

        print("Edge TTS voices (online):")
        for locale in sorted(by_locale):
            names = sorted(by_locale[locale])
            parts = []
            for n in names:
                suffix = n.split("-")[-1]
                marker = " *" if n in configured else ""
                parts.append(f"{suffix}{marker}")
            print(f"  {locale}: {', '.join(parts)}")

    # Display Qwen3 voices
    qwen3_voices = cache.get("qwen3", [])
    if qwen3_voices:
        mixed_voice = config.voices.get("mixed", "").lower()
        print()
        print("Qwen3 voices (local MLX, multilingual):")
        for v in qwen3_voices:
            marker = " (default)" if v.lower() == mixed_voice else ""
            print(f"  {v}{marker}")

    if not edge_voices and not qwen3_voices:
        print("No voices available.")

    return 0


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
