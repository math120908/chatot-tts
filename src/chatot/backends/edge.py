"""Edge TTS backend — uses Microsoft Edge's online TTS service via edge-tts."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

from chatot.backends.base import TTSBackend

logger = logging.getLogger(__name__)

# Good defaults — natural-sounding en-US voices.
_DEFAULT_VOICE = "en-US-AvaNeural"


def _run_async(coro):
    """Run a coroutine, handling both 'no loop' and 'inside a loop' cases."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (e.g. daemon server) — run in a new thread.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class EdgeBackend(TTSBackend):
    """TTS backend using edge-tts (Microsoft Edge online TTS).

    No GPU required — audio is synthesized server-side and streamed back.
    Requires internet connectivity.
    """

    def __init__(self) -> None:
        self.voice: str = _DEFAULT_VOICE
        self.rate: str = "+0%"
        self.language: str = "en"

    def load(self, config: dict) -> None:
        """Configure voice settings. No model to load."""
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise ImportError(
                "edge-tts is not installed. Run: pip install edge-tts"
            )

        self.voice = config.get("voice", _DEFAULT_VOICE)
        self.rate = config.get("rate", "+0%")
        self.language = config.get("language", "en")

        # If user specified a bare voice name (e.g. "Brian"), try to resolve
        # it to a full edge-tts voice ID.
        if "Neural" not in self.voice:
            self.voice = self._resolve_voice(self.voice)

        logger.info("Edge TTS configured: voice=%s, rate=%s", self.voice, self.rate)

    def _resolve_voice(self, short_name: str) -> str:
        """Try to resolve a short name like 'Brian' to 'en-US-BrianNeural'."""
        import edge_tts

        voices = _run_async(edge_tts.list_voices())
        lower = short_name.lower()
        for v in voices:
            vid = v["ShortName"]
            name_part = vid.split("-")[-1].replace("Neural", "").replace("Multilingual", "").lower()
            if name_part == lower:
                logger.info("Resolved voice %r -> %s", short_name, vid)
                return vid

        logger.warning(
            "Could not resolve voice %r, using as-is. "
            "Run 'edge-tts --list-voices' to see available voices.",
            short_name,
        )
        return short_name

    def speak(self, text: str, **kwargs) -> None:
        """Generate audio via edge-tts and play with afplay."""
        import edge_tts

        voice = kwargs.get("voice", self.voice)
        if "Neural" not in voice:
            voice = self._resolve_voice(voice)

        logger.info("Speaking: %.80s (voice=%s)", text, voice)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            rate = kwargs.get("rate", self.rate)
            comm = edge_tts.Communicate(text, voice, rate=rate)
            _run_async(comm.save(str(tmp_path)))

            # Save a copy if requested.
            save_path = kwargs.get("save_path")
            if save_path:
                import shutil
                shutil.copy2(str(tmp_path), str(save_path))
                logger.info("Audio saved to %s", save_path)

            # Play audio — afplay is available on all macOS.
            result = subprocess.run(
                ["afplay", str(tmp_path)],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.error("afplay failed: %s", result.stderr.decode())
                raise RuntimeError(f"Audio playback failed: {result.stderr.decode()}")

            logger.info("speak() completed successfully")
        finally:
            tmp_path.unlink(missing_ok=True)

    def unload(self) -> None:
        """Nothing to unload — no persistent model."""
        logger.info("Edge TTS backend unloaded")
