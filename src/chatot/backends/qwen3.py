"""Qwen3 TTS backend using mlx-audio."""

from __future__ import annotations

import gc
import io
import logging
import sys
from typing import Any

from chatot.backends.base import TTSBackend

logger = logging.getLogger(__name__)


def _patch_audio_player() -> None:
    """Fix AudioPlayer.stop() — upstream skips stream cleanup when the
    callback already set ``self.playing = False`` before stop() runs.

    This leaks ``sd.OutputStream`` handles and exhausts the audio system
    after a few calls (typically 2-3).
    """
    try:
        from mlx_audio.tts.generate import AudioPlayer
    except ImportError:
        return

    if getattr(AudioPlayer.stop, "_chatot_patched", False):
        return  # already patched

    def _fixed_stop(self: Any) -> None:
        # Wait for audio drain only if still playing.
        if self.playing:
            self.wait_for_drain()
            import sounddevice as sd
            sd.sleep(100)
        # ALWAYS close the stream — the original skips this when
        # self.playing was already cleared by the callback.
        self.stop_stream()
        self.playing = False

    _fixed_stop._chatot_patched = True  # type: ignore[attr-defined]
    AudioPlayer.stop = _fixed_stop  # type: ignore[assignment]
    logger.debug("Patched AudioPlayer.stop() to always close stream")


class _OutputCapture:
    """Tee stdout+stderr to a buffer while still passing through to originals."""

    def __init__(self) -> None:
        self.buffer = io.StringIO()

    def __enter__(self) -> "_OutputCapture":
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        sys.stdout = self._Tee(self._orig_out, self.buffer)  # type: ignore[assignment]
        sys.stderr = self._Tee(self._orig_err, self.buffer)  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        sys.stdout = self._orig_out
        sys.stderr = self._orig_err

    class _Tee:
        def __init__(self, original: Any, buf: io.StringIO) -> None:
            self._orig = original
            self._buf = buf

        def write(self, s: str) -> int:
            self._buf.write(s)
            self._orig.write(s)
            return len(s)

        def flush(self) -> None:
            self._orig.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._orig, name)


class Qwen3Backend(TTSBackend):
    """TTS backend powered by Qwen3-TTS via mlx-audio.

    Lazy-imports mlx_audio so the package isn't required at import time.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.voice: str = "Ryan"
        self.language: str = "en"
        self._poisoned: bool = False

    def load(self, config: dict) -> None:
        """Load Qwen3-TTS model via mlx-audio."""
        from mlx_audio.tts.utils import load_model  # type: ignore[import-untyped]

        # Patch AudioPlayer before any usage.
        _patch_audio_player()

        model_id = config.get("model", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
        self.voice = config.get("voice", "Ryan")
        self.language = config.get("language", "en")

        logger.info("Loading model: %s", model_id)
        self.model = load_model(model_id)
        self._poisoned = False
        logger.info("Model loaded successfully")

    def speak(self, text: str, **kwargs) -> None:
        """Generate and play audio. Blocks until playback complete."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._poisoned:
            raise RuntimeError(
                "Backend is poisoned after a previous failure (likely Metal GPU "
                "corruption). Restart the daemon: chatotd stop && chatotd"
            )

        from mlx_audio.tts.generate import generate_audio  # type: ignore[import-untyped]

        voice = kwargs.get("voice", self.voice)
        language = kwargs.get("language", self.language)
        save_path = kwargs.get("save_path")

        logger.info("Speaking: %.80s (voice=%s, lang=%s)", text, voice, language)

        # generate_audio() has a bare `except Exception` that swallows errors
        # (prints traceback but doesn't re-raise).  Capture output to detect it.
        play = not save_path
        cap = _OutputCapture()
        with cap:
            generate_audio(
                text=text,
                model=self.model,
                voice=voice,
                lang_code=language,
                stream=True,
                play=play,
                verbose=False,
                save=bool(save_path),
                **({"output_file": save_path} if save_path else {}),
            )

        captured = cap.buffer.getvalue()
        if "Error loading model:" in captured or "RuntimeError" in captured:
            self._poisoned = True
            logger.error(
                "generate_audio() swallowed an error — backend poisoned. "
                "Restart daemon to recover. Captured:\n%s",
                captured[-500:],
            )
            raise RuntimeError(
                "TTS generation failed (Metal GPU error). "
                "Restart the daemon: chatotd stop && chatotd"
            )

        if play:
            import time
            time.sleep(0.8)
        logger.info("speak() completed successfully")

    def unload(self) -> None:
        """Release model reference."""
        logger.info("Unloading model")
        del self.model
        self.model = None
        gc.collect()
        logger.info("Model unloaded")
