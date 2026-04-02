"""Qwen3 TTS backend using mlx-audio."""

from __future__ import annotations

import gc
import logging
from typing import Any

from chatot.backends.base import TTSBackend

logger = logging.getLogger(__name__)


class Qwen3Backend(TTSBackend):
    """TTS backend powered by Qwen3-TTS via mlx-audio.

    Lazy-imports mlx_audio so the package isn't required at import time.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.voice: str = "Ryan"
        self.language: str = "en"

    def load(self, config: dict) -> None:
        """Load Qwen3-TTS model via mlx-audio.

        Args:
            config: Keys: 'model', 'voice', 'language'.
        """
        from mlx_audio.tts.utils import load_model  # type: ignore[import-untyped]

        model_id = config.get("model", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
        self.voice = config.get("voice", "Ryan")
        self.language = config.get("language", "en")

        logger.info("Loading model: %s", model_id)
        self.model = load_model(model_id)
        logger.info("Model loaded successfully")

    def speak(self, text: str, **kwargs) -> None:
        """Generate and play audio. Blocks until playback complete.

        Args:
            text: Text to synthesize.
            **kwargs: Per-call overrides -- 'voice', 'language'.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from mlx_audio.tts.generate import generate_audio  # type: ignore[import-untyped]

        voice = kwargs.get("voice", self.voice)
        language = kwargs.get("language", self.language)

        logger.debug("Speaking: %.60s... (voice=%s, lang=%s)", text, voice, language)
        generate_audio(
            text=text,
            model=self.model,
            voice=voice,
            lang_code=language,
            stream=True,
            play=True,
            verbose=False,
            save=False,
        )
        # Extra buffer time — mlx-audio's AudioPlayer.stop() only sleeps 100ms
        # after drain, which can truncate the final audio chunk
        import time
        time.sleep(0.3)

    def unload(self) -> None:
        """Release model reference."""
        logger.info("Unloading model")
        del self.model
        self.model = None
        gc.collect()
        logger.info("Model unloaded")
