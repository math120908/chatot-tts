"""Abstract base class for TTS backends."""

from abc import ABC, abstractmethod


class TTSBackend(ABC):
    """Interface that all TTS backends must implement."""

    @abstractmethod
    def load(self, config: dict) -> None:
        """Load model into memory.

        Args:
            config: Backend-specific configuration (model_id, voice, language, etc.)
        """
        ...

    @abstractmethod
    def speak(self, text: str, **kwargs) -> None:
        """Generate and play audio. Blocks until playback complete.

        Args:
            text: Text to synthesize.
            **kwargs: Per-call overrides (voice, language, etc.)
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...
