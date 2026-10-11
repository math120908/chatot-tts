"""Backend registry for chatot-tts."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatot.backends.base import TTSBackend

# Map short name -> fully qualified class path
BACKENDS: dict[str, str] = {
    "edge": "chatot.backends.edge.EdgeBackend",
}


def get_backend(name: str) -> TTSBackend:
    """Get a backend instance by short name.

    Args:
        name: Backend identifier (e.g. 'edge').

    Returns:
        An instantiated (but not yet loaded) TTSBackend.

    Raises:
        ValueError: If the backend name is not registered.
        ImportError: If the backend module cannot be imported.
    """
    if name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend: {name!r}. Available: {available}")

    module_path, class_name = BACKENDS[name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
