"""Configuration loading and defaults for chatot-tts."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path("~/.config/chatot").expanduser()
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

_DEFAULT_YAML = """\
backend: qwen3
model: mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
voice: vivian
language: en

daemon:
  socket: ~/.config/chatot/var/process/chatotd.sock
  pid_file: ~/.config/chatot/var/process/chatotd.pid
  log_file: ~/.config/chatot/var/logs/chatotd.log
  log_level: INFO
"""


def _default_socket() -> Path:
    return Path("~/.config/chatot/var/process/chatotd.sock").expanduser()


def _default_pid() -> Path:
    return Path("~/.config/chatot/var/process/chatotd.pid").expanduser()


def _default_log() -> Path:
    return Path("~/.config/chatot/var/logs/chatotd.log").expanduser()


@dataclass
class ChatotConfig:
    """Runtime configuration for chatot daemon and client."""

    backend: str = "qwen3"
    model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    voice: str = "Ryan"
    language: str = "en"
    socket_path: Path = field(default_factory=_default_socket)
    pid_file: Path = field(default_factory=_default_pid)
    log_file: Path = field(default_factory=_default_log)
    log_level: str = "INFO"


def _expand_path(value: Any) -> Path:
    """Expand ~ and resolve a path value."""
    return Path(str(value)).expanduser()


def _parse_config(data: dict[str, Any]) -> ChatotConfig:
    """Build ChatotConfig from parsed YAML dict."""
    daemon = data.get("daemon", {}) or {}
    kwargs: dict[str, Any] = {}

    if "backend" in data:
        kwargs["backend"] = str(data["backend"])
    if "model" in data:
        kwargs["model"] = str(data["model"])
    if "voice" in data:
        kwargs["voice"] = str(data["voice"])
    if "language" in data:
        kwargs["language"] = str(data["language"])
    if "socket" in daemon:
        kwargs["socket_path"] = _expand_path(daemon["socket"])
    if "pid_file" in daemon:
        kwargs["pid_file"] = _expand_path(daemon["pid_file"])
    if "log_file" in daemon:
        kwargs["log_file"] = _expand_path(daemon["log_file"])
    if "log_level" in daemon:
        kwargs["log_level"] = str(daemon["log_level"]).upper()

    return ChatotConfig(**kwargs)


def load_config(path: Path | None = None) -> ChatotConfig:
    """Load configuration from a YAML file.

    If *path* is ``None``, uses ``~/.config/chatot/config.yaml``.
    If the file does not exist, it is created with sensible defaults.
    All parent directories are created automatically.

    Returns a fully resolved :class:`ChatotConfig` with ``~`` expanded.
    """
    config_path = Path(path).expanduser() if path is not None else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_DEFAULT_YAML)

    text = config_path.read_text()
    data = yaml.safe_load(text) or {}

    config = _parse_config(data)

    # Ensure runtime directories exist for socket, pid, and log files.
    config.socket_path.parent.mkdir(parents=True, exist_ok=True)
    config.pid_file.parent.mkdir(parents=True, exist_ok=True)
    config.log_file.parent.mkdir(parents=True, exist_ok=True)

    return config
