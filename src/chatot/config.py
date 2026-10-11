"""Configuration loading and defaults for chatot-tts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chatot._migrate import is_v1, migrate, migrate_file

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path("~/.config/chatot").expanduser()
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

_DEFAULT_YAML = """\
routing:
  default_lang: auto
  voices:
    en: en-US-AvaNeural
    zh: zh-TW-HsiaoChenNeural
    ko: ko-KR-SunHiNeural
    ja: ja-JP-NanamiNeural
    sw: sw-TZ-RehemaNeural
    mixed: Ryan

backends:
  primary: edge
  secondary: null

daemon:
  socket: ~/.config/chatot/var/process/chatotd.sock
  pid_file: ~/.config/chatot/var/process/chatotd.pid
  log_file: ~/.config/chatot/var/logs/chatotd.log
  log_level: INFO
"""

_DEFAULT_VOICES = {
    "en": "en-US-AvaNeural",
    "zh": "zh-TW-HsiaoChenNeural",
    "ko": "ko-KR-SunHiNeural",
    "ja": "ja-JP-NanamiNeural",
    "sw": "sw-TZ-RehemaNeural",
    "mixed": "Ryan",
}


def _default_socket() -> Path:
    return Path("~/.config/chatot/var/process/chatotd.sock").expanduser()


def _default_pid() -> Path:
    return Path("~/.config/chatot/var/process/chatotd.pid").expanduser()


def _default_log() -> Path:
    return Path("~/.config/chatot/var/logs/chatotd.log").expanduser()


@dataclass
class ChatotConfig:
    """Runtime configuration for chatot daemon and client."""

    # Routing
    default_lang: str = "auto"
    voices: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_VOICES))

    # Backends
    primary_backend: str = "edge"
    secondary_backend: str | None = None
    backend_options: dict[str, dict[str, str]] = field(default_factory=dict)

    # Daemon infra
    socket_path: Path = field(default_factory=_default_socket)
    pid_file: Path = field(default_factory=_default_pid)
    log_file: Path = field(default_factory=_default_log)
    log_level: str = "INFO"

    @property
    def default_voice(self) -> str:
        """Default voice derived from voices map and default_lang."""
        if self.default_lang in self.voices:
            return self.voices[self.default_lang]
        return self.voices.get("en", next(iter(self.voices.values())))

    def get_backend_option(self, backend: str, key: str, default: str = "") -> str:
        """Get a backend-specific option."""
        return self.backend_options.get(backend, {}).get(key, default)


def _expand_path(value: Any) -> Path:
    """Expand ~ and resolve a path value."""
    return Path(str(value)).expanduser()


def _parse_config(data: dict[str, Any]) -> ChatotConfig:
    """Build ChatotConfig from parsed YAML dict.

    Accepts both v1 (flat) and v2 (structured) formats — v1 is
    auto-migrated in memory before parsing.
    """
    if is_v1(data):
        data = migrate(data)

    routing = data.get("routing", {}) or {}
    backends = data.get("backends", {}) or {}
    daemon = data.get("daemon", {}) or {}
    kwargs: dict[str, Any] = {}

    # Routing
    if "default_lang" in routing:
        kwargs["default_lang"] = str(routing["default_lang"])

    voices = dict(_DEFAULT_VOICES)
    if "voices" in routing and isinstance(routing["voices"], dict):
        voices.update({str(k): str(v) for k, v in routing["voices"].items()})
    kwargs["voices"] = voices

    # Backends
    if "primary" in backends:
        kwargs["primary_backend"] = str(backends["primary"])
    if "secondary" in backends:
        raw = backends["secondary"]
        kwargs["secondary_backend"] = str(raw) if raw is not None else None

    # Collect per-backend options (keys other than primary/secondary).
    backend_options: dict[str, dict[str, str]] = {}
    for key, val in backends.items():
        if key in ("primary", "secondary"):
            continue
        if isinstance(val, dict):
            backend_options[key] = {str(k): str(v) for k, v in val.items()}
    kwargs["backend_options"] = backend_options

    # Daemon infra
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
    v1 configs are auto-migrated on disk.

    Returns a fully resolved :class:`ChatotConfig` with ``~`` expanded.
    """
    config_path = Path(path).expanduser() if path is not None else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_DEFAULT_YAML)

    text = config_path.read_text()
    data = yaml.safe_load(text) or {}

    # Auto-migrate v1 config file on disk.
    if is_v1(data):
        logger.info("Migrating v1 config to v2 format: %s", config_path)
        migrate_file(config_path)
        text = config_path.read_text()
        data = yaml.safe_load(text) or {}

    config = _parse_config(data)

    # Ensure runtime directories exist.
    config.socket_path.parent.mkdir(parents=True, exist_ok=True)
    config.pid_file.parent.mkdir(parents=True, exist_ok=True)
    config.log_file.parent.mkdir(parents=True, exist_ok=True)

    return config
