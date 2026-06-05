"""Migrate chatot config.yaml from v1 (flat) to v2 (structured) format.

v1 format (pre-multilingual):
    backend: edge
    voice: en-US-AvaNeural
    language: en
    rate: "+0%"
    model: ""
    secondary_backend: qwen3
    voices: {en: ..., zh: ..., ko: ..., mixed: ...}
    daemon: {...}

v2 format:
    routing:
      default_lang: auto
      voices: {en: ..., zh: ..., ko: ..., mixed: ...}
    backends:
      primary: edge
      secondary: qwen3
      edge: {rate: "+0%"}
      qwen3: {model: ""}
    daemon: {...}

This module can be deleted once all configs have been migrated.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_VOICES = {
    "en": "en-US-AvaNeural",
    "zh": "zh-TW-HsiaoChenNeural",
    "ko": "ko-KR-SunHiNeural",
    "mixed": "Ryan",
}


def is_v1(data: dict) -> bool:
    """Return True if *data* is v1 (flat) format."""
    if "routing" in data or "backends" in data:
        return False
    return any(k in data for k in ("backend", "voice", "language", "rate", "model"))


def migrate(data: dict) -> dict:
    """Convert v1 dict to v2 dict. Returns a new dict."""
    if not is_v1(data):
        return data

    primary = data.get("backend", "edge")
    secondary_raw = data.get("secondary_backend", "qwen3")
    secondary = str(secondary_raw) if secondary_raw is not None else None

    # Build voices map.
    voices = dict(_DEFAULT_VOICES)
    if "voices" in data and isinstance(data["voices"], dict):
        voices.update({str(k): str(v) for k, v in data["voices"].items()})
    elif "voice" in data:
        # Old config had a single voice — use it as the "en" entry.
        voices["en"] = str(data["voice"])

    routing: dict = {
        "default_lang": "auto",
        "voices": voices,
    }

    backends: dict = {"primary": primary}
    if secondary is not None:
        backends["secondary"] = secondary

    edge_opts: dict = {}
    if "rate" in data and data["rate"] != "+0%":
        edge_opts["rate"] = str(data["rate"])
    if edge_opts:
        backends["edge"] = edge_opts

    qwen3_opts: dict = {}
    if "model" in data and data["model"]:
        qwen3_opts["model"] = str(data["model"])
    if qwen3_opts:
        backends["qwen3"] = qwen3_opts

    result: dict = {
        "routing": routing,
        "backends": backends,
    }

    if "daemon" in data:
        result["daemon"] = data["daemon"]

    return result


def migrate_file(path: Path) -> bool:
    """Migrate a config file in-place. Returns True if migrated."""
    if not path.exists():
        return False

    text = path.read_text()
    data = yaml.safe_load(text) or {}

    if not is_v1(data):
        return False

    new_data = migrate(data)
    new_text = yaml.dump(
        new_data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    path.write_text(new_text)
    return True
