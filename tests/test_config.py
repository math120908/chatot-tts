"""Tests for chatot.config module."""

from __future__ import annotations

from pathlib import Path

import yaml

from chatot.config import ChatotConfig, load_config, _parse_config


class TestDefaultConfig:
    """Test default ChatotConfig values."""

    def test_default_primary_backend(self) -> None:
        cfg = ChatotConfig()
        assert cfg.primary_backend == "edge"

    def test_default_secondary_backend(self) -> None:
        cfg = ChatotConfig()
        assert cfg.secondary_backend == "qwen3"

    def test_default_lang(self) -> None:
        cfg = ChatotConfig()
        assert cfg.default_lang == "auto"

    def test_default_voice_property(self) -> None:
        cfg = ChatotConfig()
        # default_lang is "auto" which isn't in voices, so falls back to "en"
        assert cfg.default_voice == "en-US-AvaNeural"

    def test_default_log_level(self) -> None:
        cfg = ChatotConfig()
        assert cfg.log_level == "INFO"

    def test_default_socket_path_expanded(self) -> None:
        cfg = ChatotConfig()
        assert "~" not in str(cfg.socket_path)
        assert str(cfg.socket_path).endswith("chatotd.sock")

    def test_default_pid_file_expanded(self) -> None:
        cfg = ChatotConfig()
        assert "~" not in str(cfg.pid_file)
        assert str(cfg.pid_file).endswith("chatotd.pid")

    def test_default_log_file_expanded(self) -> None:
        cfg = ChatotConfig()
        assert "~" not in str(cfg.log_file)
        assert str(cfg.log_file).endswith("chatotd.log")


class TestLoadConfig:
    """Test load_config behavior."""

    def test_creates_config_if_not_exists(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        assert not config_file.exists()

        cfg = load_config(config_file)
        assert config_file.exists()
        assert cfg.primary_backend == "edge"

    def test_reads_existing_v2_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        data = {
            "routing": {
                "default_lang": "zh",
                "voices": {
                    "en": "en-US-BrianNeural",
                    "zh": "zh-TW-HsiaoChenNeural",
                },
            },
            "backends": {
                "primary": "edge",
                "secondary": "qwen3",
                "edge": {"rate": "+10%"},
            },
            "daemon": {
                "socket": str(tmp_path / "test.sock"),
                "pid_file": str(tmp_path / "test.pid"),
                "log_file": str(tmp_path / "test.log"),
                "log_level": "debug",
            },
        }
        config_file.write_text(yaml.dump(data))

        cfg = load_config(config_file)
        assert cfg.primary_backend == "edge"
        assert cfg.default_lang == "zh"
        assert cfg.voices["en"] == "en-US-BrianNeural"
        assert cfg.get_backend_option("edge", "rate") == "+10%"
        assert cfg.log_level == "DEBUG"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        config_file = tmp_path / "deep" / "nested" / "config.yaml"
        cfg = load_config(config_file)
        assert config_file.exists()
        assert cfg.primary_backend == "edge"


class TestParseConfigV2:
    """Test parsing of v2 structured config."""

    def test_defaults_from_empty(self) -> None:
        config = _parse_config({})
        assert config.primary_backend == "edge"
        assert config.secondary_backend == "qwen3"
        assert config.default_lang == "auto"

    def test_default_voices(self) -> None:
        config = _parse_config({})
        assert config.voices["en"] == "en-US-AvaNeural"
        assert config.voices["zh"] == "zh-TW-HsiaoChenNeural"
        assert config.voices["ko"] == "ko-KR-SunHiNeural"
        assert config.voices["ja"] == "ja-JP-NanamiNeural"
        assert config.voices["sw"] == "sw-TZ-RehemaNeural"
        assert config.voices["mixed"] == "Ryan"

    def test_custom_voices(self) -> None:
        data = {
            "routing": {
                "voices": {
                    "en": "en-US-BrianNeural",
                    "mixed": "Vivian",
                }
            }
        }
        config = _parse_config(data)
        assert config.voices["en"] == "en-US-BrianNeural"
        assert config.voices["mixed"] == "Vivian"
        # Defaults preserved for unspecified keys
        assert config.voices["zh"] == "zh-TW-HsiaoChenNeural"

    def test_backend_options(self) -> None:
        data = {
            "backends": {
                "primary": "edge",
                "edge": {"rate": "+20%"},
                "qwen3": {"model": "my-model"},
            }
        }
        config = _parse_config(data)
        assert config.get_backend_option("edge", "rate") == "+20%"
        assert config.get_backend_option("qwen3", "model") == "my-model"

    def test_secondary_backend_none(self) -> None:
        data = {"backends": {"primary": "edge", "secondary": None}}
        config = _parse_config(data)
        assert config.secondary_backend is None

    def test_default_voice_property_with_known_lang(self) -> None:
        data = {"routing": {"default_lang": "zh"}}
        config = _parse_config(data)
        assert config.default_voice == "zh-TW-HsiaoChenNeural"

    def test_default_voice_property_with_auto(self) -> None:
        config = _parse_config({})
        # "auto" not in voices → falls back to "en"
        assert config.default_voice == "en-US-AvaNeural"

    def test_get_backend_option_missing_returns_default(self) -> None:
        config = _parse_config({})
        assert config.get_backend_option("edge", "rate", "+0%") == "+0%"
        assert config.get_backend_option("nonexistent", "foo") == ""


class TestPathExpansion:
    """Test that ~ paths get expanded."""

    def test_tilde_expanded_in_socket(self) -> None:
        data = {"daemon": {"socket": "~/custom/chatotd.sock"}}
        cfg = _parse_config(data)
        assert "~" not in str(cfg.socket_path)
        assert str(cfg.socket_path).endswith("custom/chatotd.sock")

    def test_tilde_expanded_in_pid(self) -> None:
        data = {"daemon": {"pid_file": "~/custom/chatotd.pid"}}
        cfg = _parse_config(data)
        assert "~" not in str(cfg.pid_file)

    def test_tilde_expanded_in_log(self) -> None:
        data = {"daemon": {"log_file": "~/custom/chatotd.log"}}
        cfg = _parse_config(data)
        assert "~" not in str(cfg.log_file)


class TestV1Migration:
    """Test that v1 configs are correctly parsed via migration."""

    def test_v1_flat_config(self) -> None:
        data = {
            "backend": "edge",
            "voice": "en-US-AvaNeural",
            "language": "en",
        }
        config = _parse_config(data)
        assert config.primary_backend == "edge"
        assert config.default_lang == "auto"
        assert config.voices["en"] == "en-US-AvaNeural"

    def test_v1_with_secondary(self) -> None:
        data = {
            "backend": "edge",
            "voice": "en-US-AvaNeural",
            "language": "en",
            "secondary_backend": "kokoro",
        }
        config = _parse_config(data)
        assert config.secondary_backend == "kokoro"

    def test_v1_with_rate_and_model(self) -> None:
        data = {
            "backend": "edge",
            "voice": "en-US-AvaNeural",
            "language": "en",
            "rate": "+10%",
            "model": "my-model",
        }
        config = _parse_config(data)
        assert config.get_backend_option("edge", "rate") == "+10%"
        assert config.get_backend_option("qwen3", "model") == "my-model"

    def test_v1_with_voices_map(self) -> None:
        data = {
            "backend": "edge",
            "language": "en",
            "voices": {"en": "en-US-BrianNeural", "zh": "zh-CN-YunxiNeural"},
        }
        config = _parse_config(data)
        assert config.voices["en"] == "en-US-BrianNeural"
        assert config.voices["zh"] == "zh-CN-YunxiNeural"

    def test_v1_file_migrated_on_disk(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        old_data = {
            "backend": "edge",
            "voice": "en-US-AvaNeural",
            "language": "en",
            "daemon": {
                "socket": str(tmp_path / "chatotd.sock"),
                "pid_file": str(tmp_path / "chatotd.pid"),
                "log_file": str(tmp_path / "chatotd.log"),
            },
        }
        config_file.write_text(yaml.dump(old_data))

        cfg = load_config(config_file)
        assert cfg.primary_backend == "edge"
        assert cfg.default_lang == "auto"

        # Verify the file was rewritten to v2 format.
        reread = yaml.safe_load(config_file.read_text())
        assert "routing" in reread
        assert "backends" in reread
        assert "backend" not in reread
