"""Tests for chatot.config module."""

from __future__ import annotations

from pathlib import Path

import yaml

from chatot.config import ChatotConfig, load_config, _parse_config


class TestDefaultConfig:
    """Test default ChatotConfig values."""

    def test_default_backend(self) -> None:
        cfg = ChatotConfig()
        assert cfg.backend == "qwen3"

    def test_default_model(self) -> None:
        cfg = ChatotConfig()
        assert cfg.model == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

    def test_default_voice(self) -> None:
        cfg = ChatotConfig()
        assert cfg.voice == "Ryan"

    def test_default_language(self) -> None:
        cfg = ChatotConfig()
        assert cfg.language == "en"

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
        assert cfg.backend == "qwen3"

    def test_reads_existing_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        data = {
            "backend": "custom_backend",
            "model": "my-model",
            "voice": "Alice",
            "language": "zh",
            "daemon": {
                "socket": str(tmp_path / "test.sock"),
                "pid_file": str(tmp_path / "test.pid"),
                "log_file": str(tmp_path / "test.log"),
                "log_level": "debug",
            },
        }
        config_file.write_text(yaml.dump(data))

        cfg = load_config(config_file)
        assert cfg.backend == "custom_backend"
        assert cfg.model == "my-model"
        assert cfg.voice == "Alice"
        assert cfg.language == "zh"
        assert cfg.log_level == "DEBUG"

    def test_custom_values_override_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        data = {"voice": "Zoe", "language": "ja"}
        config_file.write_text(yaml.dump(data))

        cfg = load_config(config_file)
        assert cfg.voice == "Zoe"
        assert cfg.language == "ja"
        # Unset fields keep defaults
        assert cfg.backend == "qwen3"
        assert cfg.model == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        config_file = tmp_path / "deep" / "nested" / "config.yaml"
        cfg = load_config(config_file)
        assert config_file.exists()
        assert cfg.backend == "qwen3"


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
