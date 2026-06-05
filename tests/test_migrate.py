"""Tests for chatot._migrate module."""

from __future__ import annotations

from pathlib import Path

import yaml

from chatot._migrate import is_v1, migrate, migrate_file


class TestIsV1:
    def test_v1_flat(self) -> None:
        assert is_v1({"backend": "edge", "voice": "Ava", "language": "en"})

    def test_v1_minimal(self) -> None:
        assert is_v1({"backend": "edge"})

    def test_v2_structured(self) -> None:
        assert not is_v1({"routing": {}, "backends": {}})

    def test_empty_is_not_v1(self) -> None:
        assert not is_v1({})

    def test_daemon_only_is_not_v1(self) -> None:
        assert not is_v1({"daemon": {"log_level": "DEBUG"}})


class TestMigrate:
    def test_basic_migration(self) -> None:
        old = {"backend": "edge", "voice": "en-US-AvaNeural", "language": "en"}
        new = migrate(old)
        assert new["routing"]["default_lang"] == "auto"
        assert new["backends"]["primary"] == "edge"
        assert new["routing"]["voices"]["en"] == "en-US-AvaNeural"

    def test_preserves_daemon_section(self) -> None:
        old = {
            "backend": "edge",
            "language": "en",
            "daemon": {"log_level": "DEBUG"},
        }
        new = migrate(old)
        assert new["daemon"]["log_level"] == "DEBUG"

    def test_rate_in_backend_options(self) -> None:
        old = {"backend": "edge", "language": "en", "rate": "+15%"}
        new = migrate(old)
        assert new["backends"]["edge"]["rate"] == "+15%"

    def test_default_rate_not_stored(self) -> None:
        old = {"backend": "edge", "language": "en", "rate": "+0%"}
        new = migrate(old)
        assert "edge" not in new["backends"]

    def test_model_in_backend_options(self) -> None:
        old = {"backend": "edge", "language": "en", "model": "my-model"}
        new = migrate(old)
        assert new["backends"]["qwen3"]["model"] == "my-model"

    def test_empty_model_not_stored(self) -> None:
        old = {"backend": "edge", "language": "en", "model": ""}
        new = migrate(old)
        assert "qwen3" not in new["backends"]

    def test_secondary_backend(self) -> None:
        old = {"backend": "edge", "language": "en", "secondary_backend": "kokoro"}
        new = migrate(old)
        assert new["backends"]["secondary"] == "kokoro"

    def test_secondary_backend_none(self) -> None:
        old = {"backend": "edge", "language": "en", "secondary_backend": None}
        new = migrate(old)
        assert "secondary" not in new["backends"]

    def test_voices_map_preserved(self) -> None:
        old = {
            "backend": "edge",
            "language": "en",
            "voices": {"en": "Brian", "zh": "Yunxi"},
        }
        new = migrate(old)
        assert new["routing"]["voices"]["en"] == "Brian"
        assert new["routing"]["voices"]["zh"] == "Yunxi"
        # Defaults filled in
        assert new["routing"]["voices"]["ko"] == "ko-KR-SunHiNeural"

    def test_single_voice_becomes_en_entry(self) -> None:
        old = {"backend": "edge", "language": "en", "voice": "CustomVoice"}
        new = migrate(old)
        assert new["routing"]["voices"]["en"] == "CustomVoice"

    def test_v2_passthrough(self) -> None:
        v2 = {"routing": {"default_lang": "auto"}, "backends": {"primary": "edge"}}
        assert migrate(v2) is v2


class TestMigrateFile:
    def test_migrates_v1_file(self, tmp_path: Path) -> None:
        f = tmp_path / "config.yaml"
        f.write_text(yaml.dump({"backend": "edge", "voice": "Ava", "language": "en"}))
        assert migrate_file(f)

        data = yaml.safe_load(f.read_text())
        assert "routing" in data
        assert "backends" in data
        assert "backend" not in data

    def test_skips_v2_file(self, tmp_path: Path) -> None:
        f = tmp_path / "config.yaml"
        f.write_text(yaml.dump({"routing": {}, "backends": {}}))
        assert not migrate_file(f)

    def test_missing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "nope.yaml"
        assert not migrate_file(f)
