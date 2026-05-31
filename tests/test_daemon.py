"""Tests for chatot.daemon BackendPool."""

from __future__ import annotations

import pytest

from chatot.daemon import BackendPool


class _FakeBackend:
    def __init__(self) -> None:
        self.loaded = False
        self.spoken: list[tuple[str, dict]] = []

    def load(self, _config: dict) -> None:
        self.loaded = True

    def speak(self, text: str, **kwargs) -> None:
        self.spoken.append((text, kwargs))

    def unload(self) -> None:
        self.loaded = False


class TestBackendPool:
    def _make_pool(self) -> BackendPool:
        pool = BackendPool.__new__(BackendPool)
        pool._backends = {}
        pool._primary_name = "edge"
        pool._secondary_name = "qwen3"
        pool._backend_configs = {
            "edge": {"voice": "en-US-AvaNeural", "language": "en"},
            "qwen3": {"voice": "Ryan", "language": "en"},
        }
        pool._factory = lambda _name: _FakeBackend()
        return pool

    def test_load_primary(self) -> None:
        pool = self._make_pool()
        pool.load_primary()
        assert "edge" in pool._backends
        assert pool._backends["edge"].loaded

    def test_get_primary_returns_loaded(self) -> None:
        pool = self._make_pool()
        pool.load_primary()
        backend = pool.get("edge")
        assert backend.loaded

    def test_get_secondary_lazy_loads(self) -> None:
        pool = self._make_pool()
        pool.load_primary()
        assert "qwen3" not in pool._backends
        backend = pool.get("qwen3")
        assert backend.loaded

    def test_get_secondary_cached(self) -> None:
        pool = self._make_pool()
        pool.load_primary()
        b1 = pool.get("qwen3")
        b2 = pool.get("qwen3")
        assert b1 is b2

    def test_unload_all(self) -> None:
        pool = self._make_pool()
        pool.load_primary()
        pool.get("qwen3")
        pool.unload_all()
        assert not pool._backends["edge"].loaded
        assert not pool._backends["qwen3"].loaded

    def test_get_unknown_raises(self) -> None:
        pool = self._make_pool()
        pool.load_primary()
        with pytest.raises(ValueError, match="Unknown backend"):
            pool.get("nonexistent")

    def test_secondary_none(self) -> None:
        pool = self._make_pool()
        pool._secondary_name = None
        pool.load_primary()
        with pytest.raises(ValueError, match="Unknown backend"):
            pool.get("qwen3")
