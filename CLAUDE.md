# CLAUDE.md

## Project overview

chatot-tts is a TTS CLI tool with a persistent daemon. Two entry points: `chatot` (client) and `chatotd` (daemon). macOS only (uses `afplay`).

## Source layout

```
src/chatot/
  __init__.py          # version
  client.py            # CLI client — sends text to daemon via Unix socket
  daemon.py            # daemon server — async socket server, backend pool, lifecycle
  config.py            # YAML config loading, ChatotConfig dataclass
  router.py            # language detection (script-based) and backend/voice routing
  protocol.py          # Request/Response wire format (newline-delimited JSON)
  _migrate.py          # v1 → v2 config migration (temporary)
  backends/
    __init__.py        # backend registry, get_backend() factory
    base.py            # TTSBackend ABC (load/speak/unload)
    edge.py            # EdgeBackend — edge-tts (online Microsoft TTS)
    qwen3.py           # Qwen3Backend — mlx-audio (local Apple Silicon GPU)
tests/
  test_config.py
  test_protocol.py
  test_client.py
  test_daemon.py
  test_router.py
  test_migrate.py
```

## Key conventions

- Python 3.10+, type hints throughout
- Config lives at `~/.config/chatot/config.yaml` (v2 structured YAML format)
- Client ↔ daemon use newline-delimited JSON over Unix domain socket
- Daemon runs speak() in a ThreadPoolExecutor (1 worker), serialized by asyncio.Lock
- Secondary backend (qwen3) is lazy-loaded on first use
- AudioPlayer.stop() is monkey-patched in qwen3.py to fix upstream stream leak

## Running tests

```bash
pytest
```

## Build and install

```bash
pip install -e .           # minimal (edge-tts only)
pip install -e '.[mlx]'   # with Qwen3/MLX support
```

## Important details

- `_spawn_daemon` uses subprocess.Popen (not fork) to avoid Metal/XPC deadlocks on macOS
- Qwen3Backend has a "poisoned" state — if generate_audio() fails, the backend refuses further calls until daemon restart
- `_SpeakExecutorPool` replaces the ThreadPoolExecutor after a timeout to prevent livelock from zombied threads
- Metal GPU health check runs `mlx.core` matmul in a subprocess to detect dead MTLCompilerService
- Config auto-migrates v1 (flat) → v2 (structured) format on disk via `_migrate.py`
