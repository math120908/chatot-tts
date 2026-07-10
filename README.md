# chatot-tts

TTS CLI tool with persistent daemon for Claude Code integration. Supports multilingual speech synthesis with automatic language detection and backend routing.

## Features

- **Daemon architecture** — `chatotd` stays resident with the TTS model loaded; `chatot` sends text via Unix socket for instant playback
- **Dual backend** — Edge TTS (Microsoft, online) as primary; Qwen3-TTS via MLX (local, Apple Silicon GPU) as secondary
- **Multilingual routing** — auto-detects CJK / Hangul / Latin scripts and picks the right voice; mixed-language text routes to Qwen3 for natural code-switching
- **Per-language voices** — configurable voice map (`en`, `zh`, `ko`, `ja`, `sw`, `mixed`)
- **Save to file** — `--save path.mp3` writes audio alongside playback

## Requirements

- Python 3.10+
- macOS (uses `afplay` for audio playback)
- For Qwen3 backend: Apple Silicon Mac with MLX

## Install

```bash
pip install -e .            # edge-tts only (default)
pip install -e '.[mlx]'     # + Qwen3-TTS via mlx-audio
```

## Quick start

```bash
chatotd                     # start daemon (daemonizes automatically)
chatot "Hello world"        # speak text
echo "你好世界" | chatot     # pipe text (auto-detects Chinese → zh voice)
chatot --lang zh "Hello"    # force language
chatot --voice vivian "Hi"  # override voice
chatot --save out.mp3 "Hi"  # save audio to file
```

## Daemon management

```bash
chatotd                     # start (daemonizes, tails log until ready)
chatotd status              # show running state and config
chatotd stop                # graceful shutdown (SIGTERM → SIGKILL)
chatotd logs                # tail daemon log
chatotd --no-daemon         # run in foreground (debug)
```

## Configuration

Config file: `~/.config/chatot/config.yaml` (auto-created on first run)

```yaml
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
  secondary: qwen3

daemon:
  socket: ~/.config/chatot/var/process/chatotd.sock
  pid_file: ~/.config/chatot/var/process/chatotd.pid
  log_file: ~/.config/chatot/var/logs/chatotd.log
  log_level: INFO
```

## Architecture

```
chatot (client)
  │  sends JSON request via Unix socket
  ▼
chatotd (daemon)
  ├─ router.py      auto-detect language → pick backend + voice
  ├─ EdgeBackend     edge-tts (online, fast, many voices)
  └─ Qwen3Backend    mlx-audio (local GPU, mixed-language)
```

- Client ↔ daemon communicate via newline-delimited JSON over Unix domain socket
- Daemon serializes speak calls through an `asyncio.Lock` + `ThreadPoolExecutor`
- Secondary backend (Qwen3) is lazy-loaded on first mixed-language request
- Metal GPU health check runs at startup; interactive recovery prompt if MTLCompilerService is unresponsive

## Language routing logic

| Input | Backend | Voice |
|---|---|---|
| Single script (e.g. English only) | primary (edge) | per-language voice |
| Japanese kana (Hiragana/Katakana, ± kanji) | primary (edge) | `ja` voice |
| Mixed scripts (e.g. English + Chinese) | secondary (qwen3) | `mixed` voice |
| `--lang zh` | primary (edge) | zh voice |
| `--lang ja` | primary (edge) | ja voice |
| `--lang sw` | primary (edge) | sw voice |
| `--lang auto` / `--lang mixed` | secondary (qwen3) | `mixed` voice |

### Language notes

- **Japanese (`ja`)** — auto-detected via kana. Pure-kanji text (no kana) is
  indistinguishable from Chinese and is treated as `zh`; force with `--lang ja`.
  The code `jp` is **invalid** — use `ja`. Passing `--lang jp` returns a clear
  error rather than silently correcting.
- **Swahili (`sw`)** — uses the Latin script, so it **cannot be auto-detected**
  (it looks like English). It must be requested explicitly with `--lang sw`.
- **Backends** — `ja` and `sw` both route to Edge TTS. (Qwen3 supports Japanese
  but not Swahili; to keep behaviour simple both use Edge.)

## License

MIT
