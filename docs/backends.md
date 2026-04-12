# TTS Backends

chatot-tts supports multiple TTS backends. Choose one in `~/.config/chatot/config.yaml`:

```yaml
backend: edge    # or "qwen3"
```

## Backend Comparison

| | Edge TTS | Qwen3 (MLX) |
|---|---|---|
| **Engine** | Microsoft Edge online TTS | Qwen3-TTS via mlx-audio |
| **Runs on** | Any Mac (or Linux) | Apple Silicon only |
| **GPU required** | No | Yes (Metal GPU) |
| **Internet required** | Yes | No |
| **Install size** | ~1 MB (`pip install edge-tts`) | ~2 GB (model weights + mlx-audio) |
| **Startup time** | ~2 seconds | ~10-30 seconds (model load + Metal kernel compile) |
| **Latency per request** | ~2-3 seconds (network round-trip) | ~1-2 seconds (local inference) |
| **Voice quality** | High (Azure Neural voices) | High (Qwen3 neural model) |
| **Voice variety** | 100+ voices across 50+ languages | Custom voices via model config |
| **Stability** | Stable (no GPU dependency) | Prone to Metal GPU crashes (see below) |
| **Cost** | Free (unofficial API) | Free (local) |

## Recommended: Edge TTS

For most users, **Edge TTS is the recommended backend**. It works reliably on any machine with internet access, requires no GPU, and starts almost instantly.

```yaml
backend: edge
voice: en-US-AvaNeural
language: en
```

### Popular Edge voices

| Voice | Gender | Notes |
|---|---|---|
| `en-US-AvaNeural` | Female | Default. Natural and clear. |
| `en-US-EmmaNeural` | Female | Warm tone. |
| `en-US-JennyNeural` | Female | Conversational. |
| `en-US-BrianNeural` | Male | Clear and natural. |
| `en-US-AndrewNeural` | Male | Calm, professional. |

Full list: `edge-tts --list-voices`

### Edge TTS caveats

- **Unofficial API** -- `edge-tts` reverse-engineers the Microsoft Edge browser's WebSocket TTS endpoint. There is no official API key or terms of service. Microsoft could change or block it at any time, though the community (`rany2/edge-tts`) has historically patched breakages quickly.
- **Requires internet** -- audio is synthesized server-side. No offline fallback.

## Qwen3 (MLX) -- Local inference on Apple Silicon

If you need fully offline TTS and have an Apple Silicon Mac, Qwen3 via MLX is an option:

```yaml
backend: qwen3
model: mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
voice: vivian
language: en
```

Install the MLX dependency:

```bash
pip install chatot-tts[mlx]
```

### Why Metal GPU is unstable

The Qwen3 backend runs on Apple's Metal GPU via the MLX framework. In practice, this introduces several stability issues that make it unsuitable as a long-running daemon:

**1. MTLCompilerService crashes**

Metal GPU operations depend on `MTLCompilerService`, a system daemon that compiles GPU shader/kernel code. This service can become unresponsive after:

- macOS sleep/wake cycles
- Prolonged system uptime
- A previous chatotd process being killed (SIGKILL)

When this happens, any MLX operation hangs indefinitely. Recovery requires either:
- `sudo killall -9 MTLCompilerService` (needs admin password)
- Locking and unlocking the Mac (Ctrl+Cmd+Q)

**2. fork() corrupts Metal state**

Standard Unix daemonization uses `fork()`, which corrupts the XPC connection to `MTLCompilerService`. Forked child processes inherit invalidated Mach port state, causing deadlocks. chatot works around this by spawning a fresh Python subprocess instead of forking, but this adds complexity.

**3. GPU thread zombies**

If the daemon is killed abruptly (SIGKILL, system crash, OOM), Metal GPU threads can become zombies that block future GPU access system-wide. This isn't just a chatot problem -- it affects all Metal/MLX applications until the GPU session is reset.

**4. Backend poisoning**

When a Metal GPU error occurs mid-inference, the MLX model state can become corrupted. chatot detects this and "poisons" the backend, refusing further requests until the daemon is restarted. This is a safety measure, but it means a single GPU glitch takes down the entire TTS service.

### Summary

Metal GPU issues are not bugs in chatot -- they are fundamental limitations of running long-lived GPU workloads on macOS. Apple's Metal stack was designed for interactive graphics (games, UI rendering), not persistent inference daemons. Until Apple improves Metal's resilience for ML workloads, **Edge TTS is the more reliable choice for a daemon that needs to stay up**.
