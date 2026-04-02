"""chatotd — TTS daemon with Unix socket server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from chatot.backends import get_backend
from chatot.config import ChatotConfig, load_config
from chatot.protocol import Response, decode_request, encode_response

if TYPE_CHECKING:
    pass

logger = logging.getLogger("chatotd")

# ---------------------------------------------------------------------------
# Metal GPU health check
# ---------------------------------------------------------------------------


_METAL_HEALTH_CHECK = (
    "import mlx.core as mx; "
    "a = mx.random.normal((64, 64)); "
    "b = mx.random.normal((64, 64)); "
    # mx.eval triggers Metal kernel compilation for matmul — the same kind
    # of complex kernel that TTS inference needs.  A trivial mx.add would
    # pass even when MTLCompilerService is dead.
    "mx.eval(a @ b); "
    "print('ok')"
)


def _check_metal_health() -> bool:
    """Test whether the Metal GPU compiler service is responsive.

    Uses subprocess to run MLX operations in a child process, avoiding MLX
    initialization in the current process (which would create Metal GPU
    threads that deadlock after fork).

    Tests both random init AND matmul — the latter requires
    MTLCompilerService to compile complex kernels, which is exactly what
    fails after a SIGKILL'd daemon leaves zombie Metal threads.
    """
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-c", _METAL_HEALTH_CHECK],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0 and b"ok" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def _prompt_metal_recovery() -> None:
    """Interactively guide the user through recovering MTLCompilerService.

    Only called in foreground mode (before daemonize).
    """
    print(
        "\n"
        "ERROR: Metal GPU compiler service (MTLCompilerService) is not responding.\n"
        "       This prevents MLX from loading models on the GPU.\n"
        "\n"
        "This typically happens after macOS sleep/wake or prolonged uptime.\n"
    )

    # Option 1: sudo killall
    answer = input(
        "Try restarting MTLCompilerService?\n"
        "  This will run: sudo killall -9 MTLCompilerService\n"
        "  [y/N] "
    ).strip().lower()

    if answer == "y":
        import subprocess
        ret = subprocess.run(
            ["sudo", "killall", "-9", "MTLCompilerService"],
            capture_output=True,
            text=True,
        )
        if ret.returncode == 0:
            print("MTLCompilerService restarted. Rechecking...")
            import time
            time.sleep(2)
            if _check_metal_health():
                print("Metal GPU is healthy now. Continuing startup.\n")
                return
            else:
                print("Still not responding after restart.\n")
        else:
            stderr = ret.stderr.strip()
            print(f"Command failed: {stderr}\n")

    # Option 2: lock/unlock
    print(
        "Alternative: lock and unlock your Mac (Ctrl+Cmd+Q), then try again.\n"
        "This resets the GPU session and usually fixes the issue.\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_pid(pid_file: Path) -> int | None:
    """Read PID from file, return None if missing or invalid."""
    try:
        return int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with *pid* is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _remove_stale_socket(config: ChatotConfig) -> None:
    """Remove socket file if the owning process is dead."""
    if not config.socket_path.exists():
        return
    pid = _read_pid(config.pid_file)
    if pid is None or not _is_pid_alive(pid):
        logger.info("Removing stale socket %s", config.socket_path)
        config.socket_path.unlink(missing_ok=True)
        config.pid_file.unlink(missing_ok=True)


def _write_pid(pid_file: Path) -> None:
    """Write current PID to file."""
    pid_file.write_text(str(os.getpid()))


def _cleanup(config: ChatotConfig) -> None:
    """Remove socket and PID files."""
    config.socket_path.unlink(missing_ok=True)
    config.pid_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Daemonize
# ---------------------------------------------------------------------------


def _spawn_daemon(config_path: Path | None) -> None:
    """Spawn the daemon as a fresh subprocess (avoids fork + Metal deadlock).

    Classic double-fork corrupts the XPC connection to MTLCompilerService
    because forked children inherit invalidated Mach port state.  Instead,
    we launch a brand-new Python interpreter with --no-daemon, redirecting
    its stdio to the log file and detaching it from the terminal.
    """
    import subprocess

    config = load_config(config_path)
    log_file = config.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "chatot.daemon", "--no-daemon"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])

    log_fd = open(str(log_file), "a")
    devnull = open(os.devnull, "r")

    subprocess.Popen(
        cmd,
        stdin=devnull,
        stdout=log_fd,
        stderr=log_fd,
        start_new_session=True,  # detach from terminal (like setsid)
    )
    # Parent continues — caller will tail the log.


# ---------------------------------------------------------------------------
# Async server
# ---------------------------------------------------------------------------


_SPEAK_TIMEOUT_S = 30  # max seconds for a single speak() call
_SHUTDOWN_LOCK_TIMEOUT_S = 3  # max seconds to wait for speak_lock during shutdown


async def _send_response(writer: asyncio.StreamWriter, resp: Response) -> None:
    """Write response, drain, and close the writer."""
    try:
        writer.write(encode_response(resp))
        await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


class _SpeakExecutorPool:
    """Manages a dedicated ThreadPoolExecutor for speak() calls.

    After a speak() timeout, the underlying thread is zombied (Python cannot
    kill threads).  This class replaces the poisoned executor with a fresh one
    so subsequent requests aren't permanently blocked (livelock prevention).

    It also tracks zombie threads so shutdown can wait for them before calling
    backend.unload(), preventing use-after-free on the model.
    """

    def __init__(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatot-speak")
        self._zombie_executors: list = []

    @property
    def executor(self):
        return self._executor

    def replace_after_timeout(self) -> None:
        """Retire the current executor and create a fresh one."""
        from concurrent.futures import ThreadPoolExecutor
        logger.warning("Replacing speak executor (previous thread is zombied)")
        self._zombie_executors.append(self._executor)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatot-speak")

    def shutdown(self) -> None:
        """Shutdown all executors.  Waits for zombie threads to finish so
        backend.unload() doesn't race with a still-running speak()."""
        self._executor.shutdown(wait=False)
        for zombie in self._zombie_executors:
            # Give zombie threads a bounded wait — they hold the model reference.
            zombie.shutdown(wait=True, cancel_futures=True)
        self._zombie_executors.clear()


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    backend: object,
    speak_lock: asyncio.Lock,
    loop: asyncio.AbstractEventLoop,
    pool: _SpeakExecutorPool,
) -> None:
    """Handle a single client connection."""
    peer = writer.get_extra_info("peername") or "unknown"
    try:
        data = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not data:
            # Client disconnected without sending anything — still must close.
            await _send_response(writer, Response("error", "empty request"))
            return

        req = decode_request(data)
        logger.info("Request from %s: %r (lang=%s)", peer, req.text[:80], req.lang)

        kwargs: dict[str, str] = {}
        if req.lang is not None:
            kwargs["language"] = req.lang

        async with speak_lock:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(pool.executor, partial(backend.speak, req.text, **kwargs)),  # type: ignore[arg-type]
                    timeout=_SPEAK_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning("speak() timed out for %s after %ds", peer, _SPEAK_TIMEOUT_S)
                # The executor thread is zombied — replace it so subsequent
                # requests don't queue behind the stuck thread (livelock fix).
                pool.replace_after_timeout()
                await _send_response(writer, Response("error", f"speak timeout ({_SPEAK_TIMEOUT_S}s)"))
                return

        await _send_response(writer, Response("ok"))
        return  # writer already closed

    except asyncio.TimeoutError:
        logger.warning("Client %s read timed out", peer)
        resp = Response("error", "read timeout")
    except ValueError as exc:
        logger.warning("Bad request from %s: %s", peer, exc)
        resp = Response("error", str(exc))
    except Exception as exc:
        logger.exception("Error handling request from %s", peer)
        resp = Response("error", str(exc))

    await _send_response(writer, resp)


async def _serve(config: ChatotConfig) -> None:
    """Main async entry: load backend, run socket server, handle signals."""
    # Load backend --------------------------------------------------------
    backend = get_backend(config.backend)
    logger.info(
        "Loading backend %r (model=%s, voice=%s)",
        config.backend,
        config.model,
        config.voice,
    )
    backend.load(
        {
            "model": config.model,
            "voice": config.voice,
            "language": config.language,
        }
    )
    logger.info("Backend loaded successfully")

    # Write PID early so the parent's _tail_log can check we're alive
    # during the warmup (which blocks for several seconds).
    _write_pid(config.pid_file)

    # Warmup — run a real TTS call to force Metal to compile every kernel
    # the model needs.  After this, all pipeline states are cached
    # in-process and MTLCompilerService is no longer needed.
    logger.info("Running TTS warmup (compiling Metal kernels)…")
    try:
        backend.speak("Chatot is now ready to talk.", language=config.language)
        logger.info("TTS warmup completed — all Metal kernels compiled")
    except Exception:
        logger.critical(
            "TTS warmup failed — Metal GPU compiler service is likely dead.\n"
            "Fix: lock/unlock your Mac (Ctrl+Cmd+Q), or run:\n"
            "  sudo killall -9 MTLCompilerService",
            exc_info=True,
        )
        _cleanup(config)
        sys.exit(1)

    # Shutdown machinery --------------------------------------------------
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    # Start Unix socket server -------------------------------------------
    speak_lock = asyncio.Lock()
    pool = _SpeakExecutorPool()

    handler = partial(
        _handle_client,
        backend=backend,
        speak_lock=speak_lock,
        loop=loop,
        pool=pool,
    )

    server = await asyncio.start_unix_server(handler, path=str(config.socket_path))
    # Make socket accessible
    config.socket_path.chmod(0o660)

    logger.info("Listening on %s (pid %d)", config.socket_path, os.getpid())

    # Wait for shutdown ---------------------------------------------------
    try:
        await shutdown_event.wait()
    finally:
        logger.info("Shutting down…")
        server.close()
        await server.wait_closed()

        # Wait for in-flight speak to finish — bounded so shutdown doesn't
        # hang indefinitely if speak() is stuck.  _cmd_stop has its own
        # SIGKILL escalation, but a manual `kill <pid>` would block here.
        try:
            await asyncio.wait_for(speak_lock.acquire(), timeout=_SHUTDOWN_LOCK_TIMEOUT_S)
            speak_lock.release()
        except asyncio.TimeoutError:
            logger.warning(
                "speak_lock still held after %ds — proceeding with forced shutdown",
                _SHUTDOWN_LOCK_TIMEOUT_S,
            )

        # Wait for zombie threads before unloading the model they reference.
        pool.shutdown()
        backend.unload()
        logger.info("Backend unloaded")
        _cleanup(config)
        logger.info("Clean exit")


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def _cmd_stop(config: ChatotConfig) -> int:
    """Send SIGTERM to running daemon and wait for it to exit."""
    import time

    pid = _read_pid(config.pid_file)
    if pid is None or not _is_pid_alive(pid):
        # Process is gone — clean up stale files if any.
        _cleanup(config)
        print("chatotd is not running")
        return 1

    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to chatotd (pid {pid}), waiting…")

    # Poll until the process exits (up to 5 seconds).
    for _ in range(50):
        time.sleep(0.1)
        if not _is_pid_alive(pid):
            _cleanup(config)  # belt-and-suspenders
            print("chatotd stopped")
            return 0

    # Escalate to SIGKILL.
    print("Daemon still running — sending SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    for _ in range(20):
        time.sleep(0.1)
        if not _is_pid_alive(pid):
            break

    _cleanup(config)
    print("chatotd killed")
    return 0


def _cmd_status(config: ChatotConfig) -> int:
    """Print daemon status."""
    pid = _read_pid(config.pid_file)
    if pid is None:
        print("chatotd is not running (no PID file)")
        return 1
    if not _is_pid_alive(pid):
        print(f"chatotd is not running (stale PID {pid})")
        return 1
    print(f"chatotd is running (pid {pid})")
    if config.socket_path.exists():
        print(f"  socket: {config.socket_path}")
    return 0


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(config: ChatotConfig, *, foreground: bool) -> None:
    """Configure logging for the daemon.

    When the daemon is spawned via ``_spawn_daemon`` (subprocess with
    stdout/stderr → log file), we skip the FileHandler to avoid duplicate
    lines (the subprocess's stdio already goes to the log file).
    """
    spawned_daemon = not foreground and not sys.stderr.isatty()

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.log_level, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if spawned_daemon:
        # stdout/stderr already point to the log file (set by _spawn_daemon).
        # Only add a StreamHandler so logging goes to the same fd.
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
        return

    # Normal foreground or true --no-daemon mode: log to file.
    file_handler = logging.FileHandler(str(config.log_file))
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # In foreground mode, also log to stdout
    if foreground:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _tail_log(config: ChatotConfig, *, until_ready: bool = False) -> int:
    """Tail the daemon log file.

    If *until_ready* is True, exits with 0 once the backend is loaded or 1
    on error/timeout.  Otherwise tails forever (Ctrl-C to stop).

    Returns exit code.
    """
    import time

    log_path = config.log_file
    # Wait for the log file to appear (daemon may be starting up).
    for _ in range(50):
        if log_path.exists():
            break
        time.sleep(0.1)
    else:
        print(f"Timeout waiting for log file: {log_path}", file=sys.stderr)
        return 1

    # Snapshot the file size *before* opening — if the daemon writes
    # between stat() and open(), we'll still read those lines because
    # we seek to the pre-open size, not the post-open EOF.
    start_pos = log_path.stat().st_size if until_ready else 0

    try:
        with open(log_path) as f:
            f.seek(start_pos)

            no_output_since = time.monotonic()
            deadline = time.monotonic() + 120 if until_ready else float("inf")
            while time.monotonic() < deadline:
                line = f.readline()
                if not line:
                    if until_ready:
                        # Check if daemon crashed before writing PID file.
                        pid = _read_pid(config.pid_file)
                        if pid is not None and not _is_pid_alive(pid):
                            print("Daemon exited unexpectedly", file=sys.stderr)
                            return 1
                        if pid is None and (time.monotonic() - no_output_since) > 10:
                            print("Daemon appears to have crashed (no PID file, no log output)", file=sys.stderr)
                            return 1
                    time.sleep(0.1)
                    continue

                no_output_since = time.monotonic()

                sys.stdout.write(line)
                sys.stdout.flush()

                if until_ready:
                    if "Listening on" in line:
                        return 0
                    if "FATAL:" in line or "Fatal error" in line or "CRITICAL" in line:
                        return 1

    except KeyboardInterrupt:
        return 0

    print("Timeout waiting for daemon to become ready", file=sys.stderr)
    return 1


def _cmd_logs(config: ChatotConfig) -> int:
    """Tail daemon log file (Ctrl-C to stop)."""
    log_path = config.log_file
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1
    return _tail_log(config)


def main() -> None:
    """Entry point for ``chatotd`` command."""
    parser = argparse.ArgumentParser(
        prog="chatotd",
        description="chatot TTS daemon",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        choices=["logs", "stop", "status"],
        help="Subcommand: logs (tail log), stop, status",
    )
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Run in foreground (don't daemonize)",
    )
    # Keep --stop/--status as hidden aliases for backwards compat.
    parser.add_argument("--stop", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--status", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config early — needed by all subcommands.
    config = load_config(args.config)

    # Subcommands (positional or --flag).
    if args.command == "stop" or args.stop:
        sys.exit(_cmd_stop(config))
    if args.command == "status" or args.status:
        sys.exit(_cmd_status(config))
    if args.command == "logs":
        sys.exit(_cmd_logs(config))

    # Guard: don't start if already running.
    _remove_stale_socket(config)
    pid = _read_pid(config.pid_file)
    if pid is not None and _is_pid_alive(pid):
        print(f"chatotd is already running (pid {pid})", file=sys.stderr)
        sys.exit(1)

    # Check Metal GPU health before loading anything.
    print("Checking Metal GPU health...")
    if not _check_metal_health():
        _prompt_metal_recovery()
    else:
        print("Metal GPU is healthy.\n")

    # Daemonize unless --no-daemon.
    foreground = args.no_daemon
    if not foreground:
        # Spawn a fresh subprocess instead of fork() — fork corrupts the
        # XPC connection to MTLCompilerService (Apple Metal bug).
        _spawn_daemon(args.config)
        # Parent tails the log until the daemon is ready.
        sys.exit(_tail_log(config, until_ready=True))

    # Set up logging (foreground mode, or we ARE the spawned daemon).
    _setup_logging(config, foreground=foreground)

    # Install crash handler — in daemon mode, unhandled exceptions must be
    # written to the log file directly (logging handlers may not flush).
    def _crash_handler(exc_type, exc_value, exc_tb):
        import traceback
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical("FATAL: %s", msg)
        # Also write directly to stderr (which is the log file after daemonize)
        sys.stderr.write(f"FATAL: {msg}\n")
        sys.stderr.flush()

    sys.excepthook = _crash_handler

    logger.info("chatotd starting (pid %d)", os.getpid())

    try:
        asyncio.run(_serve(config))
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Fatal error in chatotd")
        sys.stderr.flush()
    finally:
        # Belt-and-suspenders cleanup
        _cleanup(config)


if __name__ == "__main__":
    main()
