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
# Daemonize (double-fork)
# ---------------------------------------------------------------------------


def _daemonize(log_file: Path) -> None:
    """Classic Unix double-fork to detach from terminal.

    After this function returns (in the grandchild), stdin is /dev/null
    and stdout/stderr point to *log_file*.
    """
    # First fork — parent exits, child gets adopted by init.
    pid = os.fork()
    if pid > 0:
        # Parent: exit immediately.
        os._exit(0)

    # Child: become session leader.
    os.setsid()

    # Second fork — prevent accidental terminal acquisition.
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    # Grandchild: redirect file descriptors.
    sys.stdout.flush()
    sys.stderr.flush()

    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, sys.stdin.fileno())
    os.close(devnull)

    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)


# ---------------------------------------------------------------------------
# Async server
# ---------------------------------------------------------------------------


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    backend: object,
    speak_lock: asyncio.Lock,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Handle a single client connection."""
    peer = writer.get_extra_info("peername") or "unknown"
    try:
        data = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not data:
            return

        req = decode_request(data)
        logger.info("Request from %s: %r (lang=%s)", peer, req.text[:80], req.lang)

        kwargs: dict[str, str] = {}
        if req.lang is not None:
            kwargs["language"] = req.lang

        async with speak_lock:
            # speak() is blocking — run in thread executor
            await loop.run_in_executor(None, partial(backend.speak, req.text, **kwargs))  # type: ignore[arg-type]

        writer.write(encode_response(Response("ok")))

    except asyncio.TimeoutError:
        logger.warning("Client %s timed out", peer)
        writer.write(encode_response(Response("error", "read timeout")))
    except ValueError as exc:
        logger.warning("Bad request from %s: %s", peer, exc)
        writer.write(encode_response(Response("error", str(exc))))
    except Exception as exc:
        logger.exception("Error handling request from %s", peer)
        writer.write(encode_response(Response("error", str(exc))))
    finally:
        try:
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


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

    # Write PID -----------------------------------------------------------
    _write_pid(config.pid_file)

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

    handler = partial(
        _handle_client,
        backend=backend,
        speak_lock=speak_lock,
        loop=loop,
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

        # Wait for in-flight requests (lock holders) to finish
        async with speak_lock:
            pass

        backend.unload()
        logger.info("Backend unloaded")
        _cleanup(config)
        logger.info("Clean exit")


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def _cmd_stop(config: ChatotConfig) -> int:
    """Send SIGTERM to running daemon."""
    pid = _read_pid(config.pid_file)
    if pid is None or not _is_pid_alive(pid):
        print("chatotd is not running")
        return 1
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to chatotd (pid {pid})")
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
    """Configure logging for the daemon."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.log_level, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Always log to file
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


def main() -> None:
    """Entry point for ``chatotd`` command."""
    parser = argparse.ArgumentParser(
        prog="chatotd",
        description="chatot TTS daemon",
    )
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Run in foreground (don't daemonize)",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop running daemon",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show daemon status",
    )
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

    # Subcommands that don't start the server.
    if args.stop:
        sys.exit(_cmd_stop(config))
    if args.status:
        sys.exit(_cmd_status(config))

    # Guard: don't start if already running.
    _remove_stale_socket(config)
    pid = _read_pid(config.pid_file)
    if pid is not None and _is_pid_alive(pid):
        print(f"chatotd is already running (pid {pid})", file=sys.stderr)
        sys.exit(1)

    # Daemonize unless --no-daemon.
    foreground = args.no_daemon
    if not foreground:
        _daemonize(config.log_file)

    # Set up logging *after* potential daemonize (fd redirects).
    _setup_logging(config, foreground=foreground)

    logger.info("chatotd starting (pid %d)", os.getpid())

    try:
        asyncio.run(_serve(config))
    except KeyboardInterrupt:
        pass
    finally:
        # Belt-and-suspenders cleanup
        _cleanup(config)


if __name__ == "__main__":
    main()
