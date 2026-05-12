"""Logging configuration for apple-basefm.

Provides:
  - OSLogHandler: routes Python logging to the macOS Unified Logging System
    (os_log) via ctypes. No-op NullHandler on non-Darwin platforms.
  - configure_logging(): one-call setup for structured, levelled logging with
    optional OSLog and OpenTelemetry integration.

Structured log fields
---------------------
All semantic log calls in this package emit an ``extra`` dict so that log
aggregators can filter without regex. The canonical field names are:

  backend       "foundation" | "mlx"
  model         model identifier string
  elapsed_ms    wall-clock time in milliseconds (float)
  prompt_tokens token count for the prompt (int)
  completion_tokens  token count for the completion (int)
  correlation_id    opaque caller-provided trace ID (str), passed through
                    kwargs["correlation_id"] on __call__ if present

OSLog
-----
On macOS 10.12+ (Darwin), ``OSLogHandler`` writes log records to the system's
Unified Logging System (Console.app, ``log stream``, ``log show``). The
subsystem is ``com.apple-basefm``; the category matches the module name from
the logger hierarchy.

The C ABI used here (``os_log_create`` / ``os_log_with_type``) is stable
since macOS 10.12 and is part of ``libSystem.B.dylib``.

On Linux, Windows, and any platform where the ABI is unavailable, the handler
degrades silently to ``logging.NullHandler``.

Usage::

    from apple_basefm._logging import configure_logging
    configure_logging(level="DEBUG", oslog=True)
"""
from __future__ import annotations

import logging
import platform
import sys
from typing import Any

__all__ = ["OSLogHandler", "configure_logging"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OS_LOG level constants (macOS os/log.h)
# ---------------------------------------------------------------------------
_OS_LOG_TYPE_DEBUG: int = 0x02
_OS_LOG_TYPE_INFO: int = 0x01
_OS_LOG_TYPE_DEFAULT: int = 0x00
_OS_LOG_TYPE_ERROR: int = 0x10
_OS_LOG_TYPE_FAULT: int = 0x11

_LEVEL_TO_OS_LOG: dict[int, int] = {
    logging.DEBUG: _OS_LOG_TYPE_DEBUG,
    logging.INFO: _OS_LOG_TYPE_INFO,
    logging.WARNING: _OS_LOG_TYPE_DEFAULT,
    logging.ERROR: _OS_LOG_TYPE_ERROR,
    logging.CRITICAL: _OS_LOG_TYPE_FAULT,
}

_SUBSYSTEM = b"com.apple-basefm"


def _build_oslog_handler() -> logging.Handler:
    """Attempt to load libSystem and return a live OSLogHandler.

    Returns a NullHandler on any failure (non-Darwin, missing symbols, etc.).
    """
    if platform.system() != "Darwin":
        return logging.NullHandler()

    try:
        import ctypes
        import ctypes.util

        lib_path = ctypes.util.find_library("System")
        if lib_path is None:
            return logging.NullHandler()

        lib = ctypes.CDLL(lib_path, use_errno=True)

        # os_log_t os_log_create(const char *subsystem, const char *category)
        os_log_create = lib.os_log_create
        os_log_create.restype = ctypes.c_void_p
        os_log_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

        # void _os_log_impl(void *dso, os_log_t log, os_log_type_t type,
        #                   const char *format, uint8_t *buf, uint32_t size)
        # The public C macro os_log_with_type expands to _os_log_impl.
        # We call it directly with a static "%{public}s" format string so that
        # the message is always visible in Console.app without privacy redaction.
        os_log_impl = lib._os_log_impl
        os_log_impl.restype = None
        os_log_impl.argtypes = [
            ctypes.c_void_p,   # dso (NULL → default)
            ctypes.c_void_p,   # os_log_t
            ctypes.c_uint8,    # os_log_type_t
            ctypes.c_char_p,   # format string (static "%{public}s")
            ctypes.c_char_p,   # buf (message bytes encoded as the single arg)
            ctypes.c_uint32,   # size
        ]

    except Exception:
        return logging.NullHandler()

    class _OSLogHandler(logging.Handler):
        """Python logging handler that writes to macOS Unified Logging.

        Each log record is routed to a per-category os_log object. The category
        is derived from the logger name (the part after the last dot, or the
        full name if there is no dot).

        Log levels map as follows:
          DEBUG    → OS_LOG_TYPE_DEBUG  (visible with ``log stream --level debug``)
          INFO     → OS_LOG_TYPE_INFO
          WARNING  → OS_LOG_TYPE_DEFAULT (always on)
          ERROR    → OS_LOG_TYPE_ERROR
          CRITICAL → OS_LOG_TYPE_FAULT
        """

        def __init__(self) -> None:
            super().__init__()
            self._logs: dict[bytes, ctypes.c_void_p] = {}

        def _get_log(self, category: bytes) -> ctypes.c_void_p:
            if category not in self._logs:
                self._logs[category] = ctypes.c_void_p(
                    os_log_create(_SUBSYSTEM, category)
                )
            return self._logs[category]

        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record).encode("utf-8", errors="replace")
                # Derive category from the logger name.
                name_parts = record.name.rsplit(".", 1)
                category = name_parts[-1].encode("utf-8", errors="replace")
                log_obj = self._get_log(category)
                os_type = _LEVEL_TO_OS_LOG.get(record.levelno, _OS_LOG_TYPE_DEFAULT)
                # Format: "%{public}s" so the message is always visible.
                fmt = b"%{public}s"
                os_log_impl(
                    None,                 # dso
                    log_obj,              # os_log_t
                    os_type,              # level
                    fmt,                  # format string
                    msg,                  # message (single positional arg)
                    ctypes.c_uint32(len(msg)),
                )
            except Exception:
                self.handleError(record)

    return _OSLogHandler()


class OSLogHandler(logging.Handler):
    """Forwards log records to macOS Unified Logging (os_log).

    On macOS 10.12+, records are written to the system log and visible in
    Console.app and via ``log stream --predicate 'subsystem == "com.apple-basefm"'``.

    On Linux, Windows, and any platform where the ABI is unavailable, this
    handler is a no-op (it delegates to NullHandler internally).

    Instantiate once and add to the root or package logger::

        handler = OSLogHandler()
        logging.getLogger("apple_basefm").addHandler(handler)
    """

    def __init__(self) -> None:
        super().__init__()
        self._delegate = _build_oslog_handler()
        self._delegate.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def setFormatter(self, fmt: logging.Formatter | None) -> None:  # type: ignore[override]
        super().setFormatter(fmt)
        if fmt is not None:
            self._delegate.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        self._delegate.emit(record)

    def createLock(self) -> None:
        super().createLock()
        if hasattr(self._delegate, "createLock"):
            self._delegate.createLock()


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

def configure_logging(
    level: str | int = "INFO",
    *,
    structured: bool = True,
    oslog: bool = False,
    stream: Any = None,
) -> None:
    """Configure apple-basefm package logging in one call.

    Sets the log level, optionally adds a stream handler with a structured
    formatter, and optionally installs OSLogHandler for macOS system log
    integration.

    Args:
        level: Log level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``) or an integer constant from
            the ``logging`` module. Default: ``"INFO"``.
        structured: If True (default), the stream handler uses a formatter
            that includes structured extra fields when present (correlation_id,
            backend, model, elapsed_ms, prompt_tokens, completion_tokens,
            total_tokens).
        oslog: If True, install OSLogHandler. On macOS 10.12+ this routes
            records to the Unified Logging System (Console.app). No-op on
            other platforms. Default: False.
        stream: Stream for the stream handler. Defaults to ``sys.stderr``.
            Pass None to skip adding a stream handler entirely.

    Example::

        from apple_basefm._logging import configure_logging
        configure_logging(level="DEBUG", oslog=True)
    """
    pkg_logger = logging.getLogger("apple_basefm")

    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"configure_logging: invalid level {level!r}")
        level = numeric_level

    pkg_logger.setLevel(level)

    if stream is not None:
        handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
        if structured:
            handler.setFormatter(_StructuredFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
            )
        pkg_logger.addHandler(handler)

    if oslog:
        oslog_handler = OSLogHandler()
        oslog_handler.setLevel(level)
        pkg_logger.addHandler(oslog_handler)


# ---------------------------------------------------------------------------
# Structured formatter
# ---------------------------------------------------------------------------

class _StructuredFormatter(logging.Formatter):
    """Formatter that appends structured extra fields to each log line.

    Fields emitted when present in ``record.__dict__``:
      correlation_id, backend, model, elapsed_ms,
      prompt_tokens, completion_tokens, total_tokens
    """

    _STRUCTURED_FIELDS = (
        "correlation_id",
        "backend",
        "model",
        "elapsed_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for field in self._STRUCTURED_FIELDS:
            value = record.__dict__.get(field)
            if value is not None:
                extras.append(f"{field}={value!r}")
        if extras:
            return f"{base} [{', '.join(extras)}]"
        return base
