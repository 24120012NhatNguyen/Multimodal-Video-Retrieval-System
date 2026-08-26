import logging
import os
import sys
from typing import Optional


DEFAULT_LOG_LEVEL = "INFO"
SUPPORTED_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _get_log_level() -> int:
    """Resolve log level from environment, defaulting to INFO."""
    raw = (os.environ.get("LOG_LEVEL") or DEFAULT_LOG_LEVEL).strip().upper()
    return getattr(logging, raw, logging.INFO)


def _build_handlers(
    log_level: int,
    log_file: Optional[str],
) -> list[logging.Handler]:
    """Build list of handlers: always console, optionally file."""
    formatter = logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    handlers: list[logging.Handler] = []

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    handlers.append(console)

    # Optional file handler
    if log_file and log_file.strip():
        try:
            file_handler = logging.FileHandler(
                log_file.strip(),
                encoding="utf-8",
                mode="a",
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except (OSError, PermissionError) as e:
            # Fallback: log to stderr that file logging failed
            sys.stderr.write(f"[logger_config] Could not open log file: {e}\n")

    return handlers


def _configure_root_logger() -> None:
    """Configure the root logger once. Idempotent after first call."""
    global _configured
    if _configured:
        return

    log_level = _get_log_level()
    log_file = os.environ.get("LOG_FILE")
    handlers = _build_handlers(log_level, log_file)

    root = logging.getLogger()
    root.setLevel(log_level)
    # Avoid duplicate handlers when get_logger is called from multiple modules
    for h in handlers:
        root.addHandler(h)
    root.debug(
        "Logging configured: level=%s, file=%s",
        logging.getLevelName(log_level),
        log_file or "none",
    )

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given module name.

    The root logger is configured on first call (level and handlers from
    LOG_LEVEL and LOG_FILE). Subsequent calls only create/return child
    loggers and do not add duplicate handlers.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    _configure_root_logger()
    return logging.getLogger(name)
