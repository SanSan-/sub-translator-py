from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_MAX_BYTES = 50 * 1024 * 1024
LOG_BACKUP_COUNT = 10


def configure_rotating_logger(
    name: str,
    log_file: Path,
    verbose: bool,
    *,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt: str | None = None,
) -> logging.Logger:
    """Настраивает логгер с выводом в консоль и во вращаемый файл."""
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


__all__ = [
    "configure_rotating_logger",
    "LOG_MAX_BYTES",
    "LOG_BACKUP_COUNT",
]