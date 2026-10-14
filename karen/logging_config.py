"""Application logging setup with local file output only."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    """Log warnings to the console and everything at ``level`` to a capped local file."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "karen.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
    except OSError:
        return
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    # Third-party libraries are noisy at INFO.
    for name in ("urllib3", "faster_whisper", "comtypes"):
        logging.getLogger(name).setLevel(logging.WARNING)
