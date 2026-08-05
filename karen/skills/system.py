"""Read-only system information skill."""

from __future__ import annotations

import datetime as dt
import platform
from dataclasses import dataclass

try:
    import psutil
except ImportError:  # pragma: no cover - optional fallback for development
    psutil = None


@dataclass(frozen=True, slots=True)
class SystemResult:
    success: bool
    message: str


def get_system_information() -> SystemResult:
    """Return OS, time, and best-effort CPU/RAM information without changing state."""

    details = [f"Operating system: {platform.platform()}", f"Time: {current_time()}"]
    if psutil is not None:
        details.extend(
            [
                f"CPU usage: {psutil.cpu_percent(interval=0.1):.1f}%",
                f"Memory usage: {psutil.virtual_memory().percent:.1f}%",
            ]
        )
    else:
        details.append("CPU/RAM usage: unavailable (install psutil)")
    return SystemResult(True, "\n".join(details))


def current_time() -> str:
    """Return the local time in a human-readable format."""

    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
