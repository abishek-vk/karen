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


def get_system_information(topic: str | None = None) -> SystemResult:
    """Answer a time, date or battery question, or give a short status report."""

    now = dt.datetime.now().astimezone()
    if topic == "time":
        return SystemResult(True, f"It's {now.strftime('%I:%M %p').lstrip('0')}.")
    if topic == "date":
        return SystemResult(True, f"Today is {now.strftime('%A, %B')} {now.day}, {now.year}.")
    if topic == "battery":
        return SystemResult(True, _battery() or "I can't read a battery on this computer.")

    details = [f"It's {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B')} {now.day}."]
    if psutil is not None:
        details.append(
            f"CPU is at {psutil.cpu_percent(interval=0.2):.0f} percent and memory at "
            f"{psutil.virtual_memory().percent:.0f} percent."
        )
        battery = _battery()
        if battery:
            details.append(battery)
    else:
        details.append("CPU and memory usage are unavailable because psutil is not installed.")
    details.append(f"You're running {platform.system()} {platform.release()}.")
    return SystemResult(True, " ".join(details))


def _battery() -> str | None:
    if psutil is None:
        return None
    try:
        battery = psutil.sensors_battery()
    except (AttributeError, NotImplementedError, OSError):
        return None
    if battery is None:
        return None
    state = "charging" if battery.power_plugged else "on battery"
    return f"Battery is at {battery.percent:.0f} percent and {state}."


def current_time() -> str:
    """Return the local time in a human-readable format."""

    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
