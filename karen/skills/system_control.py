"""Volume, media and lock-screen control through standard Windows media keys."""

from __future__ import annotations

import ctypes
import os

from .base import SkillResult

_KEYEVENTF_KEYUP = 0x0002
_KEYS = {
    "volume_up": (0xAF, 5, "Turning the volume up."),
    "volume_down": (0xAE, 5, "Turning the volume down."),
    "mute": (0xAD, 1, "Toggling mute."),
    "play_pause": (0xB3, 1, "Done."),
    "next_track": (0xB0, 1, "Skipping to the next track."),
    "previous_track": (0xB1, 1, "Going back a track."),
}
ACTIONS = frozenset({*_KEYS, "lock"})


def _press(key_code: int, times: int) -> None:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    for _ in range(times):
        user32.keybd_event(key_code, 0, 0, 0)
        user32.keybd_event(key_code, 0, _KEYEVENTF_KEYUP, 0)


def control(action: str | None) -> SkillResult:
    """Run one allowlisted media/lock action; nothing else is reachable."""

    if os.name != "nt":
        return SkillResult(False, "System controls are available on Windows only.")
    if action == "lock":
        if not ctypes.windll.user32.LockWorkStation():  # type: ignore[attr-defined]
            return SkillResult(False, "I couldn't lock the computer.")
        return SkillResult(True, "Locking your computer.")
    entry = _KEYS.get(action or "")
    if entry is None:
        return SkillResult(False, "I can change the volume, control media playback, or lock the computer.")
    key_code, times, message = entry
    try:
        _press(key_code, times)
    except OSError:
        return SkillResult(False, "That system control isn't available right now.")
    return SkillResult(True, message)
