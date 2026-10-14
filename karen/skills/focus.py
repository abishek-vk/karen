"""Bring whatever Karen just opened to the front.

Windows keeps windows launched by a background process (Karen's server) behind
the window the user is looking at, which is Karen's own fullscreen interface,
so apps, folders and web pages "opened" fine but stayed hidden. After launching,
we look for the new window (or the one whose title names what was opened) and
raise it.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)

_KAREN_TITLE = "Karen"  # Title of Karen's own interface window; never raised by this helper.
_SW_RESTORE, _SW_SHOW = 9, 5
_VK_MENU, _KEYEVENTF_KEYUP = 0x12, 0x0002


def _user32():  # noqa: ANN202 - ctypes handle
    return ctypes.windll.user32 if sys.platform == "win32" else None  # type: ignore[attr-defined]


def _windows() -> list[tuple[int, str]]:
    """Visible top-level windows with a title, front to back."""

    user32 = _user32()
    if user32 is None:
        return []
    found: list[tuple[int, str]] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def visit(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd) and not user32.GetWindow(hwnd, 4):  # GW_OWNER: skip dialogs/tooltips
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                title = buffer.value.strip()
                if title and title != _KAREN_TITLE and title != "Program Manager":
                    found.append((int(hwnd), title))
        return True

    user32.EnumWindows(callback_type(visit), 0)
    return found


def snapshot() -> set[int]:
    """Handles of the windows that exist before something is launched."""

    return {hwnd for hwnd, _title in _windows()}


def raise_window(hwnd: int) -> bool:
    user32 = _user32()
    if user32 is None:
        return False
    user32.ShowWindow(hwnd, _SW_RESTORE if user32.IsIconic(hwnd) else _SW_SHOW)
    # Tapping Alt lets a background process change the foreground window.
    user32.keybd_event(_VK_MENU, 0, 0, 0)
    user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
    user32.BringWindowToTop(hwnd)
    return bool(user32.SetForegroundWindow(hwnd))


def find_window(before: set[int], hints: Iterable[str]) -> int | None:
    """Prefer a new window that matches a hint, then any window titled with a hint, then any new window."""

    keys = [hint.lower() for hint in hints if hint and hint.strip()]
    windows = _windows()
    new = [(hwnd, title) for hwnd, title in windows if hwnd not in before]
    for pool in (
        [item for item in new if any(key in item[1].lower() for key in keys)],
        [item for item in windows if any(key in item[1].lower() for key in keys)],
        new,
    ):
        if pool:
            return pool[0][0]
    return None


def bring_to_front(before: set[int], hints: Iterable[str], timeout: float = 8.0, grace: float = 1.5) -> None:
    """Raise the launched window as soon as it appears (in the background)."""

    if _user32() is None:
        return
    keys = list(hints)

    def work() -> None:
        deadline = time.monotonic() + timeout
        # A new tab in an already-open browser only changes that window's title,
        # so title matches are checked too, after a short grace period.
        grace_until = time.monotonic() + grace
        while time.monotonic() < deadline:
            new_only = time.monotonic() < grace_until
            hwnd = find_window(before, keys) if not new_only else _new_matching(before, keys)
            if hwnd is not None:
                if raise_window(hwnd):
                    return
            time.sleep(0.25)
        LOGGER.info("Could not find a window to bring forward for %s", keys)

    threading.Thread(target=work, name="karen-focus", daemon=True).start()


def _new_matching(before: set[int], hints: list[str]) -> int | None:
    keys = [hint.lower() for hint in hints if hint]
    for hwnd, title in _windows():
        if hwnd not in before and (not keys or any(key in title.lower() for key in keys)):
            return hwnd
    return None
