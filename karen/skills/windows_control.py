"""Allowlisted Windows navigation and read-only control actions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import SkillResult


HOME = Path.home()
FOLDERS = {
    "home": HOME,
    "user folder": HOME,
    "desktop": HOME / "Desktop",
    "documents": HOME / "Documents",
    "downloads": HOME / "Downloads",
    "pictures": HOME / "Pictures",
    "photos": HOME / "Pictures",
    "videos": HOME / "Videos",
    "music": HOME / "Music",
    "3d objects": HOME / "3D Objects",
    "favorites": HOME / "Favorites",
}


def open_windows_settings() -> SkillResult:
    """Open the Windows Settings application without a shell."""

    if os.name != "nt":
        return SkillResult(False, "Windows controls are available on Windows only.")
    try:
        os.startfile("ms-settings:")  # type: ignore[attr-defined]
    except OSError:
        return SkillResult(False, "I couldn't open Windows Settings.")
    return SkillResult(True, "Opening Windows Settings.")


def open_task_manager() -> SkillResult:
    """Open the allowlisted Windows Task Manager executable."""

    if os.name != "nt":
        return SkillResult(False, "Windows controls are available on Windows only.")
    try:
        subprocess.Popen(["taskmgr.exe"], shell=False, close_fds=True)
    except OSError:
        return SkillResult(False, "I couldn't open Task Manager.")
    return SkillResult(True, "Opening Task Manager.")


def open_folder(target: str | None) -> SkillResult:
    """Open one of the user's standard folders, never an arbitrary path."""

    name = " ".join(target.strip().lower().split()) if target else ""
    folder = FOLDERS.get(name)
    if folder is None:
        return SkillResult(
            False,
            "I can open Home, Desktop, Documents, Downloads, Pictures, Videos, or Music.",
        )
    if os.name != "nt":
        return SkillResult(False, "Windows controls are available on Windows only.")
    if not folder.exists():
        return SkillResult(False, f"Your {name.title()} folder was not found.")
    try:
        os.startfile(str(folder))  # type: ignore[attr-defined]
    except OSError:
        return SkillResult(False, f"I couldn't open {name.title()}.")
    return SkillResult(True, f"Opening {name.title()}.")
