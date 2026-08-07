"""Allowlisted Windows navigation and read-only control actions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import SkillResult


FOLDERS = {
    "desktop": Path.home() / "Desktop",
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
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

    name = target.strip().lower() if target else ""
    folder = FOLDERS.get(name)
    if folder is None:
        return SkillResult(False, "I can only open Desktop, Documents, or Downloads.")
    if os.name != "nt":
        return SkillResult(False, "Windows controls are available on Windows only.")
    if not folder.exists():
        return SkillResult(False, f"Your {name.title()} folder was not found.")
    try:
        os.startfile(str(folder))  # type: ignore[attr-defined]
    except OSError:
        return SkillResult(False, f"I couldn't open {name.title()}.")
    return SkillResult(True, f"Opening {name.title()}.")
