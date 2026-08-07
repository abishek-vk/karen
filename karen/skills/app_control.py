"""Safe application launching and closing for an explicit allowlist."""

from __future__ import annotations

import os
import subprocess
from karen.skills.base import SkillResult


APPLICATIONS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "chrome": "chrome.exe",
}



def _application_name(target: str | None) -> str | None:
    if not target:
        return None
    return target.strip().lower().removesuffix(".exe")


def open_application(target: str | None) -> SkillResult:
    """Launch one allowlisted Windows app without invoking a shell."""

    name = _application_name(target)
    executable = APPLICATIONS.get(name or "")
    if executable is None or name is None:
        return SkillResult(False, "I can only open Notepad, Calculator, or Chrome.")
    display_name = name.title()
    if os.name != "nt":
        return SkillResult(False, "Application control is available on Windows only.")
    try:
        _ = subprocess.Popen([executable], shell=False, close_fds=True)
    except OSError:
        return SkillResult(False, f"I couldn't open {display_name}.")
    return SkillResult(True, f"Opening {display_name}.")


def close_application(target: str | None) -> SkillResult:
    """Refuse closing for now; no process termination is exposed in Phase 1.1."""

    _ = target
    return SkillResult(False, "Closing applications is not enabled yet.")
