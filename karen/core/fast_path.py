"""Deterministic handling for common low-risk commands."""

from __future__ import annotations

import re

from karen.skills.app_control import resolve_application_name

from .intent import Intent


_APPLICATION_COMMANDS = {
    "notepad": "notepad",
    "calculator": "calculator",
    "chrome": "chrome",
    "vs code": "vs code",
    "vscode": "vscode",
    "edge": "edge",
}


def parse_fast_path(text: str) -> Intent | None:
    """Return a safe intent for a small set of exact command patterns.

    False negatives intentionally fall through to the local model. This parser
    must remain narrow because a false positive could launch an unintended app.
    """

    normalized = re.sub(r"\s+", " ", text.strip().lower().rstrip(".!?"))
    match = re.fullmatch(
        r"(?:please )?open (notepad|calculator|chrome|vs code|vscode|edge)",
        normalized,
    )
    if match:
        return Intent("open_application", _APPLICATION_COMMANDS[match.group(1)], {}, False)

    if normalized in {"what time is it", "tell me the time", "current time"}:
        return Intent("system_information", None, {}, False)

    if normalized in {"show system information", "system information", "system status"}:
        return Intent("system_information", None, {}, False)

    if normalized in {"open whatsapp", "open whatsapp web", "start whatsapp"}:
        return Intent("open_whatsapp", "whatsapp", {}, False)

    if normalized in {"open settings", "open windows settings", "open windows settings app"}:
        return Intent("open_windows_settings", None, {}, False)

    if normalized in {"open task manager", "start task manager"}:
        return Intent("open_task_manager", None, {}, False)

    folder_match = re.fullmatch(r"(?:please )?open (desktop|documents|downloads)", normalized)
    if folder_match:
        return Intent("open_folder", folder_match.group(1), {}, False)

    catalog_match = re.fullmatch(r"(?:please )?open (.+)", normalized)
    if catalog_match:
        resolved = resolve_application_name(catalog_match.group(1))
        if resolved is not None:
            return Intent("open_application", resolved, {}, False)

    return None
