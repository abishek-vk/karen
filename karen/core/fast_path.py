"""Deterministic handling for common low-risk commands."""

from __future__ import annotations

import re

from .intent import Intent


_APPLICATION_COMMANDS = {
    "notepad": "notepad",
    "calculator": "calculator",
    "chrome": "chrome",
}


def parse_fast_path(text: str) -> Intent | None:
    """Return a safe intent for a small set of exact command patterns.

    False negatives intentionally fall through to the local model. This parser
    must remain narrow because a false positive could launch an unintended app.
    """

    normalized = re.sub(r"\s+", " ", text.strip().lower().rstrip(".!?"))
    match = re.fullmatch(r"(?:please )?open (notepad|calculator|chrome)", normalized)
    if match:
        return Intent("open_application", _APPLICATION_COMMANDS[match.group(1)], {}, False)

    if normalized in {"what time is it", "tell me the time", "current time"}:
        return Intent("system_information", None, {}, False)

    if normalized in {"show system information", "system information", "system status"}:
        return Intent("system_information", None, {}, False)

    return None
