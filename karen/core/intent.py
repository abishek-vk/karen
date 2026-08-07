"""Strict parsing and validation of model-produced intent JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_INTENTS = frozenset(
    {
            "open_application",
            "close_application",
            "system_information",
            "open_windows_settings",
            "open_task_manager",
            "open_folder",
            "general_chat",
        }
)


@dataclass(frozen=True, slots=True)
class Intent:
    """Validated action data. Only this type may enter the router."""

    name: str
    target: str | None
    parameters: dict[str, Any]
    requires_confirmation: bool


def parse_intent(raw_response: str) -> Intent:
    """Extract the first valid intent object from model output.

    Qwen may wrap its answer in ``<think>`` text or repeat a JSON object.
    We scan JSON object boundaries rather than trusting the final brace, while
    still validating the resulting object strictly below.
    """

    candidate = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL | re.IGNORECASE)
    search_text = fenced.group(1) if fenced else candidate
    decoder = json.JSONDecoder()
    data: Any | None = None
    for match in re.finditer(r"\{", search_text):
        try:
            parsed, _end = decoder.raw_decode(search_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
    if data is None:
        raise ValueError("The model returned invalid intent JSON")
    if not isinstance(data, dict):
        raise ValueError("Intent must be a JSON object")

    name = data.get("intent")
    if name not in SUPPORTED_INTENTS:
        raise ValueError(f"Unsupported intent: {name!r}")
    target = data.get("target")
    if target is not None and not isinstance(target, str):
        raise ValueError("Intent target must be a string or null")
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Intent parameters must be an object")
    confirmation = data.get("requires_confirmation", False)
    if not isinstance(confirmation, bool):
        raise ValueError("requires_confirmation must be boolean")
    return Intent(name, target.strip() if target else None, parameters, confirmation)
