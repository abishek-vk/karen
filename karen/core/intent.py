"""Strict parsing and validation of model-produced intent JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_INTENTS = frozenset(
    {"open_application", "close_application", "system_information", "general_chat"}
)


@dataclass(frozen=True, slots=True)
class Intent:
    """Validated action data. Only this type may enter the router."""

    name: str
    target: str | None
    parameters: dict[str, Any]
    requires_confirmation: bool


def parse_intent(raw_response: str) -> Intent:
    """Parse one JSON object, including models that wrap it in code fences."""

    candidate = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("The model returned invalid intent JSON") from error
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
