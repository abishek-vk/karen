"""Configuration for the text-based Karen core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CoreSettings:
    """Settings needed by the text application."""

    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    timeout_seconds: float = 60.0


def load_settings(path: Path) -> CoreSettings:
    """Load settings from JSON, falling back to safe local defaults."""

    if not path.exists():
        return CoreSettings()
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("settings must be a JSON object")
        return CoreSettings(
            ollama_url=str(values.get("ollama_url", CoreSettings.ollama_url)),
            model=str(values.get("model", values.get("ollama_model", CoreSettings.model))),
            timeout_seconds=float(values.get("timeout_seconds", 60.0)),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load settings from {path}: {error}") from error
