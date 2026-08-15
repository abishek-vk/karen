"""Persistent, local-only application configuration."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


def _normalize_settings(values: dict[str, Any]) -> dict[str, Any]:
    """Accept the original Phase 1 keys while using canonical names."""

    normalized = dict(values)
    if "ollama_model" not in normalized and "model" in normalized:
        normalized["ollama_model"] = normalized["model"]
    if "request_timeout_seconds" not in normalized and "timeout_seconds" in normalized:
        normalized["request_timeout_seconds"] = normalized["timeout_seconds"]
    normalized.pop("model", None)
    normalized.pop("timeout_seconds", None)
    return normalized


@dataclass(slots=True)
class Settings:
    """Runtime settings that are safe to persist as JSON."""

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"
    request_timeout_seconds: float = 180.0
    privacy_mode: bool = True
    start_with_windows: bool = False
    log_level: str = "INFO"
    memory_enabled: bool = True
    recent_message_limit: int = 15
    max_memory_results: int = 5
    live_information_enabled: bool = True
    live_information_timeout_seconds: float = 8.0

    @property
    def model(self) -> str:
        """Legacy alias for callers from the original text core."""

        return self.ollama_model

    @property
    def timeout_seconds(self) -> float:
        """Legacy alias for callers from the original text core."""

        return self.request_timeout_seconds


@dataclass(slots=True)
class Preferences:
    """User preferences kept separate from service configuration."""

    voice_enabled: bool = False
    wake_word_enabled: bool = False
    speak_responses: bool = False


class ConfigurationStore:
    """Loads and atomically saves local JSON configuration files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.settings_path = data_dir / "settings.json"
        self.preferences_path = data_dir / "preferences.json"

    def load_settings(self) -> Settings:
        return self._load(self.settings_path, Settings)

    def load_preferences(self) -> Preferences:
        return self._load(self.preferences_path, Preferences)

    def save_settings(self, settings: Settings) -> None:
        self._save(self.settings_path, asdict(settings))

    def save_preferences(self, preferences: Preferences) -> None:
        self._save(self.preferences_path, asdict(preferences))

    @staticmethod
    def _load(path: Path, model: type[Settings] | type[Preferences]) -> Any:
        if not path.exists():
            return model()
        try:
            with path.open("r", encoding="utf-8") as file:
                values = json.load(file)
            if not isinstance(values, dict):
                raise ValueError("configuration root must be an object")
            if model is Settings:
                values = _normalize_settings(values)
            return model(**values)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"Could not load {path.name}: {error}") from error

    @staticmethod
    def _save(path: Path, values: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as file:
                json.dump(values, file, indent=2)
                file.write("\n")
                temporary_path = file.name
            os.replace(temporary_path, path)
        except OSError as error:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
            raise ConfigurationError(f"Could not save {path.name}: {error}") from error
