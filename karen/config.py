"""Persistent, local-only application configuration."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError

LOGGER = logging.getLogger(__name__)


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

    # Local model (Ollama). Smaller models answer faster on CPU-only laptops.
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"
    request_timeout_seconds: float = 180.0
    context_window: int = 4096
    keep_alive: str = "20m"
    max_answer_tokens: int = 400
    llm_threads: int = 0  # 0 lets Ollama choose; set e.g. 4 to leave CPU headroom.

    # Assistant
    assistant_name: str = "Karen"
    privacy_mode: bool = True
    start_with_windows: bool = False
    log_level: str = "INFO"

    # Memory
    memory_enabled: bool = True
    recent_message_limit: int = 6
    max_memory_results: int = 5
    max_stored_messages: int = 2000

    # Web research
    live_information_enabled: bool = True
    live_information_timeout_seconds: float = 8.0
    web_max_pages: int = 3

    # Voice output
    speak_responses: bool = True
    tts_engine: str = "sapi"  # "sapi" (built into Windows), "piper", or "none"
    tts_voice: str = ""  # Part of a Windows voice name, e.g. "Zira" or "David".
    tts_rate: int = 1  # SAPI speaking rate from -10 to 10.
    piper_executable: str = "piper"
    piper_model_path: str = ""

    # Voice input
    whisper_model: str = "base.en"
    whisper_threads: int = 0  # 0 uses half of the CPU cores.
    listen_max_seconds: float = 12.0
    listen_silence_seconds: float = 0.8
    wake_word_enabled: bool = False
    wake_word_model: str = "hey_jarvis"
    wake_word_threshold: float = 0.5

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
    """User toggles changed from the UI, kept separate from service configuration."""

    voice_enabled: bool = False
    wake_word_enabled: bool = False
    speak_responses: bool = False


def _coerce(model: type, values: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys and repair wrong types instead of crashing at startup."""

    defaults = model()
    known = {field.name for field in fields(model)}
    clean: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known:
            LOGGER.warning("Ignoring unknown %s key %r", model.__name__, key)
            continue
        expected = type(getattr(defaults, key))
        if isinstance(value, expected) and not (expected is not bool and isinstance(value, bool)):
            clean[key] = value
            continue
        try:
            if expected is bool:
                raise TypeError
            clean[key] = expected(value)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid value for %s.%s; using the default", model.__name__, key)
    return clean


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

    def load_preferences_for(self, settings: Settings) -> Preferences:
        """Return saved UI toggles, or defaults derived from settings."""

        if self.preferences_path.exists():
            try:
                return self.load_preferences()
            except ConfigurationError as error:
                LOGGER.warning("%s", error)
        return Preferences(
            voice_enabled=True,
            wake_word_enabled=settings.wake_word_enabled,
            speak_responses=settings.speak_responses,
        )

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
            return model(**_coerce(model, values))
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
