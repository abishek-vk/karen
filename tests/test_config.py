import json
from pathlib import Path

from karen.config import ConfigurationStore, Preferences, Settings


def test_missing_configuration_uses_safe_defaults(tmp_path: Path) -> None:
    store = ConfigurationStore(tmp_path)

    assert store.load_settings() == Settings()
    assert store.load_preferences() == Preferences()


def test_legacy_settings_keys_are_supported(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"ollama_url": "http://127.0.0.1:11434", "model": "legacy", "timeout_seconds": 12}),
        encoding="utf-8",
    )

    settings = ConfigurationStore(tmp_path).load_settings()

    assert settings.ollama_model == "legacy"
    assert settings.request_timeout_seconds == 12


def test_configuration_round_trip(tmp_path: Path) -> None:
    store = ConfigurationStore(tmp_path)
    settings = Settings(ollama_model="custom", privacy_mode=True, log_level="DEBUG")
    preferences = Preferences(voice_enabled=True)

    store.save_settings(settings)
    store.save_preferences(preferences)

    assert store.load_settings() == settings
    assert store.load_preferences() == preferences
