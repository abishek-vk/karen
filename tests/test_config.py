from pathlib import Path

from karen.config import ConfigurationStore, Preferences, Settings


def test_missing_configuration_uses_safe_defaults(tmp_path: Path) -> None:
    store = ConfigurationStore(tmp_path)

    assert store.load_settings() == Settings()
    assert store.load_preferences() == Preferences()


def test_configuration_round_trip(tmp_path: Path) -> None:
    store = ConfigurationStore(tmp_path)
    settings = Settings(ollama_model="custom", privacy_mode=True, log_level="DEBUG")
    preferences = Preferences(voice_enabled=True)

    store.save_settings(settings)
    store.save_preferences(preferences)

    assert store.load_settings() == settings
    assert store.load_preferences() == preferences
