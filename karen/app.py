"""Composition root: builds every Karen service once for the text and GUI front ends."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from .ai.ollama import OllamaClient
from .config import ConfigurationStore, Preferences, Settings
from .core.application_service import ApplicationService
from .core.llm import LocalLLM
from .core.router import IntentRouter
from .exceptions import ConfigurationError
from .logging_config import configure_logging
from .memory import MemoryService
from .memory.reminders import ReminderService
from .voice.recognizer import FasterWhisperRecognizer
from .voice.speaker import SpeechOutput
from .voice.tts import build_text_to_speech
from .voice.wakeword import WakeWordListener
from .web import CurrentInfoService

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent


class KarenRuntime:
    """Own Karen's long-lived services and shut them down cleanly."""

    def __init__(self, project_dir: Path = PROJECT_DIR, *, on_speech_state: Callable[[bool], None] | None = None) -> None:
        self.project_dir = project_dir
        self.data_dir = project_dir / "data"
        self.store = ConfigurationStore(project_dir / "config")
        try:
            self.settings: Settings = self.store.load_settings()
        except ConfigurationError as error:
            # A broken settings file must not stop the assistant from starting.
            self.settings = Settings()
            configure_logging(self.data_dir, "INFO")
            LOGGER.error("%s. Using default settings.", error)
        else:
            configure_logging(self.data_dir, self.settings.log_level)
        self.preferences: Preferences = self.store.load_preferences_for(self.settings)

        settings = self.settings
        self.client = OllamaClient(
            settings.ollama_url,
            settings.ollama_model,
            settings.request_timeout_seconds,
            context_window=settings.context_window,
            keep_alive=settings.keep_alive,
            num_thread=settings.llm_threads,
        )
        self.llm = LocalLLM(self.client, assistant_name=settings.assistant_name, max_answer_tokens=settings.max_answer_tokens)
        self.memory = MemoryService(
            self.data_dir / "karen.db",
            enabled=settings.memory_enabled,
            recent_message_limit=settings.recent_message_limit,
            max_memory_results=settings.max_memory_results,
            max_stored_messages=settings.max_stored_messages,
        )
        self.reminders = ReminderService(self.memory)
        self.service = ApplicationService(
            self.llm,
            IntentRouter(),
            memory=self.memory,
            reminders=self.reminders,
            live_information=CurrentInfoService(
                enabled=settings.live_information_enabled,
                timeout_seconds=settings.live_information_timeout_seconds,
                max_pages=settings.web_max_pages,
            ),
        )
        self.speaker = SpeechOutput(
            build_text_to_speech(
                settings.tts_engine,
                voice_hint=settings.tts_voice,
                rate=settings.tts_rate,
                piper_executable=settings.piper_executable,
                piper_model_path=settings.piper_model_path,
            ),
            on_state=on_speech_state,
            pronunciations=settings.pronunciations,
        )
        self.speaker.enabled = self.speaker.available and self.preferences.speak_responses
        self._recognizer: FasterWhisperRecognizer | None = None
        self.wake_word: WakeWordListener | None = None
        LOGGER.info("Karen initialized (model %s)", settings.ollama_model)

    # -- lifecycle --------------------------------------------------------

    def start_background(self, notify: Callable[[str], None]) -> None:
        """Start reminders and warm up the model without blocking the caller."""

        self.reminders.notify = notify
        self.reminders.start()

        def warm() -> None:
            if self.llm.ensure_available() and self.llm.use_installed_model():
                self.llm.warm_up()

        threading.Thread(target=warm, name="karen-warmup", daemon=True).start()

    @property
    def recognizer(self) -> FasterWhisperRecognizer:
        if self._recognizer is None:
            self._recognizer = FasterWhisperRecognizer(
                model_size=self.settings.whisper_model,
                record_seconds=self.settings.listen_max_seconds,
                silence_seconds=self.settings.listen_silence_seconds,
                cpu_threads=self.settings.whisper_threads,
            )
        return self._recognizer

    def start_wake_word(self, on_detect: Callable[[], None], on_error: Callable[[str], None] | None = None) -> None:
        if self.wake_word is None:
            self.wake_word = WakeWordListener(
                on_detect,
                model_name=self.settings.wake_word_model,
                threshold=self.settings.wake_word_threshold,
                on_error=on_error,
            )
        self.wake_word.start()

    def stop_wake_word(self) -> None:
        if self.wake_word is not None:
            self.wake_word.stop()
            self.wake_word = None

    def save_preferences(self) -> None:
        try:
            self.store.save_preferences(self.preferences)
        except ConfigurationError as error:
            LOGGER.warning("%s", error)

    def shutdown(self) -> None:
        self.reminders.stop()
        self.stop_wake_word()
        if self._recognizer is not None:
            self._recognizer.cancel()
        self.speaker.close()
        self.llm.shutdown()


# Backwards-compatible name used by earlier versions.
KarenApplication = KarenRuntime
