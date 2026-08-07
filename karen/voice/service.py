"""Orchestration for optional local voice input and output."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol

from karen.core.application_service import ServiceResponse

from .contracts import SpeechRecognizer, TextToSpeech, VoiceError, VoiceResponse, VoiceState


class ApplicationLike(Protocol):
    def handle_text(self, text: str) -> ServiceResponse:
        ...


@dataclass(slots=True)
class VoiceService:
    """Connect local speech adapters to the shared Karen application service."""

    application: ApplicationLike
    recognizer: SpeechRecognizer
    speaker: TextToSpeech | None = None
    state: VoiceState = field(init=False, default=VoiceState.READY)
    _lock: Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = Lock()

    def listen_and_process(self) -> VoiceResponse:
        """Capture one utterance, process it, and optionally speak the response."""

        if not self._lock.acquire(blocking=False):
            raise VoiceError("Voice input is already being processed.")
        try:
            self.state = VoiceState.LISTENING
            transcript = self.recognizer.transcribe_once()
            self.state = VoiceState.TRANSCRIBING
            response = self.application.handle_text(transcript)
            if self.speaker is not None:
                self.state = VoiceState.SPEAKING
                self.speaker.speak(response.message)
            self.state = VoiceState.READY
            return VoiceResponse(transcript, response.message)
        except VoiceError:
            self.state = VoiceState.ERROR
            raise
        except Exception as error:  # pragma: no cover - adapter dependent
            self.state = VoiceState.ERROR
            raise VoiceError(f"Voice processing failed: {error}") from error
        finally:
            self._lock.release()
