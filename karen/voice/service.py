"""Orchestration for local voice input and output."""

from __future__ import annotations

from collections.abc import Callable
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
    """Connect a recognizer and a speaker to the shared application service.

    With a streaming speaker (``SpeechOutput``) the reply is spoken sentence by
    sentence while it is still being generated.
    """

    application: ApplicationLike
    recognizer: SpeechRecognizer
    speaker: TextToSpeech | None = None
    state: VoiceState = field(init=False, default=VoiceState.READY)
    _lock: Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = Lock()

    def listen_and_process(
        self,
        on_transcript: Callable[[str], None] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> VoiceResponse:
        """Capture one utterance, process it, and speak the response."""

        if not self._lock.acquire(blocking=False):
            raise VoiceError("Voice input is already being processed.")
        try:
            self.state = VoiceState.LISTENING
            transcript = self.recognizer.transcribe_once()
            if on_transcript is not None:
                on_transcript(transcript)
            self.state = VoiceState.THINKING
            feed = getattr(self.speaker, "feed", None)
            if feed is not None:
                def chunk(text: str) -> None:
                    feed(text)
                    if on_chunk is not None:
                        on_chunk(text)

                response = self.application.handle_text(transcript, on_chunk=chunk)  # type: ignore[call-arg]
                self.speaker.flush()  # type: ignore[union-attr]
            else:
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
