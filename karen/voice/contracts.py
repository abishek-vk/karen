"""Contracts and errors for optional voice components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class VoiceError(Exception):
    """Expected local voice setup or processing failure."""


class VoiceState(StrEnum):
    DISABLED = "disabled"
    READY = "ready"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    SPEAKING = "speaking"
    ERROR = "error"


class SpeechRecognizer(Protocol):
    def transcribe_once(self) -> str:
        """Capture a short local recording and return its transcript."""


class TextToSpeech(Protocol):
    def speak(self, text: str) -> None:
        """Speak text locally."""


@dataclass(frozen=True, slots=True)
class VoiceResponse:
    transcript: str
    response: str
