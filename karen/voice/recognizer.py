"""Lazy Faster-Whisper microphone recognizer."""

from __future__ import annotations

from typing import Any

from .contracts import VoiceError


_DEFAULT_PROMPT = (
    "Karen, open Beekeeper Studio, Google Chrome, Microsoft Edge, Brave, "
    "VS Code, Discord, Calculator, Notepad, or Task Manager."
)


class FasterWhisperRecognizer:
    """Record a short clip locally and transcribe it with Faster-Whisper."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        record_seconds: float = 4.0,
        sample_rate: int = 16_000,
        language: str | None = "en",
        initial_prompt: str = _DEFAULT_PROMPT,
    ) -> None:
        if record_seconds <= 0:
            raise ValueError("record_seconds must be positive")
        self.model_size = model_size
        self.record_seconds = record_seconds
        self.sample_rate = sample_rate
        self.language = language
        self.initial_prompt = initial_prompt
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise VoiceError(
                "Voice recognition is unavailable. Install the optional voice dependencies."
            ) from error
        try:
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception as error:  # pragma: no cover - hardware/model dependent
            raise VoiceError(f"Could not load the local speech model: {error}") from error
        return self._model

    def transcribe_once(self) -> str:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise VoiceError(
                "Microphone input is unavailable. Install the optional voice dependencies."
            ) from error

        try:
            recording = sd.rec(
                int(self.record_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception as error:  # pragma: no cover - device dependent
            raise VoiceError(f"Could not record from the microphone: {error}") from error

        try:
            segments, _info = self._load_model().transcribe(
                recording[:, 0],
                language=self.language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                initial_prompt=self.initial_prompt,
                vad_filter=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as error:  # pragma: no cover - model dependent
            raise VoiceError(f"Could not transcribe the recording: {error}") from error
        if not transcript:
            raise VoiceError("No speech was detected.")
        return transcript
