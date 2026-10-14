"""Faster-Whisper speech recognition with end-of-speech detection.

Recording stops as soon as the user stops talking (energy-based endpointing)
instead of after a fixed duration, and the model can be preloaded in the
background so the first command is not delayed.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Any

from .contracts import VoiceError

LOGGER = logging.getLogger(__name__)

_BLOCK_SECONDS = 0.03
# Whisper tends to "hear" these phrases in silence or noise.
_HALLUCINATIONS = frozenset({
    "", "you", "thank you", "thanks for watching", "thank you for watching", "bye", "okay",
    "subtitles by the amara.org community", "please subscribe",
})


class FasterWhisperRecognizer:
    """Record one utterance locally and transcribe it with Faster-Whisper."""

    def __init__(
        self,
        model_size: str = "base.en",
        record_seconds: float = 12.0,
        sample_rate: int = 16_000,
        language: str | None = "en",
        initial_prompt: str | None = None,
        *,
        silence_seconds: float = 0.8,
        start_timeout_seconds: float = 6.0,
        cpu_threads: int = 0,
    ) -> None:
        if record_seconds <= 0:
            raise ValueError("record_seconds must be positive")
        self.model_size = model_size
        self.record_seconds = record_seconds
        self.sample_rate = sample_rate
        self.language = language if not model_size.endswith(".en") else "en"
        self.initial_prompt = initial_prompt
        self.silence_seconds = max(0.3, silence_seconds)
        self.start_timeout_seconds = max(1.0, start_timeout_seconds)
        self.cpu_threads = cpu_threads or max(2, (os.cpu_count() or 4) // 2)
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._cancel = threading.Event()

    # -- model ------------------------------------------------------------

    def _load_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise VoiceError(
                    "Speech recognition needs the voice extras: pip install -e .[voice]"
                ) from error
            try:
                self._model = WhisperModel(
                    self.model_size, device="cpu", compute_type="int8", cpu_threads=self.cpu_threads
                )
            except Exception as error:  # pragma: no cover - hardware/model dependent
                raise VoiceError(f"Could not load the local speech model: {error}") from error
            return self._model

    def preload(self) -> None:
        """Load the model in a background thread; errors surface on first use."""

        def load() -> None:
            try:
                self._load_model()
                LOGGER.info("Speech model %s loaded", self.model_size)
            except VoiceError as error:
                LOGGER.warning("%s", error)

        threading.Thread(target=load, name="karen-whisper-load", daemon=True).start()

    def cancel(self) -> None:
        self._cancel.set()

    # -- capture ----------------------------------------------------------

    def _record(self) -> Any:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as error:
            raise VoiceError("Microphone input needs the voice extras: pip install -e .[voice]") from error

        self._cancel.clear()
        block = int(self.sample_rate * _BLOCK_SECONDS)
        max_blocks = int(self.record_seconds / _BLOCK_SECONDS)
        start_blocks = int(self.start_timeout_seconds / _BLOCK_SECONDS)
        silence_blocks = int(self.silence_seconds / _BLOCK_SECONDS)
        preroll: deque[Any] = deque(maxlen=10)  # ~300 ms before speech begins
        captured: list[Any] = []
        noise: list[float] = []
        threshold = 0.015
        loud_run = 0
        quiet_run = 0
        speaking = False
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=block) as stream:
                for index in range(max_blocks):
                    if self._cancel.is_set():
                        raise VoiceError("Listening cancelled.")
                    data, _overflowed = stream.read(block)
                    samples = data[:, 0].copy()
                    level = float(np.sqrt(np.mean(samples * samples)))
                    if index < 8:
                        # Calibrate to the room's background noise first.
                        noise.append(level)
                        threshold = max(0.012, float(np.median(noise)) * 3.0)
                    if not speaking:
                        preroll.append(samples)
                        loud_run = loud_run + 1 if level > threshold else 0
                        if loud_run >= 3:
                            speaking = True
                            captured.extend(preroll)
                        elif index >= start_blocks:
                            raise VoiceError("I didn't hear anything.")
                        continue
                    captured.append(samples)
                    quiet_run = quiet_run + 1 if level < threshold * 0.8 else 0
                    if quiet_run >= silence_blocks:
                        break
        except VoiceError:
            raise
        except Exception as error:  # pragma: no cover - device dependent
            raise VoiceError(f"Could not record from the microphone: {error}") from error
        if not captured:
            raise VoiceError("I didn't hear anything.")
        return np.concatenate(captured)

    def transcribe_once(self) -> str:
        audio = self._record()
        try:
            segments, _info = self._load_model().transcribe(
                audio,
                language=self.language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                initial_prompt=self.initial_prompt,
                condition_on_previous_text=False,
                vad_filter=False,  # Recording already ends at silence.
                without_timestamps=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
        except VoiceError:
            raise
        except Exception as error:  # pragma: no cover - model dependent
            raise VoiceError(f"Could not transcribe the recording: {error}") from error
        if transcript.lower().strip(" .!?,") in _HALLUCINATIONS:
            raise VoiceError("I didn't catch that.")
        return transcript
