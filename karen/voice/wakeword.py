"""Optional always-listening wake word ("Hey Jarvis") using openWakeWord.

The listener closes the microphone while paused (during a command or while
Karen is busy) so it never competes with the recognizer for the device or CPU.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .contracts import VoiceError

LOGGER = logging.getLogger(__name__)

_FRAME = 1280  # 80 ms at 16 kHz, the frame size openWakeWord expects.


class WakeWordListener:
    def __init__(
        self,
        on_detect: Callable[[], None],
        *,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.on_detect = on_detect
        self.on_error = on_error
        self.model_name = model_name
        self.threshold = threshold
        self._active = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._active.set()
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="karen-wakeword", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        if not self._stopped.is_set():
            self._active.set()

    def stop(self) -> None:
        self._stopped.set()
        self._active.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stopped.is_set()

    def _load(self):  # noqa: ANN202 - third-party model
        try:
            import openwakeword.utils
            from openwakeword.model import Model
        except ImportError as error:
            raise VoiceError("The wake word needs: pip install openwakeword") from error
        try:
            return Model(wakeword_models=[self.model_name], inference_framework="onnx")
        except Exception:
            # First run: fetch the small pretrained models once, then retry.
            try:
                openwakeword.utils.download_models([self.model_name])
                return Model(wakeword_models=[self.model_name], inference_framework="onnx")
            except Exception as error:  # pragma: no cover - network/model dependent
                raise VoiceError(f"Could not load the wake word model: {error}") from error

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd

            model = self._load()
        except (ImportError, VoiceError) as error:
            message = str(error) if isinstance(error, VoiceError) else "The wake word needs the voice extras."
            LOGGER.warning("%s", message)
            if self.on_error is not None:
                self.on_error(message)
            self._thread = None
            return

        while not self._stopped.is_set():
            self._active.wait()
            if self._stopped.is_set():
                break
            try:
                with sd.InputStream(samplerate=16_000, channels=1, dtype="int16", blocksize=_FRAME) as stream:
                    while self._active.is_set() and not self._stopped.is_set():
                        data, _ = stream.read(_FRAME)
                        scores = model.predict(np.asarray(data[:, 0], dtype=np.int16))
                        if max(scores.values(), default=0.0) >= self.threshold:
                            model.reset()
                            self._active.clear()
                            self.on_detect()
                            break
            except Exception as error:  # pragma: no cover - device dependent
                LOGGER.warning("Wake word microphone error: %s", error)
                time.sleep(2.0)
