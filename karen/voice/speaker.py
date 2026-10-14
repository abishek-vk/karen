"""Sentence-by-sentence speech output on a background thread.

Streamed answer text is fed in as it arrives; each completed sentence is
spoken immediately, so Karen starts talking long before the model finishes.
``stop()`` interrupts speech at once (barge-in).
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Callable

from .contracts import TextToSpeech, VoiceError

LOGGER = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n+")
_ABBREVIATIONS = re.compile(r"(?:\b(?:mr|mrs|ms|dr|st|vs|etc|e\.g|i\.e|no|jr|sr|inc|ltd)|\b[a-z])\.$", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+")


def clean_for_speech(text: str) -> str:
    """Turn display text into something pleasant to hear."""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(?:sources?|source:|links?:)", stripped, re.IGNORECASE):
            continue
        stripped = re.sub(r"^(?:[-*•]+|\d+[.)])\s+", "", stripped)
        lines.append(stripped)
    spoken = " ".join(line for line in lines if line)
    spoken = _URL.sub("", spoken)
    spoken = re.sub(r"(?<=\w)_(?=\w)", " ", spoken)  # reverse_string -> "reverse string"
    spoken = re.sub(r"[*_`#>|]", "", spoken)
    spoken = spoken.replace("°C", " degrees Celsius").replace("°F", " degrees Fahrenheit")
    spoken = spoken.replace("km/h", "kilometres per hour").replace("&", " and ")
    return re.sub(r"\s+", " ", spoken).strip()


class SpeechOutput:
    """Queue sentences for a single TTS worker thread."""

    def __init__(self, engine: TextToSpeech | None, on_state: Callable[[bool], None] | None = None) -> None:
        self.engine = engine
        self.on_state = on_state
        self.enabled = engine is not None
        self._queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self._buffer = ""
        self._in_code = False
        self._generation = 0
        self._lock = threading.Lock()
        self._speaking = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self.engine is not None

    @property
    def is_speaking(self) -> bool:
        return not self._idle.is_set()

    def _start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="karen-speech", daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        """Speak a complete message (after any queued speech)."""

        self.feed(text)
        self.flush()

    def speak(self, text: str) -> None:
        """Speak a complete message and wait until it has been spoken."""

        self.say(text)
        self.wait()

    def feed(self, chunk: str) -> None:
        """Add streamed text; complete sentences are spoken right away."""

        if not self.enabled or not chunk:
            return
        with self._lock:
            self._buffer += chunk
            pieces = _SENTENCE_END.split(self._buffer)
            complete, self._buffer = pieces[:-1], pieces[-1]
            sentence = ""
            for piece in complete:
                if piece.strip().startswith("```"):
                    # Code is for the screen; reading it aloud symbol by symbol is useless.
                    if sentence:
                        self._enqueue(sentence)
                        sentence = ""
                    self._in_code = not self._in_code
                    if self._in_code:
                        self._enqueue("I've put the code on screen.")
                    continue
                if self._in_code:
                    continue
                sentence = f"{sentence} {piece}".strip() if sentence else piece.strip()
                # Don't split after "Dr." and similar, and avoid tiny fragments.
                if _ABBREVIATIONS.search(sentence) or len(sentence) < 12:
                    continue
                self._enqueue(sentence)
                sentence = ""
            if sentence:
                # Keep the buffer's trailing space, or the next streamed word gets glued on.
                self._buffer = f"{sentence} {self._buffer}".lstrip()

    def flush(self) -> None:
        with self._lock:
            remainder, self._buffer = self._buffer, ""
            in_code, self._in_code = self._in_code, False
            if remainder.strip() and not in_code and not remainder.strip().startswith("```"):
                self._enqueue(remainder)

    def _enqueue(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if spoken and self.enabled:
            self._idle.clear()
            self._start()
            self._queue.put((self._generation, spoken))

    def stop(self) -> None:
        """Interrupt current speech and drop everything queued."""

        with self._lock:
            self._generation += 1
            self._buffer = ""
            self._in_code = False
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
        stop = getattr(self.engine, "stop", None)
        if stop is not None:
            try:
                stop()
            except Exception:  # pragma: no cover - engine dependent
                LOGGER.debug("TTS stop failed", exc_info=True)
        if self._queue.empty() and not self._speaking.is_set():
            self._set_idle()

    def wait(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def close(self) -> None:
        self.stop()
        self._queue.put(None)
        close = getattr(self.engine, "close", None)
        if close is not None:
            close()

    def _set_idle(self) -> None:
        if not self._idle.is_set():
            self._idle.set()
            if self.on_state is not None:
                self.on_state(False)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, text = item
            if generation != self._generation or self.engine is None:
                if self._queue.empty():
                    self._set_idle()
                continue
            if self.on_state is not None and not self._speaking.is_set():
                self.on_state(True)
            self._speaking.set()
            try:
                self.engine.speak(text)
            except VoiceError as error:
                LOGGER.warning("Speech output failed: %s", error)
                self.enabled = False
            except Exception:  # pragma: no cover - engine dependent
                LOGGER.exception("Speech output failed")
            finally:
                self._speaking.clear()
            if self._queue.empty():
                self._set_idle()
