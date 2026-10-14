"""Local text-to-speech engines. Every engine is interruptible via ``stop()``.

* ``SapiTextToSpeech`` – the voices built into Windows, through COM
  (``pip install comtypes``). Fast, offline and nearly free on CPU.
* ``PowerShellTextToSpeech`` – the same Windows voices with zero Python
  dependencies, through one long-lived PowerShell process.
* ``PiperTextToSpeech`` – optional neural voice for a more natural sound.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

from .contracts import TextToSpeech, VoiceError

LOGGER = logging.getLogger(__name__)

_SVSF_ASYNC = 1
_SVSF_PURGE = 2


class SapiTextToSpeech:
    """Windows SAPI voice through COM. All calls must happen on one thread."""

    def __init__(self, voice_hint: str = "", rate: int = 1, volume: int = 100) -> None:
        self.voice_hint = voice_hint.strip().lower()
        self.rate = max(-10, min(10, rate))
        self.volume = max(0, min(100, volume))
        self._voice = None
        self._stop = threading.Event()

    @staticmethod
    def available() -> bool:
        if os.name != "nt":
            return False
        try:
            import comtypes.client  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure(self):  # noqa: ANN202 - COM object
        if self._voice is not None:
            return self._voice
        try:
            import comtypes
            import comtypes.client
        except ImportError as error:
            raise VoiceError("Install comtypes for Windows speech: pip install comtypes") from error
        comtypes.CoInitialize()
        voice = comtypes.client.CreateObject("SAPI.SpVoice")
        if self.voice_hint:
            tokens = voice.GetVoices()
            for index in range(tokens.Count):
                token = tokens.Item(index)
                if self.voice_hint in token.GetDescription().lower():
                    voice.Voice = token
                    break
        voice.Rate = self.rate
        voice.Volume = self.volume
        self._voice = voice
        return voice

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        voice = self._ensure()
        self._stop.clear()
        voice.Speak(text, _SVSF_ASYNC)
        while not voice.WaitUntilDone(80):
            if self._stop.is_set():
                voice.Speak("", _SVSF_ASYNC | _SVSF_PURGE)
                return

    def stop(self) -> None:
        self._stop.set()


_PS_SCRIPT = r"""
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$hint = $env:KAREN_TTS_VOICE
if ($hint) { foreach ($v in $s.GetInstalledVoices()) { if ($v.VoiceInfo.Name -like "*$hint*") { $s.SelectVoice($v.VoiceInfo.Name); break } } }
$s.Rate = [int]$env:KAREN_TTS_RATE
[Console]::Out.WriteLine('ready'); [Console]::Out.Flush()
while ($null -ne ($line = [Console]::In.ReadLine())) {
  if ($line.Length -gt 0) { $s.Speak($line) }
  [Console]::Out.WriteLine('done'); [Console]::Out.Flush()
}
"""


class PowerShellTextToSpeech:
    """Windows voices without extra packages; ``stop()`` kills the speaking process."""

    def __init__(self, voice_hint: str = "", rate: int = 1) -> None:
        self.voice_hint = voice_hint
        self.rate = max(-10, min(10, rate))
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        return os.name == "nt" and shutil.which("powershell") is not None

    def _ensure(self) -> subprocess.Popen[str]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self._process
            environment = dict(os.environ, KAREN_TTS_VOICE=self.voice_hint, KAREN_TTS_RATE=str(self.rate))
            encoded = base64.b64encode(_PS_SCRIPT.encode("utf-16-le")).decode("ascii")
            process = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if process.stdout is None or process.stdout.readline().strip() != "ready":
                process.kill()
                raise VoiceError("Windows speech could not start.")
            self._process = process
            return process

    def speak(self, text: str) -> None:
        line = " ".join(text.split())
        if not line:
            return
        process = self._ensure()
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(line + "\n")
            process.stdin.flush()
            process.stdout.readline()  # Returns "done", or "" if stop() killed the process.
        except (OSError, ValueError):
            pass

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()

    def close(self) -> None:
        self.stop()


class PiperTextToSpeech:
    """Speak text with a locally installed Piper executable and voice model."""

    def __init__(self, model_path: Path, executable: str = "piper") -> None:
        self.model_path = model_path
        self.executable = executable
        self._stop = threading.Event()

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        executable = shutil.which(self.executable)
        if executable is None:
            raise VoiceError("Piper is not installed or is not available on PATH.")
        if not self.model_path.exists():
            raise VoiceError(f"Piper voice model was not found: {self.model_path}")
        if os.name != "nt":
            raise VoiceError("Piper playback is currently implemented for Windows only.")
        import winsound

        self._stop.clear()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
            output_path = Path(output.name)
        try:
            subprocess.run(
                [executable, "--model", str(self.model_path), "--output_file", str(output_path)],
                input=text,
                text=True,
                check=True,
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if self._stop.is_set():
                return
            with wave.open(str(output_path), "rb") as audio:
                duration = audio.getnframes() / float(audio.getframerate() or 1)
            winsound.PlaySound(str(output_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            deadline = time.monotonic() + duration + 0.2
            while time.monotonic() < deadline:
                if self._stop.wait(0.05):
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    break
        except subprocess.CalledProcessError as error:
            raise VoiceError("Piper could not synthesize the response.") from error
        finally:
            output_path.unlink(missing_ok=True)

    def stop(self) -> None:
        self._stop.set()


def build_text_to_speech(
    engine: str,
    *,
    voice_hint: str = "",
    rate: int = 1,
    piper_executable: str = "piper",
    piper_model_path: str = "",
) -> TextToSpeech | None:
    """Return the best available engine for the configured preference."""

    engine = engine.strip().lower()
    if engine == "none":
        return None
    if engine == "piper" and piper_model_path and shutil.which(piper_executable):
        return PiperTextToSpeech(Path(piper_model_path), piper_executable)
    if engine == "piper":
        LOGGER.warning("Piper is not fully configured; falling back to Windows speech")
    if SapiTextToSpeech.available():
        return SapiTextToSpeech(voice_hint, rate)
    if PowerShellTextToSpeech.available():
        return PowerShellTextToSpeech(voice_hint, rate)
    LOGGER.warning("No text-to-speech engine is available on this system")
    return None
