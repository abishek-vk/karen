"""Optional local Piper text-to-speech adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .contracts import VoiceError


class PiperTextToSpeech:
    """Speak text with a locally installed Piper executable and voice model."""

    def __init__(self, model_path: Path, executable: str = "piper") -> None:
        self.model_path = model_path
        self.executable = executable

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        executable = shutil.which(self.executable)
        if executable is None:
            raise VoiceError("Piper is not installed or is not available on PATH.")
        if not self.model_path.exists():
            raise VoiceError(f"Piper voice model was not found: {self.model_path}")

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
            )
            if os.name != "nt":
                raise VoiceError("Piper playback is currently implemented for Windows only.")
            import winsound

            winsound.PlaySound(str(output_path), winsound.SND_FILENAME)
        except subprocess.CalledProcessError as error:
            raise VoiceError("Piper could not synthesize the response.") from error
        finally:
            output_path.unlink(missing_ok=True)
