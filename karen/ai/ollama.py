"""Small, dependency-free client for a local Ollama server."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..exceptions import LocalAIError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A message in a local model conversation."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """The useful, provider-independent part of an Ollama response."""

    content: str
    model: str
    done: bool


class OllamaClient:
    """Call Ollama over its local HTTP API without cloud dependencies."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:4b",
        timeout_seconds: float = 60.0,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ollama must use a local HTTP endpoint")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not model.strip():
            raise ValueError("model must not be empty")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._server_process: subprocess.Popen[bytes] | None = None

    def is_available(self) -> bool:
        """Return whether the local Ollama service responds to a health request."""

        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, 5.0)):
                return True
        except (HTTPError, URLError, OSError):
            return False

    def has_model(self) -> bool:
        """Return whether the configured model is installed locally."""

        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, 5.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("models", [])
            if not isinstance(models, list):
                return False
            return any(
                isinstance(model, dict) and model.get("name") == self.model
                for model in models
            )
        except (HTTPError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False

    def ensure_available(self, wait_seconds: float = 8.0) -> bool:
        """Start local Ollama if needed and wait briefly for its API."""

        if self.is_available():
            return True
        executable = shutil.which("ollama")
        if executable is None:
            LOGGER.info("Ollama executable was not found on PATH")
            return False
        if self._server_process is None or self._server_process.poll() is not None:
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                self._server_process = subprocess.Popen(
                    [executable, "serve"],
                    shell=False,
                    close_fds=True,
                    creationflags=creation_flags,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                LOGGER.exception("Could not start the local Ollama server")
                return False
        deadline = time.monotonic() + max(wait_seconds, 0.1)
        while time.monotonic() < deadline:
            if self.is_available():
                return True
            time.sleep(0.2)
        return False

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Generate a response from the local model.

        This method only returns text. It does not execute commands or interpret
        model output as an instruction, keeping execution in a future skill layer.
        """

        if not messages:
            raise ValueError("messages must not be empty")
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
            "format": "json",
            # Keep routing responses bounded on consumer CPUs.
            "options": {
                "temperature": 0.1,
                "num_predict": 256,
            },
            # Supported by recent Ollama versions for Qwen3.
            "think": False,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            LOGGER.warning("Ollama returned HTTP %s", error.code)
            raise LocalAIError(f"Ollama request failed with HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise LocalAIError("Ollama is unavailable; ensure it is running locally") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LocalAIError("Ollama returned an invalid response") from error

        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LocalAIError("Ollama response did not contain message content")
        return ChatResponse(
            content=message["content"],
            model=str(result.get("model", self.model)),
            done=bool(result.get("done", False)),
        )
