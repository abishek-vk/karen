"""Small, dependency-free client for a local Ollama server."""

from __future__ import annotations

import json
import logging
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

    def is_available(self) -> bool:
        """Return whether the local Ollama service responds to a health request."""

        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, 5.0)):
                return True
        except (HTTPError, URLError, OSError):
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
