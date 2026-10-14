"""Small, dependency-free client for a local Ollama server."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
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
        *,
        context_window: int = 4096,
        keep_alive: str = "20m",
        num_thread: int = 0,
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
        self.context_window = max(1024, context_window)
        self.keep_alive = keep_alive
        self.num_thread = max(0, num_thread)
        self._server_process: subprocess.Popen[bytes] | None = None
        self._start_lock = threading.Lock()

    # -- health -----------------------------------------------------------

    def _tags(self) -> list[dict] | None:
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, 3.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return [model for model in models if isinstance(model, dict)] if isinstance(models, list) else []

    def is_available(self) -> bool:
        """Return whether the local Ollama service responds to a health request."""

        return self._tags() is not None

    def has_model(self) -> bool:
        """Return whether the configured model is installed (``x`` matches ``x:latest``)."""

        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return any(model.get("name") in {self.model, wanted} for model in self._tags() or [])

    def use_installed_model(self) -> str | None:
        """Keep the configured model if installed, else switch to an installed one.

        Picks the smallest installed chat model, because it is the fastest on a
        CPU. Returns the model in use, or ``None`` when no chat model is installed.
        """

        tags = self._tags()
        if tags is None:
            return None
        names = {model.get("name") for model in tags}
        if self.model in names or f"{self.model}:latest" in names:
            return self.model
        chat_models = [
            model for model in tags
            if isinstance(model.get("name"), str) and "embed" not in model["name"].lower()
        ]
        if not chat_models:
            return None
        chosen = min(chat_models, key=lambda model: model.get("size") or 0)["name"]
        LOGGER.warning("Model %s is not installed; using installed model %s instead", self.model, chosen)
        self.model = chosen
        return chosen

    def ensure_available(self, wait_seconds: float = 8.0) -> bool:
        """Start local Ollama once if needed and wait briefly for its API."""

        if self.is_available():
            return True
        with self._start_lock:
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
                time.sleep(0.4)
        return False

    def warm_up(self) -> None:
        """Load the model into RAM ahead of the first question (no tokens generated)."""

        payload = {"model": self.model, "keep_alive": self.keep_alive}
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response.read()
        except (HTTPError, URLError, OSError) as error:
            LOGGER.info("Model warm-up skipped: %s", error)

    def shutdown(self) -> None:
        """Stop an Ollama server that Karen started itself."""

        process = self._server_process
        if process is not None and process.poll() is None:
            process.terminate()
        self._server_process = None

    # -- generation -------------------------------------------------------

    def _request(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool,
        json_mode: bool,
        num_predict: int,
        temperature: float,
    ) -> Request:
        if not messages:
            raise ValueError("messages must not be empty")
        options: dict[str, object] = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": self.context_window,
        }
        if self.num_thread:
            options["num_thread"] = self.num_thread
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": stream,
            "options": options,
            # Qwen3 and other reasoning models: answer directly, no hidden thinking.
            "think": False,
            "keep_alive": self.keep_alive,
        }
        if json_mode:
            payload["format"] = "json"
        return Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        json_mode: bool = True,
        num_predict: int = 256,
        temperature: float = 0.1,
    ) -> ChatResponse:
        """Generate one complete response. Output is text only and never executed."""

        request = self._request(
            messages, stream=False, json_mode=json_mode, num_predict=num_predict, temperature=temperature
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            LOGGER.warning("Ollama returned HTTP %s", error.code)
            raise LocalAIError(_http_error_message(error.code, self.model)) from error
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

    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        num_predict: int = 400,
        temperature: float = 0.4,
    ) -> Iterator[str]:
        """Yield response text as it is generated.

        Closing the generator early closes the HTTP connection, which makes
        Ollama stop generating, so callers can abandon an answer cheaply.
        """

        request = self._request(
            messages, stream=True, json_mode=False, num_predict=num_predict, temperature=temperature
        )
        try:
            response = urlopen(request, timeout=self.timeout_seconds)
        except HTTPError as error:
            LOGGER.warning("Ollama returned HTTP %s", error.code)
            raise LocalAIError(_http_error_message(error.code, self.model)) from error
        except (URLError, TimeoutError, OSError) as error:
            raise LocalAIError("Ollama is unavailable; ensure it is running locally") from error
        with response:
            try:
                for line in response:
                    if not line.strip():
                        continue
                    data = json.loads(line.decode("utf-8"))
                    if "error" in data:
                        raise LocalAIError(f"Ollama error: {data['error']}")
                    message = data.get("message") or {}
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        yield content
                    if data.get("done"):
                        return
            except (TimeoutError, OSError) as error:
                raise LocalAIError("The local model stopped responding") from error
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise LocalAIError("Ollama returned an invalid streamed response") from error


def _http_error_message(code: int, model: str) -> str:
    if code == 404:
        return f"The model {model} is not installed. Run: ollama pull {model}"
    return f"Ollama request failed with HTTP {code}"
