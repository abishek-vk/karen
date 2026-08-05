
import json
from urllib.request import Request

import pytest

from karen.ai.ollama import ChatMessage, OllamaClient
from karen.exceptions import LocalAIError


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.payload


def test_client_rejects_non_local_endpoint() -> None:
    with pytest.raises(ValueError):
        OllamaClient("https://example.com")


def test_chat_posts_only_to_local_ollama(monkeypatch) -> None:
    captured: dict[str, Request] = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse({"model": "qwen3:4b", "message": {"content": "Hello"}, "done": True})

    monkeypatch.setattr("karen.ai.ollama.urlopen", fake_urlopen)
    response = OllamaClient().chat([ChatMessage("user", "Hi")])

    assert response.content == "Hello"
    assert captured["request"].full_url == "http://127.0.0.1:11434/api/chat"
    body = json.loads(captured["request"].data.decode("utf-8"))
    assert body["stream"] is False
    assert captured["timeout"] == 60.0


def test_invalid_response_raises_local_ai_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "karen.ai.ollama.urlopen",
        lambda request, timeout: FakeResponse({"model": "qwen3:4b"}),
    )

    with pytest.raises(LocalAIError):
        OllamaClient().chat([ChatMessage("user", "Hi")])
