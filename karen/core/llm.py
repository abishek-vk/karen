"""LLM abstraction for a local Ollama instance."""

from __future__ import annotations

from dataclasses import dataclass

from karen.ai.ollama import ChatMessage, OllamaClient
from karen.exceptions import LocalAIError


KAREN_SYSTEM_PROMPT = """You are Karen, a local Windows AI companion.

Your responsibilities:
* Help the user with computer tasks.
* Protect user data.
* Never perform unsafe actions.
* Never directly execute unknown commands.
* Always convert actions into structured intents.
* Ask confirmation for sensitive operations.

Be concise, helpful, and natural. Do not describe yourself as offline or privacy-first unless the user specifically asks.

Return only one JSON object. Do not include analysis, markdown, or <think> tags.
When a user asks for a computer action, return only a JSON object with these
fields: intent, target, parameters, requires_confirmation. Supported intents are
open_application, close_application, system_information, open_windows_settings,
open_task_manager, open_folder, and general_chat. Allowed applications
include Notepad, Calculator, Chrome, VS Code, and Edge.
For general_chat, include a concise response in parameters.response.
"""


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Raw model output plus the configured model name."""

    content: str
    model: str


class LocalLLM:
    """Replaceable text generation boundary backed by Ollama."""

    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    def is_available(self) -> bool:
        """Return whether the configured local model service is reachable."""

        return self._client.is_available()

    def ensure_available(self) -> bool:
        """Start the local Ollama server when it is not already running."""

        return self._client.ensure_available()

    def has_model(self) -> bool:
        """Return whether the configured model is installed locally."""

        return self._client.has_model()

    def generate_intent(self, user_text: str) -> LLMResult:
        """Ask Ollama to classify a single user message as JSON intent."""

        if not user_text.strip():
            raise ValueError("user_text must not be empty")
        result = self._client.chat(
            [
                ChatMessage("system", KAREN_SYSTEM_PROMPT),
                ChatMessage("user", user_text.strip()),
            ]
        )
        return LLMResult(content=result.content, model=result.model)

    @staticmethod
    def unavailable_message(error: LocalAIError) -> str:
        """Return a user-safe message without exposing connection internals."""

        return f"I can't reach the local Ollama service right now. {error}"
