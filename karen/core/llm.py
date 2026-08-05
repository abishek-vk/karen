"""LLM abstraction for a local Ollama instance."""

from __future__ import annotations

from dataclasses import dataclass

from karen.ai.ollama import ChatMessage, OllamaClient
from karen.exceptions import LocalAIError


KAREN_SYSTEM_PROMPT = """You are Karen, a privacy-first offline AI companion.

Your responsibilities:
* Help the user with computer tasks.
* Protect user privacy.
* Never perform unsafe actions.
* Never directly execute unknown commands.
* Always convert actions into structured intents.
* Ask confirmation for sensitive operations.

Be concise, helpful, and natural.

When a user asks for a computer action, return only a JSON object with these
fields: intent, target, parameters, requires_confirmation. Supported intents are
open_application, close_application, system_information, and general_chat.
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
