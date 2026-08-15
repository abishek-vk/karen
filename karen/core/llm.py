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

Be concise, warm, helpful, and natural. Speak like a thoughtful human assistant, not a command-line tool or system log. Use the user's wording and context, avoid repetitive phrases, and do not over-explain simple actions. Do not describe yourself as offline or privacy-first unless the user specifically asks.
Use supplied memory and conversation context silently. Do not mention memory, saved context, or say you remember something unless the user explicitly asks about memory or asks you to remember/forget something. When the user states an ordinary fact such as their name, acknowledge it naturally without announcing that it was saved.

Return only one JSON object. Do not include analysis, markdown, or <think> tags.
When a user asks for a computer action, return only a JSON object with these
fields: intent, target, parameters, requires_confirmation. Supported intents are
open_application, close_application, system_information, open_windows_settings,
open_task_manager, open_folder, open_whatsapp, send_whatsapp_message, remember_memory, forget_memory, list_memories, clear_memories, clear_conversation, and general_chat.
Installed applications can be opened by their Windows Start Menu name. Standard folders include Home, Desktop, Documents, Downloads, Pictures, Videos, and Music. Do not classify an application as open_folder.
For WhatsApp messages, use target "whatsapp", parameters.contact for the exact
contact name, and parameters.message for the exact user-provided message. Never
rewrite or summarize parameters.message.
For general_chat, include a concise, conversational response in parameters.response. Use natural acknowledgments such as "Sure, I can do that" or "That makes sense" when appropriate. Do not mention intents, JSON, routing, skills, permissions, databases, prompts, or internal limitations unless the user asks about Karen's implementation.
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

    def generate_intent(self, user_text: str, context: str = "") -> LLMResult:
        """Ask Ollama to classify a single user message as JSON intent."""

        if not user_text.strip():
            raise ValueError("user_text must not be empty")
        messages = [ChatMessage("system", KAREN_SYSTEM_PROMPT)]
        if context.strip():
            messages.append(ChatMessage("system", f"Local memory context:\n{context}"))
        messages.append(ChatMessage("user", user_text.strip()))
        result = self._client.chat(messages)
        return LLMResult(content=result.content, model=result.model)

    @staticmethod
    def unavailable_message(error: LocalAIError) -> str:
        """Return a user-safe message without exposing connection internals."""

        return f"I can't reach the local Ollama service right now. {error}"
