"""Shared application service used by text and graphical interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from karen.exceptions import LocalAIError
from karen.memory import MemoryService
from karen.skills.base import SkillManifest

from .confirmation import ConfirmationManager, parse_confirmation
from .fast_path import parse_fast_path
from .intent import Intent, parse_intent
from .llm import LocalLLM
from .router import IntentRouter
from .skill_registry import RegisteredSkill


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    """User-facing response plus execution metadata for interfaces."""

    message: str
    source: str
    success: bool


def _confirmation_prompt(intent: Intent) -> str:
    if intent.name == "send_whatsapp_message":
        contact = intent.parameters.get("contact", "the selected contact")
        message = intent.parameters.get("message", "")
        return (
            f"Send this exact message to {contact}?\\n"
            f'“{message}”\\n'
            "Say yes to send or no to cancel."
        )
    return "This action requires your confirmation. Say yes to continue or no to cancel."


class ApplicationService:
    """Coordinate input, policy, local AI, and trusted skill execution."""

    def __init__(
        self,
        llm: LocalLLM,
        router: IntentRouter,
        *,
        memory_path: Path | None = None,
        memory_enabled: bool = True,
        recent_message_limit: int = 15,
        max_memory_results: int = 5,
        memory: MemoryService | None = None,
    ) -> None:
        self.llm = llm
        self.router = router
        self.confirmations = ConfirmationManager()
        self.memory = memory or MemoryService(
            memory_path,
            enabled=memory_enabled,
            recent_message_limit=recent_message_limit,
            max_memory_results=max_memory_results,
        )
        if self.router.registry.resolve("remember_memory") is None:
            self.router.registry.register(
                RegisteredSkill(
                    SkillManifest(
                        name="memory",
                        intents=frozenset({
                            "remember_memory", "forget_memory", "list_memories",
                            "clear_memories", "clear_conversation",
                        }),
                        capabilities=frozenset({"local_memory"}),
                        risk_level="low",
                    ),
                    self.memory.handle_intent,
                )
            )

    def _finish(self, response: ServiceResponse) -> ServiceResponse:
        self.memory.record_message("assistant", response.message)
        return response

    def handle_text(self, text: str) -> ServiceResponse:
        """Process one message without exposing orchestration to the UI."""

        try:
            self.memory.record_message("user", text)
            pending = self.confirmations.peek()
            if pending is not None:
                answer = parse_confirmation(text)
                if answer is True:
                    approved = self.confirmations.consume()
                    if approved is not None:
                        result = self.router.route(approved, confirmed=True)
                        return self._finish(ServiceResponse(result.message, "confirmed", result.success))
                elif answer is False:
                    self.confirmations.clear()
                    return self._finish(ServiceResponse("Cancelled.", "confirmation", True))
                else:
                    return self._finish(ServiceResponse(
                        "Please answer yes or no before I continue with the pending action.",
                        "confirmation",
                        False,
                    ))

            intent = parse_fast_path(text)
            source = "fast_path"
            if intent is None:
                raw = self.llm.generate_intent(text, self.memory.context_for(text))
                intent = parse_intent(raw.content)
                source = "local_llm"
            if self.router.needs_confirmation(intent):
                self.confirmations.request(intent)
                return self._finish(ServiceResponse(
                    _confirmation_prompt(intent),
                    "confirmation",
                    False,
                ))
            result = self.router.route(intent)
            return self._finish(ServiceResponse(result.message, source, result.success))
        except (ValueError, LocalAIError) as error:
            return self._finish(ServiceResponse(
                f"I couldn't safely process that request: {error}",
                "error",
                False,
            ))
