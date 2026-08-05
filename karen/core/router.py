"""Dispatch validated intents to explicitly implemented skills."""

from __future__ import annotations

from .intent import Intent
from karen.skills import app_control, system
from karen.skills.app_control import SkillResult


class IntentRouter:
    """Map intent names to skills; unknown intents never reach execution."""

    @staticmethod
    def needs_confirmation(intent: Intent) -> bool:
        """Apply policy in addition to the model's requested confirmation."""

        return intent.requires_confirmation or intent.name in {"close_application"}

    def route(self, intent: Intent, *, confirmed: bool = False) -> SkillResult:
        """Execute an approved intent and return a user-facing result."""

        if self.needs_confirmation(intent) and not confirmed:
            return SkillResult(False, "This action requires your confirmation before I can continue.")
        if intent.name == "open_application":
            return app_control.open_application(intent.target)
        if intent.name == "close_application":
            return app_control.close_application(intent.target)
        if intent.name == "system_information":
            result = system.get_system_information()
            return SkillResult(result.success, result.message)
        if intent.name == "general_chat":
            response = intent.parameters.get("response")
            if isinstance(response, str) and response.strip():
                return SkillResult(True, response.strip())
            return SkillResult(False, "I received an empty general response from the local model.")
        # Defensive fallback in case Intent validation changes without router updates.
        return SkillResult(False, "I don't have a safe skill for that request.")
