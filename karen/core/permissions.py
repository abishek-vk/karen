"""Capability-based policy checks for intent execution."""

from __future__ import annotations

from dataclasses import dataclass

from karen.skills.base import SkillManifest

from .intent import Intent


CONFIRMATION_REQUIRED_INTENTS = frozenset(
    {
        "close_application",
        "send_whatsapp_message",
        "clear_memories",
        "clear_conversation",
        "clear_reminders",
    }
)


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """The policy result returned before a skill can execute."""

    allowed: bool
    requires_confirmation: bool = False
    message: str = ""


class PermissionService:
    """Apply application policy independently of model-produced intent flags."""

    def evaluate(self, intent: Intent, manifest: SkillManifest) -> PermissionDecision:
        if intent.name not in manifest.intents:
            return PermissionDecision(False, message="This skill does not support that action.")
        if not manifest.capabilities:
            return PermissionDecision(False, message="This skill has no declared capabilities.")
        return PermissionDecision(
            allowed=True,
            requires_confirmation=(
                manifest.requires_confirmation
                or intent.requires_confirmation
                or intent.name in CONFIRMATION_REQUIRED_INTENTS
            ),
        )
