"""Route validated intents through the registry and permission policy."""

from __future__ import annotations

from karen.skills.base import SkillResult

from .intent import Intent
from .permissions import PermissionService
from .skill_registry import SkillRegistry


class IntentRouter:
    """Execute only registered skills approved by capability policy."""

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        permissions: PermissionService | None = None,
    ) -> None:
        self.registry = registry or SkillRegistry.default()
        self.permissions = permissions or PermissionService()

    def needs_confirmation(self, intent: Intent) -> bool:
        """Return the policy decision without executing the skill."""

        registered = self.registry.resolve(intent.name)
        if registered is None:
            return False
        return self.permissions.evaluate(intent, registered.manifest).requires_confirmation

    def route(self, intent: Intent, *, confirmed: bool = False) -> SkillResult:
        """Execute an approved intent and return a stable skill result."""

        registered = self.registry.resolve(intent.name)
        if registered is None:
            return SkillResult(False, "I don't have a safe skill for that request.")

        decision = self.permissions.evaluate(intent, registered.manifest)
        if not decision.allowed:
            return SkillResult(False, decision.message)
        if decision.requires_confirmation and not confirmed:
            return SkillResult(False, "This action requires your confirmation before I can continue.")
        return registered.handler(intent)
