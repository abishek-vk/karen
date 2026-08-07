"""Registry for trusted, explicitly declared Karen skills."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from karen.skills import app_control, system
from karen.skills.base import SkillManifest, SkillResult

from .intent import Intent

SkillHandler = Callable[[Intent], SkillResult]


@dataclass(frozen=True, slots=True)
class RegisteredSkill:
    manifest: SkillManifest
    handler: SkillHandler


class SkillRegistry:
    """Resolve intents only to skills registered by application code."""

    def __init__(self) -> None:
        self._skills: dict[str, RegisteredSkill] = {}
        self._names: set[str] = set()

    def register(self, skill: RegisteredSkill) -> None:
        if skill.manifest.name in self._names:
            raise ValueError(f"Skill already registered: {skill.manifest.name}")
        for intent_name in skill.manifest.intents:
            if intent_name in self._skills:
                raise ValueError(f"Intent already registered: {intent_name}")
        self._names.add(skill.manifest.name)
        for intent_name in skill.manifest.intents:
            self._skills[intent_name] = skill

    def resolve(self, intent_name: str) -> RegisteredSkill | None:
        return self._skills.get(intent_name)

    @classmethod
    def default(cls) -> "SkillRegistry":
        """Build the trusted registry shipped with Karen."""

        registry = cls()
        registry.register(
            RegisteredSkill(
                SkillManifest(
                    name="app_control",
                    intents=frozenset({"open_application", "close_application"}),
                    capabilities=frozenset(
                                            {
                                                "launch_allowlisted_application",
                                                "close_allowlisted_application",
                                            }
                                        ),
                    risk_level="medium",
                    requires_confirmation=False,
                ),
                lambda intent: (
                    app_control.open_application(intent.target)
                    if intent.name == "open_application"
                    else app_control.close_application(intent.target)
                ),
            )
        )
        registry.register(
            RegisteredSkill(
                SkillManifest(
                    name="system_information",
                    intents=frozenset({"system_information"}),
                    capabilities=frozenset({"read_system_information"}),
                    risk_level="low",
                ),
                lambda intent: _system_result(),
            )
        )
        registry.register(
            RegisteredSkill(
                SkillManifest(
                    name="conversation",
                    intents=frozenset({"general_chat"}),
                    capabilities=frozenset({"generate_local_response"}),
                    risk_level="low",
                ),
                _general_chat,
            )
        )
        return registry


def _system_result() -> SkillResult:
    result = system.get_system_information()
    return SkillResult(result.success, result.message)


def _general_chat(intent: Intent) -> SkillResult:
    response = intent.parameters.get("response")
    if isinstance(response, str) and response.strip():
        return SkillResult(True, response.strip())
    return SkillResult(False, "I received an empty general response from the local model.")
