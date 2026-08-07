"""Shared contracts for independently registered skills."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SkillResult:
    """Stable user-facing result returned by every skill."""

    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Capabilities and safety policy declared by a skill."""

    name: str
    intents: frozenset[str]
    capabilities: frozenset[str]
    risk_level: str = "low"
    requires_confirmation: bool = False
