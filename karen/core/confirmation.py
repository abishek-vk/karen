"""Short-lived in-memory confirmation state for sensitive actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .intent import Intent


@dataclass(frozen=True, slots=True)
class PendingConfirmation:
    intent: Intent
    expires_at: datetime


class ConfirmationManager:
    """Store at most one pending action and never persist it to disk."""

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._pending: PendingConfirmation | None = None

    def request(self, intent: Intent) -> None:
        """Replace any existing pending action with this exact intent."""

        self._pending = PendingConfirmation(
            intent=intent,
            expires_at=datetime.now(timezone.utc) + self._ttl,
        )

    def peek(self) -> Intent | None:
        """Return the pending intent unless its short lifetime has elapsed."""

        pending = self._pending
        if pending is None:
            return None
        if datetime.now(timezone.utc) >= pending.expires_at:
            self.clear()
            return None
        return pending.intent

    def consume(self) -> Intent | None:
        """Return and clear the pending intent."""

        intent = self.peek()
        self.clear()
        return intent

    def clear(self) -> None:
        self._pending = None


def parse_confirmation(text: str) -> bool | None:
    """Parse only unambiguous affirmative or negative responses."""

    normalized = text.strip().lower().rstrip(".!?")
    if normalized in {"yes", "y", "confirm", "do it", "go ahead"}:
        return True
    if normalized in {"no", "n", "cancel", "stop", "don't"}:
        return False
    return None
