"""Short-lived in-memory confirmation state for sensitive actions."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .intent import Intent


@dataclass(frozen=True, slots=True)
class PendingConfirmation:
    intent: Intent
    expires_at: datetime


class ConfirmationManager:
    """Store at most one pending action and never persist it to disk."""

    def __init__(self, ttl_seconds: float = 45.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._pending: PendingConfirmation | None = None
        self._lock = threading.Lock()

    def request(self, intent: Intent) -> None:
        """Replace any existing pending action with this exact intent."""

        with self._lock:
            self._pending = PendingConfirmation(
                intent=intent,
                expires_at=datetime.now(timezone.utc) + self._ttl,
            )

    def peek(self) -> Intent | None:
        """Return the pending intent unless its short lifetime has elapsed."""

        with self._lock:
            pending = self._pending
            if pending is None:
                return None
            if datetime.now(timezone.utc) >= pending.expires_at:
                self._pending = None
                return None
            return pending.intent

    def consume(self) -> Intent | None:
        """Return and clear the pending intent."""

        intent = self.peek()
        self.clear()
        return intent

    def clear(self) -> None:
        with self._lock:
            self._pending = None


_YES = frozenset({
    "yes", "y", "yeah", "yep", "yup", "yes please", "sure", "ok", "okay", "confirm", "confirmed",
    "do it", "go ahead", "go for it", "send it", "yes send it", "yes do it", "yes go ahead",
    "please do", "absolutely", "of course", "affirmative", "correct", "that's right", "proceed",
})
_NO = frozenset({
    "no", "n", "nope", "nah", "no thanks", "no thank you", "cancel", "cancel it", "stop",
    "don't", "do not", "dont", "never mind", "nevermind", "forget it", "abort", "negative",
    "no don't", "don't send it", "don't do it",
})


def parse_confirmation(text: str) -> bool | None:
    """Parse affirmative or negative replies, including natural spoken forms."""

    normalized = re.sub(r"[^a-z' ]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^(?:karen|jarvis) ", "", normalized)
    if normalized in _YES:
        return True
    if normalized in _NO:
        return False
    first = normalized.split(" ", 1)[0] if normalized else ""
    if first in {"yes", "yeah", "yep", "sure", "okay", "ok"} and "not" not in normalized and "don't" not in normalized:
        return True
    if first in {"no", "nope", "nah", "cancel", "stop"}:
        return False
    return None
