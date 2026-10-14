"""Persistent reminders announced by a single sleeping timer thread."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from karen.core.intent import Intent
from karen.core.timeparse import describe, parse_when
from karen.skills.base import SkillResult

from .service import MemoryService

LOGGER = logging.getLogger(__name__)


class ReminderService:
    """Store reminders in SQLite and call ``notify`` when each one is due.

    The worker sleeps until the next due reminder (or until a new one is added)
    instead of polling, so it costs no CPU while waiting.
    """

    def __init__(self, memory: MemoryService, notify: Callable[[str], None] | None = None) -> None:
        self.memory = memory
        self.notify = notify
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None and self.memory.available:
            self._thread = threading.Thread(target=self._run, name="karen-reminders", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()

    def add(self, text: str, due: datetime) -> str:
        due_utc = due.astimezone(UTC).isoformat()
        stored = self.memory._run(lambda db: db.add_reminder(text, due_utc, datetime.now(UTC).isoformat()), None)
        if stored is None:
            return ""
        self._wake.set()
        return describe(due)

    def pending(self) -> list[tuple[str, datetime]]:
        reminders = self.memory._run(lambda db: db.pending_reminders(), [])
        return [(reminder.text, datetime.fromisoformat(reminder.due_at).astimezone()) for reminder in reminders]

    def handle_intent(self, intent: Intent) -> SkillResult:
        if not self.memory.available:
            return SkillResult(False, "Reminders need the local memory store, which is unavailable.")
        if intent.name == "list_reminders":
            pending = self.pending()
            if not pending:
                return SkillResult(True, "You have no pending reminders.")
            lines = [f"- {text}, {describe(due)}" for text, due in pending[:10]]
            return SkillResult(True, "Your reminders:\n" + "\n".join(lines))
        if intent.name == "clear_reminders":
            count = self.memory._run(lambda db: db.clear_reminders(), 0)
            self._wake.set()
            return SkillResult(True, f"Cancelled {count} reminder{'s' if count != 1 else ''}.")
        if intent.name != "set_reminder":
            return SkillResult(False, "I don't have a safe reminder operation for that request.")

        text = str(intent.parameters.get("text") or "").strip(" .")
        when = str(intent.parameters.get("when") or "").strip()
        due: datetime | None = None
        if when:
            try:
                due = datetime.fromisoformat(when)
                if due.tzinfo is None:
                    due = due.astimezone()
            except ValueError:
                due, _ = parse_when(when)
        if due is None and text:
            due, remainder = parse_when(text)
            text = remainder or text
        if not text:
            return SkillResult(False, "What should I remind you about?")
        if due is None:
            # No usable time: keep it as a to-do rather than dropping it.
            detail = f"{text} {when}".strip() if when and when.lower() not in text.lower() else text
            if self.memory.remember(f"To-do: {detail}") is None:
                return SkillResult(False, "I couldn't save that.")
            return SkillResult(
                True,
                f"I've saved that: {detail}. If you want a reminder, tell me when, like in 2 hours or at 6 pm.",
            )
        if due <= datetime.now().astimezone():
            return SkillResult(False, "That time has already passed. Give me a time in the future.")
        description = self.add(text, due)
        if not description:
            return SkillResult(False, "I couldn't save that reminder.")
        return SkillResult(True, f"Reminder set {description}: {text}.")

    def _run(self) -> None:
        while not self._stopped.is_set():
            self._wake.clear()
            wait_seconds = 3600.0
            now = datetime.now(UTC)
            for reminder in self.memory._run(lambda db: db.pending_reminders(), []):
                due = datetime.fromisoformat(reminder.due_at)
                if due <= now:
                    self.memory._run(lambda db, rid=reminder.id: db.mark_reminder_delivered(rid), None)
                    late = (now - due).total_seconds() > 120
                    message = f"Reminder: {reminder.text}." + (" (This was due while I was offline.)" if late else "")
                    if self.notify is not None:
                        try:
                            self.notify(message)
                        except Exception:  # pragma: no cover - UI callback dependent
                            LOGGER.exception("Reminder notification failed")
                else:
                    wait_seconds = min(wait_seconds, (due - now).total_seconds())
                    break
            self._wake.wait(timeout=max(0.5, wait_seconds))
