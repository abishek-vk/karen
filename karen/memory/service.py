"""High-level, failure-tolerant memory operations for Karen."""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from karen.core.intent import Intent
from karen.skills.base import SkillResult

from .database import MemoryDatabase
from .models import LongTermMemory
from .retrieval import format_context, retrieve_relevant

LOGGER = logging.getLogger(__name__)


class MemoryService:
    """Persist conversations and explicit long-term memories locally.

    Database errors are intentionally contained here. Memory is an enhancement,
    so a locked, unavailable, or corrupt database must not stop Karen's other
    skills from working.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        enabled: bool = True,
        session_id: str | None = None,
        recent_message_limit: int = 15,
        max_memory_results: int = 5,
    ) -> None:
        self.enabled = enabled and path is not None
        self.session_id = session_id or str(uuid.uuid4())
        self.recent_message_limit = max(0, recent_message_limit)
        self.max_memory_results = max(0, max_memory_results)
        self.database: MemoryDatabase | None = None
        if self.enabled and path is not None:
            try:
                self.database = MemoryDatabase(Path(path))
                self.database.create_conversation(self.session_id, self._now())
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
                self._disable(error)

    @property
    def available(self) -> bool:
        return self.enabled and self.database is not None

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _disable(self, error: Exception) -> None:
        LOGGER.warning("Karen memory disabled: %s", error)
        self.database = None
        self.enabled = False

    def _run(self, operation, default):
        if not self.available:
            return default
        try:
            return operation()
        except (OSError, RuntimeError, ValueError, TypeError, sqlite3.Error):
            LOGGER.exception("Karen memory operation failed")
            return default

    def record_message(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"} or not content.strip():
            return
        database = self.database
        if not self.available or database is None:
            return
        clean = content.strip()
        self._run(
            lambda: database.add_message(self.session_id, role, clean, self._now()),
            None,
        )
        if role == "user":
            self._extract_safe_fact(clean)

    def _extract_safe_fact(self, content: str) -> None:
        """Promote only narrowly recognizable, user-stated facts."""
        if re.match(r"(?:please )?(?:remember|save)", content, re.IGNORECASE):
            return
        fact_patterns = (
            r"my name is .+",
            r"my (?:favorite|preferred) .+ is .+",
            r"this project is called .+",
            r"i(?:'m| am) building .+",
        )
        if any(re.fullmatch(pattern, content.strip().rstrip(".!?"), re.IGNORECASE) for pattern in fact_patterns):
            self.remember(content)

    def context_for(self, query: str) -> str:
        database = self.database
        if not self.available or database is None:
            return ""
        memories, messages = self._run(
            lambda: retrieve_relevant(
                database,
                query,
                self.session_id,
                self.max_memory_results,
                self.recent_message_limit,
            ),
            ([], []),
        )
        return format_context(memories, messages)

    @staticmethod
    def _category(content: str) -> str:
        lowered = content.lower()
        if "prefer" in lowered or "favorite" in lowered or "favourite" in lowered:
            return "preference"
        if "project" in lowered or "building" in lowered or "called" in lowered:
            return "project"
        return "fact"

    def remember(self, content: str, category: str | None = None) -> LongTermMemory | None:
        clean = re.sub(r"\s+", " ", content.strip()).strip(" .!?\n")
        if not clean:
            return None
        database = self.database
        if not self.available or database is None:
            return None
        return self._run(
            lambda: database.add_or_update_memory(
                category or self._category(clean), clean, 1, self._now()
            ),
            None,
        )

    def forget(self, query: str) -> int:
        clean = re.sub(r"\s+", " ", query.strip()).strip(" .!?\n")
        if not clean:
            return 0
        # Search terms make "forget my editor" useful without requiring the
        # user to repeat the stored sentence verbatim.
        terms = [term for term in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+#.-]{2,}", clean.lower())]
        database = self.database
        if not self.available or database is None:
            return 0
        return self._run(lambda: database.delete_memories_matching(terms or [clean]), 0)

    def list_memories(self) -> list[LongTermMemory]:
        database = self.database
        if not self.available or database is None:
            return []
        return self._run(lambda: database.all_memories(self.max_memory_results), [])

    def clear_memories(self) -> int:
        database = self.database
        if not self.available or database is None:
            return 0
        return self._run(lambda: database.clear_memories(), 0)

    def clear_conversation(self) -> int:
        database = self.database
        if not self.available or database is None:
            return 0
        return self._run(lambda: database.clear_conversation(self.session_id), 0)

    def handle_intent(self, intent: Intent) -> SkillResult:
        """Execute a validated memory intent through the normal skill router."""
        if not self.available:
            return SkillResult(False, "Local memory is unavailable, but I can still handle other requests.")
        if intent.name == "remember_memory":
            content = intent.parameters.get("content", "")
            memory = self.remember(content if isinstance(content, str) else "")
            return SkillResult(bool(memory), "Got it. I’ll remember that." if memory else "Tell me what you want me to remember.")
        if intent.name == "forget_memory":
            query = intent.parameters.get("query", "")
            count = self.forget(query if isinstance(query, str) else "")
            return SkillResult(True, "I forgot that memory." if count else "I couldn’t find a matching memory.")
        if intent.name == "list_memories":
            memories = self.list_memories()
            if not memories:
                return SkillResult(True, "I don’t have any saved memories.")
            return SkillResult(True, "I remember:\n" + "\n".join(f"- {memory.content}" for memory in memories))
        if intent.name == "clear_memories":
            count = self.clear_memories()
            return SkillResult(True, f"Cleared {count} saved memor{'y' if count == 1 else 'ies'}.")
        if intent.name == "clear_conversation":
            count = self.clear_conversation()
            return SkillResult(True, f"Cleared {count} messages from this conversation.")
        return SkillResult(False, "I don’t have a safe memory operation for that request.")
