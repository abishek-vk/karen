"""High-level, failure-tolerant memory operations for Karen."""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from karen.core.intent import Intent
from karen.skills.base import SkillResult

from .database import MemoryDatabase
from .models import LongTermMemory
from .retrieval import STOP_WORDS, earlier_statements, format_context, retrieve_memories

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

# Up to three words, stopping before "and", "but" or punctuation.
_NAME_STOP = r"(?:and|but|from|i|that|thats|that's|it|its|it's|is|my|please|ok|okay|now|here|for|as)\b"
_NAME = rf"(?!{_NAME_STOP})[a-z][a-z.'-]*(?: (?!{_NAME_STOP})[a-z][a-z.'-]*){{0,2}}"
# User statements that are safe to promote to long-term memory automatically.
_FACT_PATTERNS: tuple[tuple[str, str], ...] = (
    (rf"my name is (?P<v>{_NAME})", "identity"),
    (rf"(?:call me|you can call me|i am called|i'm called)(?: as)? (?P<v>{_NAME})", "identity"),
    (r"my (?:favorite|favourite|fav|fave|preferred) .+ (?:is|are) .+", "preference"),
    # "I love you" / "I like you" is said to Karen, not a fact about the user.
    (r"i (?:really )?(?:like|love|enjoy|prefer|hate|dislike) (?!(?:you|u|ya|karen|jarvis)\b).+", "preference"),
    (r"i(?:'m| am) allergic to .+", "fact"),
    (r"i (?:live|stay) in .+", "fact"),
    (r"i(?:'m| am) from .+", "fact"),
    (r"my birthday is .+", "fact"),
    (r"i was born (?:on|in) .+", "fact"),
    (r"i (?:work|study) (?:at|in|for|as) .+", "fact"),
    (r"my (?:best |close |childhood )?(?:wife|husband|partner|mom|mother|dad|father|brother|sister|son|daughter|"
     r"friend|boss|girlfriend|boyfriend|dog|cat|teacher|roommate|cousin|uncle|aunt|grandma|grandpa|grandmother|"
     r"grandfather|classmate|mentor|manager|crush|pet)(?:'?s)?(?: name)? is .+", "people"),
    (r"this project is called .+", "project"),
    (r"i(?:'m| am) (?:building|working on|developing) .+", "project"),
)
_PRONOUN_QUERIES = frozenset({"", "it", "that", "this"})
_THIRD_PERSON_VERBS = {
    "likes": "like", "loves": "love", "enjoys": "enjoy", "prefers": "prefer", "hates": "hate",
    "dislikes": "dislike", "lives": "live", "stays": "stay", "studies": "study", "works": "work", "is": "am",
}


def _main_clause(text: str) -> str:
    """Drop a trailing question or aside: "I like Tony Stark, do you know him?" -> "I like Tony Stark"."""

    clause = re.split(
        r"[;?]|,\s*(?=(?:do|did|does|can|could|is|are|was|what|who|how|why|its|it's|he's|she's|you)\b)",
        text.strip(), maxsplit=1, flags=re.IGNORECASE,
    )[0]
    return clause.strip().rstrip(".!")


def _first_person(sentence: str, name: str | None) -> str:
    """People often talk about themselves by name: "Abi's best friend is X" -> "my best friend is X"."""

    if not name:
        return sentence
    own = re.escape(name.split()[0])
    sentence = re.sub(rf"^{own}(?:'s|’s|s'|s| 's)\s+", "my ", sentence, flags=re.IGNORECASE)
    return re.sub(
        rf"^{own}\s+({'|'.join(_THIRD_PERSON_VERBS)})\b",
        lambda match: "i " + _THIRD_PERSON_VERBS[match.group(1).lower()],
        sentence,
        flags=re.IGNORECASE,
    )


class MemoryService:
    """Persist conversations and long-term memories locally.

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
        recent_message_limit: int = 6,
        max_memory_results: int = 5,
        max_stored_messages: int = 2000,
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
                self.database.prune_messages(max_stored_messages)
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

    def _run(self, operation: Callable[[MemoryDatabase], T], default: T) -> T:
        database = self.database
        if not self.available or database is None:
            return default
        try:
            return operation(database)
        except (OSError, RuntimeError, ValueError, TypeError, sqlite3.Error):
            LOGGER.exception("Karen memory operation failed")
            return default

    # -- conversation -----------------------------------------------------

    def record_message(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"} or not content.strip():
            return
        clean = content.strip()
        self._run(lambda db: db.add_message(self.session_id, role, clean, self._now()), None)
        if role == "user":
            self._extract_safe_fact(clean)

    def recent_turns(self) -> list[tuple[str, str]]:
        """Return the latest ``(role, content)`` turns of this session, oldest first."""

        if self.recent_message_limit <= 0:
            return []
        messages = self._run(lambda db: db.recent_messages(self.session_id, self.recent_message_limit), [])
        return [(message.role, message.content) for message in messages]

    def _extract_safe_fact(self, content: str) -> None:
        """Promote only narrowly recognizable, user-stated facts."""

        if re.match(r"(?:please )?(?:remember|save|note)\b", content, re.IGNORECASE):
            return
        sentence = _first_person(_main_clause(content), self.user_name())
        for pattern, category in _FACT_PATTERNS:
            if category == "identity":
                match = re.match(pattern + r"(?=$|[ ,.;!]|\s+(?:and|but)\b)", sentence, re.IGNORECASE)
            else:
                match = re.fullmatch(pattern, sentence, re.IGNORECASE)
            if not match:
                continue
            if category == "identity":
                name = " ".join(word.capitalize() if word.islower() else word for word in match.group("v").split())
                self.remember(f"My name is {name}", category="identity")
            else:
                self.remember(sentence, category=category)
            return

    def context_for(self, query: str) -> str:
        memories = self._run(lambda db: retrieve_memories(db, query, self.max_memory_results), [])
        statements = self._run(lambda db: earlier_statements(db, query), [])
        saved = " ".join(memory.content.lower() for memory in memories)
        # Skip statements already captured as saved facts.
        statements = [text for text in statements if text.lower().rstrip(".!") not in saved]
        return format_context(memories, statements)

    def user_name(self) -> str | None:
        identity = self._run(lambda db: db.memories_by_category("identity", 1), [])
        if not identity:
            return None
        match = re.match(r"my name is (.+)", identity[0].content, re.IGNORECASE)
        return match.group(1).strip() if match else None

    # -- long-term memory -------------------------------------------------

    @staticmethod
    def _category(content: str) -> str:
        lowered = content.lower()
        if re.match(r"my name is ", lowered):
            return "identity"
        if lowered.startswith("to-do:"):
            return "todo"
        if re.search(r"\b(?:prefer|favorite|favourite|like|love|hate)\b", lowered):
            return "preference"
        if re.search(r"\b(?:project|building)\b", lowered):
            return "project"
        return "fact"

    def remember(self, content: str, category: str | None = None) -> LongTermMemory | None:
        clean = re.sub(r"\s+", " ", content.strip()).strip(" .!?\n")
        if not clean:
            return None
        memory_category = category or self._category(clean)
        if memory_category == "identity":
            # Identity is singular: a corrected name replaces the old one.
            self._run(lambda db: db.delete_memories_by_category("identity"), 0)
        importance = 3 if memory_category == "identity" else 2 if memory_category in {"preference", "people"} else 1
        return self._run(
            lambda db: db.add_or_update_memory(memory_category, clean, importance, self._now()),
            None,
        )

    @staticmethod
    def _terms(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9][a-z0-9+#-]*", text.lower())
        return [word for word in dict.fromkeys(words) if len(word) >= 3 and word not in STOP_WORDS]

    def forget(self, query: str) -> int:
        """Delete memories containing every significant word of ``query``.

        Short or pronoun-only queries ("it", "that") delete the most recent
        memory instead of every memory that happens to contain those letters.
        """

        clean = re.sub(r"\s+", " ", query.strip()).strip(" .!?\n")
        if clean.lower() in _PRONOUN_QUERIES:
            latest = self._run(lambda db: db.latest_memory(), None)
            if latest is None:
                return 0
            return self._run(lambda db: db.delete_memory(latest.id), 0)
        terms = self._terms(clean)
        if not terms:
            return 0
        deleted = self._run(lambda db: db.delete_memories_matching_all(terms), 0)
        if deleted:
            # Keep a forgotten fact from resurfacing through conversation history.
            self._run(lambda db: db.delete_messages_matching_all(terms), 0)
        return deleted

    def list_memories(self, limit: int = 50) -> list[LongTermMemory]:
        return self._run(lambda db: db.all_memories(limit), [])

    def clear_memories(self) -> int:
        return self._run(lambda db: db.clear_memories(), 0)

    def clear_conversation(self) -> int:
        return self._run(lambda db: db.clear_conversation(self.session_id), 0)

    def handle_intent(self, intent: Intent) -> SkillResult:
        """Execute a validated memory intent through the normal skill router."""

        if not self.available:
            return SkillResult(False, "My memory store is unavailable right now, but everything else still works.")
        if intent.name == "remember_memory":
            content = intent.parameters.get("content", "")
            memory = self.remember(content if isinstance(content, str) else "")
            return SkillResult(bool(memory), "Got it. I'll remember that." if memory else "Tell me what you want me to remember.")
        if intent.name == "forget_memory":
            query = intent.parameters.get("query", "")
            if intent.parameters.get("latest"):
                query = ""
            count = self.forget(query if isinstance(query, str) else "")
            if not count:
                return SkillResult(True, "I couldn't find a matching memory.")
            return SkillResult(True, "Done, I've forgotten that." if count == 1 else f"Done, I've forgotten {count} related memories.")
        if intent.name == "list_memories":
            memories = self.list_memories()
            if not memories:
                return SkillResult(True, "I don't have anything saved about you yet.")
            return SkillResult(True, "Here's what I remember:\n" + "\n".join(f"- {memory.content}" for memory in memories))
        if intent.name == "clear_memories":
            count = self.clear_memories()
            return SkillResult(True, f"Cleared {count} saved memor{'y' if count == 1 else 'ies'}.")
        if intent.name == "clear_conversation":
            count = self.clear_conversation()
            return SkillResult(True, f"Cleared {count} messages from this conversation.")
        return SkillResult(False, "I don't have a safe memory operation for that request.")
