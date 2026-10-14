"""Keyword retrieval and compact prompt-context formatting."""

from __future__ import annotations

import re

from .database import MemoryDatabase
from .models import LongTermMemory

STOP_WORDS = frozenset(
    {
        "the", "and", "but", "for", "nor", "yet", "what", "which", "who", "whom", "whose", "when",
        "where", "why", "how", "that", "this", "these", "those", "there", "here", "about", "with",
        "from", "into", "onto", "over", "under", "does", "did", "done", "doing", "use", "used",
        "you", "your", "yours", "yourself", "remember", "karen", "jarvis", "can", "could", "would",
        "should", "will", "shall", "may", "might", "must", "please", "tell", "say", "said", "know",
        "think", "want", "like", "just", "also", "very", "really", "much", "many", "some", "any",
        "all", "are", "was", "were", "been", "being", "have", "has", "had", "not", "now", "then",
        "than", "them", "they", "their", "she", "her", "him", "his", "its", "our", "ours", "out",
        "get", "got", "give", "make", "let", "lets", "okay", "hey", "hello", "thanks", "thank",
        "again", "something", "anything", "thing", "things", "one", "today", "mine", "myself",
    }
)


def keywords(text: str) -> list[str]:
    """Extract useful, bounded search terms from a user request."""

    lowered = text.lower()
    words = [
        word.strip(".-")
        for word in re.findall(r"[a-z0-9][a-z0-9+#.-]{2,}", lowered)
        if word.strip(".-") not in STOP_WORDS
    ]
    if re.search(r"\b(?:who am i|what(?:'s| is) my name|tell me my name|my name)\b", lowered):
        words.insert(0, "name")
    return list(dict.fromkeys(word for word in words if len(word) >= 3))[:8]


_QUESTION = re.compile(
    r"^(?:who|what|whats|what's|which|where|when|why|how|is|are|was|were|do|does|did|can|could|will|would|should|"
    r"tell me|show me|his|her|their|my)\b|\?\s*$",
    re.IGNORECASE,
)


def earlier_statements(database: MemoryDatabase, query: str, limit: int = 3) -> list[str]:
    """Things the user *stated* in past conversations that relate to ``query``.

    Facts phrased in ways the automatic extractor doesn't recognise still live in
    the conversation log; this finds them by keyword so they aren't lost after a
    few turns. Questions and the current message itself are skipped.
    """

    terms = keywords(query)
    if not terms:
        return []
    current = " ".join(query.lower().split())
    scored: list[tuple[int, int, str]] = []
    for message in database.search_messages(terms, 60):
        text = " ".join(message.content.split())
        lowered = text.lower()
        if message.role != "user" or lowered == current or _QUESTION.search(text) or len(text) > 300:
            continue
        hits = sum(1 for term in terms if term in lowered)
        scored.append((hits, message.id, text))
    chosen: list[str] = []
    for _hits, _id, text in sorted(scored, key=lambda item: (-item[0], -item[1])):
        if text.lower() not in (item.lower() for item in chosen):
            chosen.append(text)
        if len(chosen) >= limit:
            break
    return chosen


def retrieve_memories(database: MemoryDatabase, query: str, limit: int) -> list[LongTermMemory]:
    """Return the user's identity plus only the memories relevant to ``query``."""

    identity = database.memories_by_category("identity", 1)
    relevant = database.search_memories(keywords(query), limit)
    seen: set[int] = set()
    combined: list[LongTermMemory] = []
    for memory in [*identity, *relevant]:
        if memory.id not in seen:
            seen.add(memory.id)
            combined.append(memory)
    return combined[: max(limit, len(identity))]


def format_context(memories: list[LongTermMemory], statements: list[str] | None = None) -> str:
    """Build a compact facts block, or nothing when no memory is relevant."""

    parts = []
    if memories:
        parts.append(
            "Facts the user has told you (first person = the user). Use them only when relevant "
            "and never invent others:\n" + "\n".join(f"- {memory.content}" for memory in memories)
        )
    if statements:
        parts.append(
            "Things the user said in earlier messages, in their own words (they may refer to themselves by name). "
            "Use them to answer questions about the user:\n" + "\n".join(f'- "{text}"' for text in statements)
        )
    return "\n".join(parts)
