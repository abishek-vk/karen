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


def format_context(memories: list[LongTermMemory]) -> str:
    """Build a compact facts block, or nothing when no memory is relevant."""

    if not memories:
        return ""
    return (
        "Facts the user has told you (first person = the user). Use them only when relevant "
        "and never invent others:\n" + "\n".join(f"- {memory.content}" for memory in memories)
    )
