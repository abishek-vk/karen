"""Keyword retrieval and compact prompt-context formatting."""

from __future__ import annotations

import re

from .database import MemoryDatabase
from .models import ConversationMessage, LongTermMemory

_STOP_WORDS = frozenset(
    {"the", "and", "what", "which", "that", "this", "about", "with", "from", "does", "use", "you", "remember"}
)


def keywords(text: str) -> list[str]:
    """Extract useful, bounded search terms from a user request."""

    lowered = text.lower()
    words = [
        word
        for word in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+#.-]{2,}", lowered)
        if word not in _STOP_WORDS
    ]
    if re.search(r"\b(?:who am i|what(?:'s| is) my name|tell me my name)\b", lowered):
        words.insert(0, "name")
    return list(dict.fromkeys(words))[:8]


def retrieve_relevant(
    database: MemoryDatabase,
    query: str,
    session_id: str,
    memory_limit: int,
    message_limit: int,
) -> tuple[list[LongTermMemory], list[ConversationMessage]]:
    """Retrieve only relevant long-term memories plus bounded recent context."""

    terms = keywords(query)
    relevant_messages = database.search_messages(terms, message_limit)
    recent_messages = database.recent_messages(session_id, message_limit)
    messages_by_id = {message.id: message for message in relevant_messages}
    messages_by_id.update({message.id: message for message in recent_messages})
    bounded_messages = list(messages_by_id.values())[-message_limit:]
    return database.search_memories(terms, memory_limit), bounded_messages


def format_context(
    memories: list[LongTermMemory],
    messages: list[ConversationMessage],
) -> str:
    """Build a compact context block and omit it when nothing is relevant."""

    sections: list[str] = []
    if memories:
        sections.append(
            "Relevant long-term memories:\n"
            + "\n".join(f"- {memory.content}" for memory in memories)
        )
    if messages:
        recent = "\n".join(f"{message.role.title()}: {message.content}" for message in messages)
        sections.append(f"Recent conversation:\n{recent}")
    if not sections:
        return ""
    return (
        "Use the following local context only when relevant. Do not invent memories. Keep this context invisible in your response; do not mention memory or saved context unless the user asks about it.\n\n"
        + "\n\n".join(sections)
    )
