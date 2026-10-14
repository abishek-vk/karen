"""Typed records returned by Karen's local memory store."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class LongTermMemory:
    id: int
    category: str
    content: str
    importance: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class Reminder:
    id: int
    text: str
    due_at: str
    created_at: str
    delivered: int
