"""Small SQLite persistence layer for local Karen memory."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import ConversationMessage, LongTermMemory


class MemoryDatabase:
    """Create and query the local memory database with short-lived connections."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_time
                    ON messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_content
                    ON messages(content);
                CREATE INDEX IF NOT EXISTS idx_memories_category
                    ON memories(category);
                CREATE INDEX IF NOT EXISTS idx_memories_content
                    ON memories(content);
                """
            )

    def create_conversation(self, session_id: str, timestamp: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations(id, started_at) VALUES (?, ?)",
                (session_id, timestamp),
            )

    def add_message(self, session_id: str, role: str, content: str, timestamp: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, timestamp),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a message id")
            return int(cursor.lastrowid)

    def recent_messages(self, session_id: str, limit: int) -> list[ConversationMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [ConversationMessage(**dict(row)) for row in reversed(rows)]

    def search_messages(self, patterns: Iterable[str], limit: int) -> list[ConversationMessage]:
        values = [pattern for pattern in patterns if pattern]
        if not values:
            return []
        clauses = " OR ".join("lower(content) LIKE ?" for _ in values)
        arguments: list[Any] = [f"%{value.lower()}%" for value in values]
        arguments.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, session_id, role, content, created_at
                FROM messages
                WHERE {clauses}
                ORDER BY id DESC
                LIMIT ?
                """,
                arguments,
            ).fetchall()
        return [ConversationMessage(**dict(row)) for row in reversed(rows)]

    def search_memories(self, patterns: Iterable[str], limit: int) -> list[LongTermMemory]:
        values = [pattern for pattern in patterns if pattern]
        if not values:
            return []
        clauses = " OR ".join("lower(content) LIKE ? OR lower(category) LIKE ?" for _ in values)
        arguments: list[Any] = []
        for value in values:
            wildcard = f"%{value.lower()}%"
            arguments.extend((wildcard, wildcard))
        arguments.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, category, content, importance, created_at, updated_at
                FROM memories
                WHERE {clauses}
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                arguments,
            ).fetchall()
        return [LongTermMemory(**dict(row)) for row in rows]

    def all_memories(self, limit: int) -> list[LongTermMemory]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, category, content, importance, created_at, updated_at
                FROM memories
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [LongTermMemory(**dict(row)) for row in rows]

    def add_or_update_memory(
        self,
        category: str,
        content: str,
        importance: int,
        timestamp: str,
    ) -> LongTermMemory:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM memories WHERE lower(content) = lower(?) LIMIT 1",
                (content,),
            ).fetchone()
            if row:
                connection.execute(
                    """
                    UPDATE memories
                    SET category = ?, importance = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (category, importance, timestamp, row["id"]),
                )
                memory_id = row["id"]
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO memories(category, content, importance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (category, content, importance, timestamp, timestamp),
                )
                memory_id = cursor.lastrowid
            result = connection.execute(
                """
                SELECT id, category, content, importance, created_at, updated_at
                FROM memories WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()
        return LongTermMemory(**dict(result))

    def delete_memories_matching(self, patterns: Iterable[str]) -> int:
        values = [pattern for pattern in patterns if pattern]
        if not values:
            return 0
        clauses = " OR ".join("lower(content) LIKE ?" for _ in values)
        arguments = [f"%{value.lower()}%" for value in values]
        with self._connect() as connection:
            cursor = connection.execute(f"DELETE FROM memories WHERE {clauses}", arguments)
            return cursor.rowcount

    def clear_memories(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories")
            return cursor.rowcount

    def clear_conversation(self, session_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            return cursor.rowcount
