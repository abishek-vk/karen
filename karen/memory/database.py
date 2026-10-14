"""Small SQLite persistence layer for local Karen memory."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import ConversationMessage, LongTermMemory, Reminder

_SCHEMA_VERSION = 2
_MEMORY_COLUMNS = "id, category, content, importance, created_at, updated_at"
_MESSAGE_COLUMNS = "id, session_id, role, content, created_at"


def _like(value: str) -> str:
    """Escape LIKE wildcards so user text is always matched literally."""

    escaped = value.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _fts_terms(values: Iterable[str]) -> list[str]:
    terms = []
    for value in values:
        clean = "".join(character for character in value if character.isalnum())
        if clean:
            terms.append(f'"{clean}"')
    return terms


class MemoryDatabase:
    """Create and query the local memory database with short-lived connections."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fts_enabled = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
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
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_time
                    ON messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_memories_category
                    ON memories(category);
                CREATE INDEX IF NOT EXISTS idx_reminders_due
                    ON reminders(delivered, due_at);
                -- LIKE '%x%' cannot use these b-tree indexes; they only cost writes.
                DROP INDEX IF EXISTS idx_messages_content;
                DROP INDEX IF EXISTS idx_memories_content;
                """
            )
            try:
                connection.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        content, category,
                        content='memories', content_rowid='id',
                        tokenize='porter unicode61'
                    );
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, content, category)
                        VALUES (new.id, new.content, new.category);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content, category)
                        VALUES ('delete', old.id, old.content, old.category);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content, category)
                        VALUES ('delete', old.id, old.content, old.category);
                        INSERT INTO memories_fts(rowid, content, category)
                        VALUES (new.id, new.content, new.category);
                    END;
                    """
                )
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version < _SCHEMA_VERSION:
                    connection.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                self.fts_enabled = True
            except sqlite3.OperationalError:
                # Python builds without FTS5 fall back to escaped LIKE queries.
                self.fts_enabled = False

    # -- conversations ----------------------------------------------------

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
                f"""
                SELECT {_MESSAGE_COLUMNS}
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
        clauses = " OR ".join("lower(content) LIKE ? ESCAPE '\\'" for _ in values)
        arguments: list[Any] = [_like(value) for value in values]
        arguments.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {_MESSAGE_COLUMNS}
                FROM messages
                WHERE {clauses}
                ORDER BY id DESC
                LIMIT ?
                """,
                arguments,
            ).fetchall()
        return [ConversationMessage(**dict(row)) for row in reversed(rows)]

    def prune_messages(self, keep: int) -> int:
        """Keep only the newest ``keep`` messages so the database stays small."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM messages WHERE id <= (
                    SELECT id FROM messages ORDER BY id DESC LIMIT 1 OFFSET ?
                )
                """,
                (max(0, keep),),
            )
            return cursor.rowcount

    def delete_messages_matching_all(self, patterns: Iterable[str]) -> int:
        values = [pattern for pattern in patterns if pattern]
        if not values:
            return 0
        clauses = " AND ".join("lower(content) LIKE ? ESCAPE '\\'" for _ in values)
        arguments = [_like(value) for value in values]
        with self._connect() as connection:
            cursor = connection.execute(f"DELETE FROM messages WHERE {clauses}", arguments)
            return cursor.rowcount

    def clear_conversation(self, session_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            return cursor.rowcount

    # -- long-term memories ----------------------------------------------

    def search_memories(self, patterns: Iterable[str], limit: int) -> list[LongTermMemory]:
        values = [pattern for pattern in patterns if pattern]
        if not values or limit <= 0:
            return []
        with self._connect() as connection:
            terms = _fts_terms(values)
            if self.fts_enabled and terms:
                rows = connection.execute(
                    f"""
                    SELECT m.id, m.category, m.content, m.importance, m.created_at, m.updated_at
                    FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY bm25(memories_fts), m.importance DESC, m.updated_at DESC
                    LIMIT ?
                    """,
                    (" OR ".join(terms), limit),
                ).fetchall()
            else:
                clauses = " OR ".join(
                    "lower(content) LIKE ? ESCAPE '\\' OR lower(category) LIKE ? ESCAPE '\\'"
                    for _ in values
                )
                arguments: list[Any] = []
                for value in values:
                    arguments.extend((_like(value), _like(value)))
                arguments.append(limit)
                rows = connection.execute(
                    f"""
                    SELECT {_MEMORY_COLUMNS}
                    FROM memories
                    WHERE {clauses}
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    arguments,
                ).fetchall()
        return [LongTermMemory(**dict(row)) for row in rows]

    def memories_by_category(self, category: str, limit: int = 5) -> list[LongTermMemory]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {_MEMORY_COLUMNS} FROM memories
                WHERE category = ? ORDER BY updated_at DESC LIMIT ?
                """,
                (category, limit),
            ).fetchall()
        return [LongTermMemory(**dict(row)) for row in rows]

    def all_memories(self, limit: int) -> list[LongTermMemory]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {_MEMORY_COLUMNS}
                FROM memories
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [LongTermMemory(**dict(row)) for row in rows]

    def latest_memory(self) -> LongTermMemory | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories ORDER BY updated_at DESC, id DESC LIMIT 1"
            ).fetchone()
        return LongTermMemory(**dict(row)) if row else None

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
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return LongTermMemory(**dict(result))

    def delete_memory(self, memory_id: int) -> int:
        with self._connect() as connection:
            return connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount

    def delete_memories_by_category(self, category: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE category = ?", (category,))
            return cursor.rowcount

    def delete_memories_matching_all(self, patterns: Iterable[str]) -> int:
        """Delete memories that contain every term as a word (not as a substring)."""

        values = [pattern for pattern in patterns if pattern]
        if not values:
            return 0
        with self._connect() as connection:
            terms = _fts_terms(values)
            if self.fts_enabled and terms:
                ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?",
                        (" AND ".join(terms),),
                    ).fetchall()
                ]
                if not ids:
                    return 0
                placeholders = ",".join("?" for _ in ids)
                return connection.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})", ids
                ).rowcount
            clauses = " AND ".join("(' ' || lower(content) || ' ') LIKE ? ESCAPE '\\'" for _ in values)
            arguments = [_like(f" {value} ") for value in values]
            return connection.execute(f"DELETE FROM memories WHERE {clauses}", arguments).rowcount

    def delete_memories_matching(self, patterns: Iterable[str]) -> int:
        """Backward-compatible alias; requires all terms to match."""

        return self.delete_memories_matching_all(patterns)

    def clear_memories(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories")
            return cursor.rowcount

    # -- reminders --------------------------------------------------------

    def add_reminder(self, text: str, due_at: str, timestamp: str) -> Reminder:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO reminders(text, due_at, created_at) VALUES (?, ?, ?)",
                (text, due_at, timestamp),
            )
            row = connection.execute(
                "SELECT id, text, due_at, created_at, delivered FROM reminders WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return Reminder(**dict(row))

    def pending_reminders(self, limit: int = 50) -> list[Reminder]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, text, due_at, created_at, delivered FROM reminders
                WHERE delivered = 0 ORDER BY due_at LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [Reminder(**dict(row)) for row in rows]

    def mark_reminder_delivered(self, reminder_id: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE reminders SET delivered = 1 WHERE id = ?", (reminder_id,))

    def clear_reminders(self) -> int:
        with self._connect() as connection:
            return connection.execute("DELETE FROM reminders WHERE delivered = 0").rowcount
