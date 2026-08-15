from pathlib import Path

from karen.core.application_service import ApplicationService
from karen.core.fast_path import parse_fast_path
from karen.core.intent import Intent
from karen.core.router import IntentRouter
from karen.memory.database import MemoryDatabase
from karen.memory.service import MemoryService
from karen.web.current_info import CurrentInfoService


def test_database_initializes_and_persists_schema(tmp_path: Path) -> None:
    path = tmp_path / "data" / "karen.db"
    database = MemoryDatabase(path)
    database.create_conversation("session", "2026-01-01T00:00:00+00:00")
    database.add_message("session", "user", "hello", "2026-01-01T00:00:01+00:00")
    assert path.exists()
    assert database.recent_messages("session", 10)[0].content == "hello"
    reopened = MemoryDatabase(path)
    assert reopened.recent_messages("session", 10)[0].session_id == "session"


def test_messages_are_stored_and_recent_limit_is_bounded(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "karen.db", recent_message_limit=2)
    service.record_message("user", "one")
    service.record_message("assistant", "two")
    service.record_message("user", "three")
    assert [message.content for message in service.database.recent_messages(service.session_id, 2)] == ["two", "three"]


def test_long_term_memory_create_update_retrieve_delete(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "karen.db", max_memory_results=5)
    first = service.remember("My favorite editor is VS Code.")
    updated = service.remember("My favorite editor is VS Code.", category="preference")
    assert first is not None and updated is not None
    assert updated.id == first.id
    assert service.context_for("What editor do I use?").find("VS Code") >= 0
    assert service.forget("favorite editor VS Code") == 1
    assert service.list_memories() == []


def test_clear_memories_and_conversation(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "karen.db")
    service.remember("This project is called Karen.")
    service.record_message("user", "hello")
    assert service.clear_conversation() == 1
    assert service.database.recent_messages(service.session_id, 5) == []
    assert service.clear_memories() == 1
    assert service.list_memories() == []


def test_explicit_memory_commands_use_fast_path_and_skill(tmp_path: Path) -> None:
    remember = parse_fast_path("Remember that my preferred language is Python.")
    assert remember == Intent(
        "remember_memory", None, {"content": "my preferred language is Python"}, False
    )
    service = MemoryService(tmp_path / "karen.db")
    router = IntentRouter()
    application = ApplicationService(object(), router, memory=service)
    response = application.handle_text("Remember that my preferred language is Python.")
    assert response.success is True
    assert "remember" in response.message.lower()
    assert application.handle_text("What do you remember?").success is True
    forget = application.handle_text("Forget that my preferred language is Python.")
    assert forget.success is True
    assert service.list_memories() == []


def test_empty_database_has_no_memory_context(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "karen.db")
    assert service.list_memories() == []
    assert service.context_for("What editor do I use?").startswith("Recent conversation") is False


def test_irrelevant_memory_is_not_injected(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "karen.db")
    service.remember("My favorite editor is VS Code.")
    assert "VS Code" not in service.context_for("What is the weather today?")


def test_memory_database_failure_is_non_fatal(tmp_path: Path) -> None:
    # A directory cannot be opened as a SQLite database file.
    service = MemoryService(tmp_path, enabled=True)
    assert service.available is False
    service.record_message("user", "still works")
    assert service.context_for("anything") == ""
    assert service.remember("a fact") is None


def test_name_is_promoted_to_long_term_memory(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "karen.db")
    service.record_message("user", "My name is Karen.")
    assert "Karen" in service.context_for("What is my name?")
    assert service.forget("my name") == 1
    assert service.list_memories() == []


def test_corrected_name_replaces_old_identity(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "karen.db")
    service.record_message("user", "My name is Karen.")
    service.record_message("user", "My name is Abi.")
    memories = service.list_memories()
    assert len(memories) == 1
    assert memories[0].category == "identity"
    assert memories[0].content == "My name is Abi"


def test_memory_survives_new_service_session(tmp_path: Path) -> None:
    path = tmp_path / "karen.db"
    first = MemoryService(path)
    first.record_message("user", "I am building an AI assistant called Karen.")
    second = MemoryService(path)
    assert "Karen" in second.context_for("What project am I building?")


def test_current_information_returns_live_search_results(monkeypatch) -> None:
    html = (
        '<a class="result__a" href="https://example.test/current">Current answer</a>'
        '<div class="result__snippet">The current office holder is Vijay.</div>'
    )
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return html.encode()

    monkeypatch.setattr("karen.web.current_info.urlopen", lambda request, timeout: Response())
    result = CurrentInfoService().handle_intent(
        Intent("current_information", None, {"query": "Who is the current CM of Tamil Nadu?"}, False)
    )
    assert result.success is True
    assert "Vijay" in result.message
    assert "example.test" in result.message


def test_current_question_fast_path_is_separate_from_model_memory() -> None:
    intent = parse_fast_path("Who is the current CM of Tamil Nadu?")
    assert intent == Intent(
        "current_information",
        None,
        {"query": "Who is the current CM of Tamil Nadu"},
        False,
    )


def test_clear_memory_command_requires_existing_confirmation_flow(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "karen.db")
    service.remember("A removable fact")
    application = ApplicationService(object(), IntentRouter(), memory=service)
    prompt = application.handle_text("Forget everything")
    assert prompt.success is False
    assert application.handle_text("yes").success is True
    assert service.list_memories() == []
