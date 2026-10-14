"""Regression tests for routing, memory safety, confirmations and streaming."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from karen.config import ConfigurationStore
from karen.core.application_service import ApplicationService
from karen.core.confirmation import parse_confirmation
from karen.core.fast_path import clean_utterance, parse_fast_path
from karen.core.intent import Intent
from karen.core.llm import LocalLLM, _ThinkFilter
from karen.core.router import IntentRouter
from karen.core.timeparse import parse_when
from karen.memory import MemoryService
from karen.skills import app_control
from karen.skills.app_control import CatalogEntry
from karen.voice.speaker import SpeechOutput, clean_for_speech
from karen.web.current_info import Research, Source, _resolve_redirect, relevant_passages


@pytest.fixture(autouse=True)
def fixed_catalog(monkeypatch, tmp_path):
    catalog = {"notepad": CatalogEntry("notepad", tmp_path / "Notepad.lnk", True)}
    monkeypatch.setattr(app_control, "_catalog", catalog)


@pytest.mark.parametrize(
    ("text", "intent", "target"),
    [
        ("Karen, open Notepad.", "open_application", "notepad"),
        ("Hey Karen open notepad please", "open_application", "notepad"),
        ("What time is it now?", "system_information", "time"),
        ("what's the date", "system_information", "date"),
        ("open youtube", "open_website", "youtube"),
        ("volume up", "system_control", "volume_up"),
        ("lock my computer", "system_control", "lock"),
        ("play despacito on youtube", "open_website", "youtube"),
    ],
)
def test_fast_path_handles_spoken_commands(text: str, intent: str, target: str) -> None:
    parsed = parse_fast_path(text)
    assert parsed is not None and parsed.name == intent and parsed.target == target


@pytest.mark.parametrize(
    "text",
    ["how are you today", "do you know who I am", "what do you know about python",
     "where do I live", "what should I do today", "who are you right now"],
)
def test_personal_questions_are_not_sent_to_the_web(text: str) -> None:
    parsed = parse_fast_path(text)
    assert parsed is None or parsed.name != "current_information"


@pytest.mark.parametrize(
    "text", ["what's the latest news", "weather in Chennai today", "search the web for python 3.14 release"]
)
def test_live_questions_go_to_the_web(text: str) -> None:
    parsed = parse_fast_path(text)
    assert parsed is not None and parsed.name == "current_information"


def test_clean_utterance_strips_wake_word() -> None:
    assert clean_utterance("Hey Jarvis, what's up?") == "what's up"


def test_forget_it_only_removes_the_latest_memory(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "k.db")
    memory.remember("My favorite editor is vim")
    memory.remember("I write python with git")
    memory.remember("My sister lives in Chennai")
    assert memory.forget("it") == 1
    remaining = {item.content for item in memory.list_memories()}
    assert remaining == {"My favorite editor is vim", "I write python with git"}


def test_forget_requires_whole_word_matches(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "k.db")
    memory.remember("My favorite editor is vim")
    assert memory.forget("edit") == 0
    assert memory.forget("percent % sign") == 0
    assert len(memory.list_memories()) == 1


def test_names_are_extracted_without_trailing_clauses(tmp_path: Path) -> None:
    memory = MemoryService(tmp_path / "k.db")
    memory.record_message("user", "my name is abishek and I study at Karunya")
    assert memory.user_name() == "Abishek"


@pytest.mark.parametrize("text", ["Yes.", "yes please", "yeah", "sure", "ok", "Yes, send it."])
def test_spoken_confirmations(text: str) -> None:
    assert parse_confirmation(text) is True


@pytest.mark.parametrize("text", ["no", "nope", "no thanks", "never mind", "don't send it"])
def test_spoken_rejections(text: str) -> None:
    assert parse_confirmation(text) is False


def test_new_request_replaces_pending_confirmation(tmp_path: Path) -> None:
    service = ApplicationService(object(), IntentRouter(), memory=MemoryService(tmp_path / "k.db"))
    service.confirmations.request(Intent("clear_memories", None, {}, True))
    response = service.handle_text("what time is it")
    assert "cancelled the pending action" in response.message
    assert service.confirmations.peek() is None


def test_reminders_parse_and_store(tmp_path: Path) -> None:
    service = ApplicationService(object(), IntentRouter(), memory=MemoryService(tmp_path / "k.db"))
    response = service.handle_text("remind me to call mom in 10 minutes")
    assert response.success is True and "call mom" in response.message
    assert "call mom" in service.handle_text("what are my reminders").message


def test_parse_when_handles_relative_and_clock_times() -> None:
    now = datetime(2026, 10, 14, 15, 0).astimezone()
    due, rest = parse_when("call mom in 10 minutes", now)
    assert due == now + timedelta(minutes=10) and rest == "call mom"
    due, _ = parse_when("at 5 pm", now)
    assert due is not None and due.hour == 17 and due.date() == now.date()


def test_settings_ignore_unknown_keys(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"model": "x", "voice": true, "tts_rate": "3"}')
    settings = ConfigurationStore(tmp_path).load_settings()
    assert settings.ollama_model == "x" and settings.tts_rate == 3


def test_think_filter_removes_reasoning() -> None:
    think = _ThinkFilter()
    output = "".join(think.feed(part) for part in ["<thi", "nk>hidden</th", "ink>Hello", " there"])
    assert output == "Hello there"


class _FakeClient:
    model = "fake"

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    def stream_chat(self, messages, **_kwargs):
        yield from self.chunks


def test_llm_streams_answers_and_detects_search() -> None:
    received: list[str] = []
    answer = LocalLLM(_FakeClient(["Tokyo", " is the capital."])).respond("capital of japan", on_chunk=received.append)
    assert answer.kind == "answer" and answer.text == "Tokyo is the capital."
    assert "".join(received) == answer.text
    search = LocalLLM(_FakeClient(["SEA", "RCH: ipl", " winner 2026\n"])).respond("who won")
    assert search.kind == "search" and search.text == "ipl winner 2026"
    action = LocalLLM(_FakeClient(['{"intent": "list_reminders"', ', "target": null, "parameters": {}}'])).respond("x")
    assert action.kind == "action" and "list_reminders" in action.text


def test_speech_is_cleaned_and_split_into_sentences() -> None:
    assert clean_for_speech("It's **31°C**.\nSources: bbc.com") == "It's 31 degrees Celsius."

    class Recorder:
        def __init__(self) -> None:
            self.spoken: list[str] = []

        def speak(self, text: str) -> None:
            self.spoken.append(text)

    engine = Recorder()
    output = SpeechOutput(engine)
    output.feed("Good evening, Dr. Banner. The weather is ")
    output.feed("clear tonight. Shall I continue?")
    output.flush()
    assert output.wait(2)
    assert engine.spoken == ["Good evening, Dr. Banner.", "The weather is clear tonight.", "Shall I continue?"]


def test_web_helpers() -> None:
    assert _resolve_redirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=1") == "https://example.com/a"
    text = (
        "Unrelated paragraph about cooking pasta at home with friends.\n"
        "The 2026 IPL final was won by the Chennai team in May."
    )
    assert "IPL final" in relevant_passages(text, ["ipl", "final", "won"], 500)
    assert Source("t", "https://www.bbc.com/news", "x").site == "bbc.com"
    assert Research("q").sources == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("who is the cm of TN?", ("Chief Minister", "Tamil Nadu")),
        ("Who is the prime minister of India", ("Prime Minister", "India")),
        ("who is the president of the US", ("President", "the United States")),
        ("who is kerala's chief minister", ("Chief Minister", "Kerala")),
        ("what is the capital of India", None),
    ],
)
def test_office_questions_are_recognized(text: str, expected) -> None:
    from karen.core.offices import parse_office_question

    assert parse_office_question(text) == expected


@pytest.mark.parametrize("text", ["who is the cm of TN?", "who won the world cup", "who is the ceo of google"])
def test_changing_facts_always_use_the_web(text: str) -> None:
    parsed = parse_fast_path(text)
    assert parsed is not None and parsed.name == "current_information"


class _FakeWeb:
    enabled = True

    def __init__(self) -> None:
        self.queries: list[str] = []

    def research(self, query: str) -> Research:
        self.queries.append(query)
        return Research(query, [Source("GPT-6", "https://example.com/gpt6", "GPT-6 Astra was released in 2026.")])

    def summarize(self, research: Research) -> str:
        return research.sources[0].excerpt

    def handle_intent(self, intent):  # pragma: no cover - not used here
        raise AssertionError


class _ScriptedClient:
    """Returns one scripted stream per call (first: routing reply, then: sourced answer)."""

    model = "fake"

    def __init__(self, *replies: list[str]) -> None:
        self.replies = list(replies)

    def stream_chat(self, messages, **_kwargs):
        yield from self.replies.pop(0)


@pytest.mark.parametrize(
    "first_reply",
    [
        ["SEARCH: gpt 6 astra"],
        ["I'm sorry, but there's no information available about \"GPT 6 Astra\" as of my last update.", " Could you"],
        ["I'm not aware of a model called GPT 6 Astra."],
    ],
)
def test_model_search_requests_and_unknowns_use_the_web(tmp_path: Path, first_reply: list[str]) -> None:
    web = _FakeWeb()
    llm = LocalLLM(_ScriptedClient(first_reply, ["GPT-6 Astra was released in 2026."]))
    service = ApplicationService(llm, IntentRouter(), memory=MemoryService(tmp_path / "k.db"), live_information=web)
    spoken: list[str] = []
    response = service.handle_text("explain the astra model to me", on_chunk=spoken.append)
    assert response.source == "web" and "released in 2026" in response.message
    assert web.queries and "no information" not in "".join(spoken)
