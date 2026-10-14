"""WhatsApp commands must reach exactly the contact the user named, with their exact words."""

from __future__ import annotations

import pytest

from karen.core.fast_path import parse_fast_path
from karen.skills.whatsapp import WhatsAppAutomationError, _resolve_contact


@pytest.mark.parametrize(
    ("text", "contact", "message"),
    [
        ("Send a WhatsApp message to Shanju saying I will be late today.", "Shanju", "I will be late today"),
        ("send a message to shanju saying hi da", "Shanju", "hi da"),
        ("whatsapp Shanju that the meeting is at 5", "Shanju", "the meeting is at 5"),
        ("tell Shanju on whatsapp that I am coming", "Shanju", "I am coming"),
        ('send "good morning" to Shanju on whatsapp', "Shanju", "good morning"),
        ("text Rahul Kumar: call me", "Rahul Kumar", "call me"),
    ],
)
def test_message_commands_name_the_contact_and_keep_exact_words(text: str, contact: str, message: str) -> None:
    intent = parse_fast_path(text)
    assert intent is not None and intent.name == "send_whatsapp_message"
    assert intent.parameters == {"contact": contact, "message": message}


@pytest.mark.parametrize("text", ["send me a message", "tell me a joke", "what is the message", "open whatsapp"])
def test_other_requests_are_not_messages(text: str) -> None:
    intent = parse_fast_path(text)
    assert intent is None or intent.name != "send_whatsapp_message"


class _Titles:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def count(self) -> int:
        return len(self.names)

    def nth(self, index: int) -> "_Title":
        return _Title(self.names[index])


class _Title:
    def __init__(self, name: str) -> None:
        self.name = name

    def is_visible(self) -> bool:
        return True

    def get_attribute(self, _name: str) -> str:
        return self.name


class _Page:
    def __init__(self, *names: str) -> None:
        self.names = list(names)

    def locator(self, _selector: str) -> _Titles:
        return _Titles(self.names)


def test_spoken_first_name_finds_the_single_matching_chat() -> None:
    assert _resolve_contact(_Page("Shanju K", "Sharan"), "shanju") == "Shanju K"
    assert _resolve_contact(_Page("Shanju", "Shanju K"), "Shanju") == "Shanju"  # exact name wins


def test_ambiguous_or_unknown_contacts_are_never_guessed() -> None:
    with pytest.raises(WhatsAppAutomationError, match="More than one chat"):
        _resolve_contact(_Page("Shanju K", "Shanju R"), "shanju")
    with pytest.raises(WhatsAppAutomationError, match="couldn't find"):
        _resolve_contact(_Page("Sharan"), "shanju")


@pytest.mark.parametrize(
    ("text", "query", "play"),
    [
        ("play shape of you on youtube", "shape of you", True),
        ("play tum hi ho", "tum hi ho", True),
        ("search youtube for python tutorials", "python tutorials", None),
    ],
)
def test_youtube_play_opens_the_video_and_search_shows_results(text: str, query: str, play: bool | None) -> None:
    intent = parse_fast_path(text)
    assert intent is not None and intent.name == "open_website" and intent.target == "youtube"
    assert intent.parameters.get("query") == query
    assert intent.parameters.get("play") is play


@pytest.mark.parametrize("text", ["play", "play music", "pause"])
def test_bare_play_still_controls_media(text: str) -> None:
    intent = parse_fast_path(text)
    assert intent is not None and intent.name == "system_control"


def test_focus_prefers_the_new_window_named_after_what_was_opened(monkeypatch) -> None:
    from karen.skills import focus

    windows = [(1, "karen - Visual Studio Code"), (2, "Inbox - Brave"), (3, "Calculator"), (4, "Untitled - Notepad")]
    monkeypatch.setattr(focus, "_windows", lambda: windows)
    before = {1, 2}
    assert focus.find_window(before, ["notepad"]) == 4  # new and named
    assert focus.find_window({1, 2, 3, 4}, ["brave"]) == 2  # existing browser whose title changed
    assert focus.find_window(before, ["spotify"]) == 3  # otherwise the first new window
