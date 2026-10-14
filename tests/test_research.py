"""Web research must stay on topic, prefer trusted sources and surface the newest facts."""

from __future__ import annotations

import time

from karen.core.application_service import _python_syntax_note
from karen.core.fast_path import parse_fast_path

from karen.core.llm import LocalLLM, _classify_prefix, _tidy
from karen.voice.speaker import SpeechOutput
from karen.web.current_info import (
    CurrentInfoService,
    SearchResult,
    _on_topic,
    _query_terms,
    _rank_by_trust,
    _with_current_year,
    parse_version_question,
    relevant_passages,
)


def test_off_topic_search_results_are_dropped() -> None:
    results = [
        SearchResult("Latest news headlines", "Breaking news from India and the world.", "https://news.example/latest"),
        SearchResult("Download Python", "Python 3.15.0 is the latest stable version.", "https://www.python.org/downloads/"),
    ]
    kept = _on_topic(results, _query_terms("latest stable version of python"))
    assert [result.url for result in kept] == ["https://www.python.org/downloads/"]


def test_trusted_sources_rank_first_and_content_farms_last() -> None:
    results = [
        SearchResult("Q", "", "https://www.quora.com/What-is-the-latest-Python"),
        SearchResult("Blog", "", "https://randomblog.example/python-version"),
        SearchResult("Official", "", "https://www.python.org/downloads/"),
    ]
    assert [result.url for result in _rank_by_trust(results)] == [
        "https://www.python.org/downloads/",
        "https://randomblog.example/python-version",
        "https://www.quora.com/What-is-the-latest-Python",
    ]


def test_recency_questions_search_for_this_year() -> None:
    year = time.localtime().tm_year
    assert _with_current_year("who won the last FIFA World Cup") == f"who won the last FIFA World Cup {year}"
    assert _with_current_year("who won the 2022 FIFA World Cup") == "who won the 2022 FIFA World Cup"
    assert _with_current_year("what is photosynthesis") == "what is photosynthesis"


def test_long_paragraphs_still_yield_their_newest_sentence() -> None:
    year = time.localtime().tm_year
    filler = "The tournament has a long history with many memorable matches and famous players. " * 6
    text = "\n".join(
        [
            "The FIFA World Cup is contested by men's national teams; the World Cup final is the last match.",
            f"{filler}Spain defeated Argentina in the latest final, staged at MetLife Stadium in {year}.",
        ]
    )
    excerpt = relevant_passages(text, _query_terms("who won the last FIFA World Cup"), 450)
    assert "Spain defeated Argentina" in excerpt


def test_speech_skips_code_blocks_and_keeps_word_spacing() -> None:
    spoken: list[str] = []

    class Engine:
        def speak(self, text: str) -> None:
            spoken.append(text)

    speech = SpeechOutput(Engine())
    text = "Sure! Here is a function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\n\nCall reverse_string with any text."
    for index in range(0, len(text), 7):
        speech.feed(text[index : index + 7])
    speech.flush()
    speech.wait(5)
    assert spoken == [
        "Sure! Here is a function to reverse a string:",
        "I've put the code on screen.",
        "Call reverse string with any text.",
    ]


def test_model_never_repeats_the_sources_footer() -> None:
    assert _tidy("Royal Challengers Bengaluru won.\nSources: en.wikipedia.org") == "Royal Challengers Bengaluru won."
    history = LocalLLM._history([("assistant", "Spain won.\nSources: fifa.com, bbc.com"), ("user", "who scored?")])
    assert [message.content for message in history] == ["Spain won.", "who scored?"]


def test_software_version_questions_are_recognised() -> None:
    assert parse_version_question("What is the latest version of Python?") == "python"
    assert parse_version_question("latest stable version of node.js") == "node.js"
    assert parse_version_question("latest React version 2026") == "react"
    assert parse_version_question("what's the latest iphone") is None
    assert parse_version_question("who is the current cm of kerala") is None


def test_latest_version_comes_from_release_data(monkeypatch) -> None:
    cycles = '[{"cycle": "26", "lts": "2026-10-28", "latest": "26.11.1", "latestReleaseDate": "2026-10-08"},' \
             ' {"cycle": "24", "lts": "2025-10-28", "latest": "24.21.0", "latestReleaseDate": "2026-10-01"}]'
    monkeypatch.setattr(CurrentInfoService, "_get_text", lambda self, url, **_: cycles if "endoflife.date" in url else "")
    monkeypatch.setattr(time, "strftime", lambda _format: "2026-10-14")
    [result] = CurrentInfoService().search("what is the latest version of node.js")
    assert "26.11.1" in result.snippet and "Node.js 24, currently 24.21.0" in result.snippet
    assert result.url == "https://endoflife.date/nodejs"


def test_code_answers_are_not_mistaken_for_actions() -> None:
    assert _classify_prefix("```python\nfor i in range(3):") == "answer"
    assert _classify_prefix("```\nprint('hi')") == "answer"
    assert _classify_prefix('```json\n{"intent": "open_application"') == "action"
    assert _classify_prefix('{"intent": "open_application"') == "action"
    assert _classify_prefix("```") is None  # wait for the fence header



def test_answers_name_sites_not_the_sources_provided() -> None:
    assert _tidy("According to the sources, Gill is captain. This information is based on the sources provided.") == "Gill is captain."
    assert _tidy("The repo rate is 5.50%, based on the information available.") == "The repo rate is 5.50%."
    assert _tidy("According to BBC, Spain won.") == "According to BBC, Spain won."


def test_python_fix_requests_get_the_real_syntax_error() -> None:
    assert "'(' was never closed" in _python_syntax_note('fix this python code: print("hello"')
    assert _python_syntax_note('fix this python code: print("hello")') == ""
    assert _python_syntax_note("what is python") == ""


def test_follow_up_questions_go_to_the_model_with_history() -> None:
    assert parse_fast_path("who was the captain of that team") is None
    assert parse_fast_path("who is the ceo of openai") is not None
