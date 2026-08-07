from typing import cast

import pytest

from karen.core.confirmation import ConfirmationManager
from karen.core.fast_path import parse_fast_path
from karen.core.intent import Intent, parse_intent
from karen.core.router import IntentRouter
from karen.core.skill_registry import RegisteredSkill, SkillRegistry
from karen.skills.base import SkillManifest, SkillResult
from karen.text_app import process_text
from karen.core.llm import LocalLLM


def test_parse_intent_accepts_json_code_fence() -> None:
    intent = parse_intent(
        '```json\n{"intent":"open_application","target":"notepad",'
        '"parameters":{},"requires_confirmation":false}\n```'
    )

    assert intent == Intent("open_application", "notepad", {}, False)


def test_parse_intent_handles_qwen_thinking_and_repeated_json() -> None:
    raw = '<think>Reasoning text with an example.</think> '
    raw += '{"intent":"general_chat","target":null,"parameters":{"response":"Hello!"},"requires_confirmation":false}'
    raw += '\n{"intent":"general_chat","target":null,"parameters":{"response":"Duplicate"},"requires_confirmation":false}'

    assert parse_intent(raw).parameters["response"] == "Hello!"


def test_parse_intent_rejects_unsupported_action() -> None:
    with pytest.raises(ValueError, match="Unsupported intent"):
        parse_intent('{"intent":"run_shell","target":"whoami"}')


def test_router_blocks_confirmation_before_skill(monkeypatch) -> None:
    called = False

    def fail_if_called(target):
        nonlocal called
        called = True
        return SkillResult(True, "unsafe")

    monkeypatch.setattr("karen.skills.app_control.open_application", fail_if_called)
    result = IntentRouter().route(Intent("open_application", "notepad", {}, True))

    assert result.success is False
    assert called is False
    assert "confirmation" in result.message


def test_fast_path_handles_common_application_command() -> None:
    assert parse_fast_path("Please open Notepad!") == Intent(
        "open_application", "notepad", {}, False
    )
    assert parse_fast_path("run whoami") is None


def test_confirmation_state_approves_exact_pending_intent(monkeypatch) -> None:
    executed = []

    def execute(target):
        executed.append(target)
        return SkillResult(True, "Opening Notepad.")

    monkeypatch.setattr("karen.skills.app_control.open_application", execute)
    confirmations = ConfirmationManager()
    router = IntentRouter()
    intent = Intent("open_application", "notepad", {}, True)
    confirmations.request(intent)

    assert process_text("yes", cast(LocalLLM, object()), router, confirmations) == "Opening Notepad."
    assert executed == ["notepad"]
    assert confirmations.peek() is None


def test_confirmation_state_can_cancel() -> None:
    confirmations = ConfirmationManager()
    confirmations.request(Intent("open_application", "notepad", {}, True))

    assert process_text("no", cast(LocalLLM, object()), IntentRouter(), confirmations) == "Cancelled."
    assert confirmations.peek() is None


def test_registry_rejects_duplicate_intents() -> None:
    registry = SkillRegistry()
    skill = RegisteredSkill(
        SkillManifest("demo", frozenset({"demo"}), frozenset({"demo"})),
        lambda intent: SkillResult(True, "ok"),
    )
    registry.register(skill)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(skill)


def test_permission_policy_requires_confirmation_for_closing() -> None:
    router = IntentRouter()

    assert router.needs_confirmation(Intent("open_application", "notepad", {}, False)) is False
    assert router.needs_confirmation(Intent("close_application", "notepad", {}, False)) is True
    assert router.route(Intent("close_application", "notepad", {}, False)).success is False


def test_registry_rejects_unknown_intent() -> None:
    result = IntentRouter().route(Intent("unknown", None, {}, False))

    assert result.success is False
    assert "safe skill" in result.message


def test_router_returns_general_chat_response() -> None:
    result = IntentRouter().route(
        Intent("general_chat", None, {"response": "Hello there."}, False)
    )

    assert result.success is True
    assert result.message == "Hello there."
