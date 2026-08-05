import pytest

from karen.core.intent import Intent, parse_intent
from karen.core.router import IntentRouter
from karen.skills.app_control import SkillResult


def test_parse_intent_accepts_json_code_fence() -> None:
    intent = parse_intent(
        '```json\n{"intent":"open_application","target":"notepad",'
        '"parameters":{},"requires_confirmation":false}\n```'
    )

    assert intent == Intent("open_application", "notepad", {}, False)


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


def test_router_returns_general_chat_response() -> None:
    result = IntentRouter().route(
        Intent("general_chat", None, {"response": "Hello there."}, False)
    )

    assert result.success is True
    assert result.message == "Hello there."
