"""Deterministic handling for common low-risk commands.

Everything matched here runs instantly without the local model, which keeps the
CPU idle for the most frequent voice commands. Patterns stay narrow: a false
negative just falls through to the model, a false positive does the wrong thing.
"""

from __future__ import annotations

import re

from karen.skills.app_control import available_applications, resolve_application_name
from karen.skills.web_open import resolve_website

from .intent import Intent
from .offices import parse_office_question
from .timeparse import parse_when

_WAKE_PREFIX = re.compile(
    r"^(?:(?:hey|hi|hello|ok|okay|yo)[\s,]+)?(?:karen|jarvis)\b[\s,.:;!-]*", re.IGNORECASE
)
_POLITE_PREFIX = re.compile(
    r"^(?:please|kindly|(?:can|could|would|will) you(?: please)?|i want you to|i need you to)\s+",
    re.IGNORECASE,
)
_POLITE_SUFFIX = re.compile(r"[\s,]+(?:please|for me|right now|now|karen|jarvis|thanks|thank you)$")
_PRONOUNS = frozenset({"it", "that", "this", "the last one", "that one", "the last thing"})

_TIME_QUESTIONS = re.compile(
    r"(?:what(?:'s| is)? the (?:current )?time(?: is it)?|what time is it|tell me the time|"
    r"current time|time now|what(?:'s| is) the time now)"
)
_DATE_QUESTIONS = re.compile(
    r"(?:what(?:'s| is)? (?:the |today'?s )?date(?: today)?|today'?s date|what day is (?:it|today)|"
    r"which day is (?:it|today)|what is today|tell me the date|current date)"
)
_SYSTEM_CONTROL = (
    (re.compile(r"(?:turn |volume |increase |raise )?(?:the )?volume up|(?:increase|raise|turn up) (?:the )?volume|louder"), "volume_up"),
    (re.compile(r"(?:turn |volume |decrease |lower )?(?:the )?volume down|(?:decrease|lower|reduce|turn down) (?:the )?volume|quieter"), "volume_down"),
    (re.compile(r"(?:mute|unmute)(?: the)?(?: volume| sound| audio| computer| laptop)?"), "mute"),
    (re.compile(r"(?:pause|resume|play)(?: the)?(?: music| song| media| video| playback)?"), "play_pause"),
    (re.compile(r"(?:next|skip)(?: the)? (?:song|track)|skip(?: this)? song|play next(?: song)?"), "next_track"),
    (re.compile(r"(?:previous|last|go back a) (?:song|track)|play previous(?: song)?"), "previous_track"),
    (re.compile(r"lock (?:my |the )?(?:computer|pc|laptop|screen|system|workstation)"), "lock"),
)
_STRONG_LIVE = re.compile(
    r"\b(?:current|currently|latest|breaking|news|headlines?|weather|forecast|temperature|"
    r"prices?|cost of|stocks?|share price|exchange rate|scores?|live score|trending|"
    r"release date|election results?|incumbent)\b"
)
_WEAK_LIVE = re.compile(
    r"\b(?:today|today's|tonight|now|right now|recent|recently|this (?:week|month|year)|"
    r"yesterday|upcoming|this season)\b"
)
_QUESTION_WORDS = re.compile(
    r"\b(?:who|what|what's|whats|which|where|when|how|is|are|was|did|does|will|tell me|give me|show me)\b"
)
_ABOUT_ASSISTANT = re.compile(r"\b(?:you|your|yourself|you're)\b")
_CHANGING_FACTS = re.compile(
    r"\b(?:who won|who is winning|who leads|who is leading|winner of|champion of|who owns|"
    r"ceo of|captain of|coach of|chairman of|richest|population of|price of|exchange rate|"
    r"how much is .+ (?:worth|cost)|is .+ (?:dead|alive|still alive)|net worth)\b"
)
_VERSIONED_THING = re.compile(
    r"^(?:tell me (?:about|more about)|(?:give|show)(?: me)? (?:some |more )?(?:info|information|details|specs)(?: about| on| of)?|"
    r"(?:some |more )?(?:info|information|details|specs|news|features)(?: about| on| of)|what(?:'s| is| are)(?: the)?|"
    r"is there(?: an?| the)?|when (?:is|was|will)(?: the)?|does(?: the)?|how good is(?: the)?)\s+"
    r"(?:the |a |an )?[a-z][a-z.+-]*(?: [a-z][a-z.+-]*)?[ -]?v?\d+(?:\.\d+)?[a-z]?\b"
)
_ABOUT_USER = re.compile(r"\b(?:i|i'm|im|my|me|mine|we|our|us)\b")
_TOPIC_FIRST = re.compile(
    r"^(?:the )?(?:weather|forecast|temperature|latest|news|headlines|breaking news|"
    r"live score|[a-z]+ (?:price|stock price|share price|score))\b"
)
_EXPLICIT_SEARCH = re.compile(
    r"(?:search|google|look up|lookup|find out|research)"
    r"(?: on)?(?: the web| online| the internet| internet| google)?(?: for| about| on)? (.+)"
)
_FOLLOW_UP = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|its|it)\b|"
    r"\b(?:that|this|those|these|the same) (?:team|person|man|woman|player|company|country|state|club|one|movie|film|"
    r"game|match|show|book|song|city|party|guy|phone|model|series|event|tournament)\b"
)
_FOLDERS = (
    "home|user folder|desktop|documents|downloads|pictures|photos|videos|music|3d objects|favorites"
)


def clean_utterance(text: str) -> str:
    """Collapse whitespace, drop a leading wake word and trailing punctuation."""

    source = re.sub(r"\s+", " ", text).strip()
    source = _WAKE_PREFIX.sub("", source).strip()
    return source.rstrip(" .!?,;:")


def _command_form(source: str) -> str:
    command = source.lower().replace(",", " ")
    command = re.sub(r"\s+", " ", command).strip()
    for _ in range(2):
        command = _POLITE_PREFIX.sub("", command).strip()
        command = _POLITE_SUFFIX.sub("", command).strip()
    return command


def _strip_prefix(source: str, pattern: str) -> str | None:
    """Return the original-case remainder after a case-insensitive prefix."""

    match = re.fullmatch(pattern, source, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_fast_path(text: str) -> Intent | None:
    """Return a safe intent for a narrow set of command patterns."""

    source = clean_utterance(text)
    if not source:
        return None
    source = _POLITE_PREFIX.sub("", source).strip()
    command = _command_form(source)
    if not command:
        return None

    memory = _memory_intent(source, command)
    if memory is not None:
        return memory

    if _TIME_QUESTIONS.fullmatch(command):
        return Intent("system_information", "time", {}, False)
    if _DATE_QUESTIONS.fullmatch(command):
        return Intent("system_information", "date", {}, False)
    if command in {"show system information", "system information", "system status", "system info",
                   "status report", "diagnostics", "run diagnostics"}:
        return Intent("system_information", None, {}, False)
    if re.fullmatch(r"(?:what(?:'s| is) (?:my |the )?)?battery(?: level| status| percentage)?|how much battery(?: is left| do i have)?", command):
        return Intent("system_information", "battery", {}, False)
    if re.fullmatch(r"(?:what(?:'s| is) (?:my |the )?)?(?:cpu|ram|memory) usage", command):
        return Intent("system_information", None, {}, False)

    for pattern, action in _SYSTEM_CONTROL:
        if pattern.fullmatch(command):
            return Intent("system_control", action, {}, False)

    youtube = re.fullmatch(r"(?:search(?: for)?|find) (.+?) (?:on|in) youtube|search youtube for (.+)|youtube (.+)", command)
    if youtube:
        query = next(group for group in youtube.groups() if group)
        return Intent("open_website", "youtube", {"query": query}, False)
    # "play shape of you (on youtube)": open the top video so it actually plays.
    play = re.fullmatch(r"play (?:the )?(?:song |video |music )?(.+?)(?: song| video)?(?: (?:on|in) youtube)?", command)
    if play:
        return Intent("open_website", "youtube", {"query": play.group(1), "play": True}, False)

    whatsapp = _whatsapp_message(source)
    if whatsapp is not None:
        return whatsapp

    if command in {"open whatsapp", "open whatsapp web", "start whatsapp", "launch whatsapp"}:
        return Intent("open_whatsapp", "whatsapp", {}, False)
    if command in {"open settings", "open windows settings", "open windows settings app", "open the settings"}:
        return Intent("open_windows_settings", None, {}, False)
    if command in {"open task manager", "start task manager", "launch task manager"}:
        return Intent("open_task_manager", None, {}, False)

    folder = re.fullmatch(rf"(?:open|show)(?: my| the)? ({_FOLDERS})(?: folder)?", command)
    if folder:
        return Intent("open_folder", folder.group(1), {}, False)

    target = re.fullmatch(r"(?:open|launch|start|go to)(?: the)? (.+?)(?: app| application| website| site)?", command)
    if target:
        name = target.group(1)
        if name in available_applications():
            return Intent("open_application", name, {}, False)
        website = resolve_website(name)
        if website is not None:
            return Intent("open_website", website, {}, False)
        resolved = resolve_application_name(name)
        if resolved is not None:
            return Intent("open_application", resolved, {}, False)
        return None

    search = _EXPLICIT_SEARCH.fullmatch(command)
    if search:
        return Intent("current_information", None, {"query": search.group(1).strip()}, False)

    # "Who was the captain of that team?" only makes sense with the conversation,
    # so let the model (which sees the history) write the search query.
    if needs_live_information(source) and not _FOLLOW_UP.search(command):
        return Intent("current_information", None, {"query": source}, False)
    return None


def is_follow_up(text: str) -> bool:
    """Does this refer back to the conversation ("that team", "he", "they")?"""

    return bool(_FOLLOW_UP.search(_command_form(clean_utterance(text))))


_CONTACT = r"(?P<contact>[^\W\d_][\w.'-]*(?: [^\W\d_][\w.'-]*){0,3}?)"
_WHATSAPP_MESSAGE = (
    # "send a (whatsapp) message to Shanju saying I'll be late"
    re.compile(
        rf"(?:please )?(?:send|drop|write)(?: an?)?(?: whatsapp| quick)?(?: message| msg| text)? to {_CONTACT}"
        rf"(?: on whatsapp)?\s*(?:saying|that says|telling (?:him|her|them)(?: that)?|that|with)\s*:?\s+(?P<message>.+)",
        re.IGNORECASE,
    ),
    # "whatsapp Shanju that I'll be late" / "message Shanju saying hi"
    re.compile(
        rf"(?:whatsapp|message|msg|text) {_CONTACT}(?: on whatsapp)?\s*(?:saying|that|:)\s*(?P<message>.+)",
        re.IGNORECASE,
    ),
    # "tell Shanju on whatsapp (that) I'll be late"
    re.compile(rf"tell {_CONTACT} (?:on|via|through) whatsapp(?: that)?\s+(?P<message>.+)", re.IGNORECASE),
    # "send 'I'll be late' to Shanju on whatsapp"
    re.compile(rf"send [\"'“]?(?P<message>.+?)[\"'”]? to {_CONTACT} (?:on|via|through) whatsapp", re.IGNORECASE),
)


def _whatsapp_message(source: str) -> Intent | None:
    """A WhatsApp message for a named contact, keeping the user's exact words."""

    for pattern in _WHATSAPP_MESSAGE:
        match = pattern.fullmatch(source.strip())
        if match is None:
            continue
        contact = match.group("contact").strip()
        message = match.group("message").strip().strip("\"'“”").strip()
        if not contact or not message or contact.lower() in {"me", "myself", "him", "her", "them", "it"}:
            return None
        if contact.islower():
            contact = " ".join(word.capitalize() for word in contact.split())
        return Intent("send_whatsapp_message", "whatsapp", {"contact": contact, "message": message}, True)
    return None


def needs_live_information(text: str) -> bool:
    """Heuristic: is this a question whose answer changes over time?"""

    lowered = text.lower()
    if _ABOUT_ASSISTANT.search(lowered):
        return False
    if _VERSIONED_THING.search(lowered):
        return True  # "iphone 18", "gpt 6", "python 3.15": likely newer than the model.
    if parse_office_question(text) is not None or _CHANGING_FACTS.search(lowered):
        return True  # Office holders and results change; never trust model memory.
    if _TOPIC_FIRST.match(lowered):
        return True  # "weather in Chennai", "latest news", "gold price today"
    if not _QUESTION_WORDS.search(lowered):
        return False
    if _STRONG_LIVE.search(lowered):
        return True
    return bool(_WEAK_LIVE.search(lowered)) and not _ABOUT_USER.search(lowered)


def _memory_intent(source: str, command: str) -> Intent | None:
    """Memory, reminder and conversation commands (keep the user's original casing)."""

    reminder = _strip_prefix(source, r"(?:please )?(?:set (?:a )?reminder (?:to |for |about )?|remind me (?:to |about |that |of )?)(.+)")
    if reminder:
        due, remainder = parse_when(reminder)
        return Intent(
            "set_reminder", None,
            {"text": remainder or reminder, "when": due.isoformat() if due else ""}, False,
        )
    if re.fullmatch(r"(?:what are |show |list |tell me )?(?:my |all )?(?:reminders|pending reminders)|do i have any reminders", command):
        return Intent("list_reminders", None, {}, False)
    if re.fullmatch(r"(?:cancel|clear|delete|remove) (?:all )?(?:my )?reminders", command):
        return Intent("clear_reminders", None, {}, True)

    todo = _strip_prefix(
        source, r"(?:please )?(?:remember to|don't let me forget to|do not let me forget to|i need to remember to) (.+)"
    )
    if todo:
        due, remainder = parse_when(todo)
        if due is not None:
            return Intent("set_reminder", None, {"text": remainder or todo, "when": due.isoformat()}, False)
        return Intent("remember_memory", None, {"content": f"To-do: {todo}"}, False)
    remember = _strip_prefix(
        source,
        r"(?:please )?(?:remember|note|note down|keep in mind|make a note|take a note|jot down|write down|"
        r"don't let me forget|do not let me forget)"
        r"(?: of| that)? (.+)",
    )
    if remember:
        return Intent("remember_memory", None, {"content": remember}, False)
    saved = _strip_prefix(source, r"(?:please )?save (.+) (?:to|in) (?:your )?memory")
    if saved:
        return Intent("remember_memory", None, {"content": saved}, False)

    if command in {"forget everything", "forget everything you remember", "forget everything about me",
                   "clear all memories", "clear my memories", "delete all memories", "erase all memories"}:
        return Intent("clear_memories", None, {}, True)
    forget = _strip_prefix(
        source,
        r"(?:please )?(?:forget(?: about| that| what i (?:said|told you) about)?|"
        r"(?:delete|remove|erase) (?:the |my )?memor(?:y|ies)(?: about| of| that)?) (.+)",
    )
    if forget is None and command in {"forget it", "forget that", "forget this", "forget"}:
        forget = "that"
    if forget is not None:
        query = forget.strip()
        if query.lower() in _PRONOUNS:
            return Intent("forget_memory", None, {"query": "", "latest": True}, False)
        return Intent("forget_memory", None, {"query": query}, False)

    if command in {"what do you remember", "what do you remember about me", "what do you know about me",
                   "show my memories", "list my memories", "show memories", "list memories",
                   "what have you remembered", "what did i tell you to remember"}:
        return Intent("list_memories", None, {}, False)
    if command in {"clear conversation", "clear this conversation", "clear our conversation",
                   "clear chat", "clear the chat", "start over", "new conversation"}:
        return Intent("clear_conversation", None, {}, True)
    return None
