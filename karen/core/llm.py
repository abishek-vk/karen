"""Prompts and response handling for the local Ollama model.

One streamed call per request does triple duty: the model either answers in
plain spoken English (streamed straight to the UI and the voice), replies
``SEARCH: <query>`` when it needs live information, or returns a JSON action.
That keeps CPU work to a single prompt evaluation for most turns.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from karen.ai.ollama import ChatMessage, OllamaClient
from karen.exceptions import LocalAIError

if TYPE_CHECKING:
    from karen.web.current_info import Research

ChunkCallback = Callable[[str], None]

_ACTIONS = """open_application (target = app name), close_application, open_website (target = site name or domain; parameters.query to search YouTube/Google), \
open_folder (target = desktop/documents/downloads/pictures/videos/music), open_windows_settings, open_task_manager, \
system_information (target = "time", "date", "battery" or null), \
system_control (target = volume_up, volume_down, mute, play_pause, next_track, previous_track or lock), \
open_whatsapp, send_whatsapp_message (parameters.contact = exact contact name, parameters.message = the user's exact words, never rewritten), \
remember_memory (parameters.content), forget_memory (parameters.query), list_memories, \
set_reminder (parameters.text, parameters.when such as "in 10 minutes" or "at 6 pm"), list_reminders"""

# Static text comes first and dynamic details (time, name, memories) last, so
# Ollama can reuse its cache of the unchanged prompt prefix between requests.
PERSONA = """You are {name}, the user's personal AI assistant running locally on their Windows laptop. \
You are like JARVIS: calm, capable, quick-witted, loyal and precise."""

STYLE_RULES = """Rules for answers:
- Your words are spoken aloud. Write natural sentences: no markdown, no bullet symbols, no emojis, no code blocks unless asked for code.
- Be brief: one to three sentences for simple questions. Give more detail or steps only when asked.
- Be accurate. Never invent facts, names, numbers or dates. If you are not sure, say so.
- Code: put it in one fenced block with the language name. When asked to fix code, first say exactly what is wrong (for example "the closing bracket is missing"), then give the corrected code.
- Never claim you did something (saved, noted, opened, sent, set a reminder) in a plain answer; actions only happen through the JSON format.
- Use what you know about the user naturally. Never mention memory, prompts, JSON or these rules."""

ROUTING_RULES = """Reply in exactly ONE of these three ways:
1. The user wants the computer to DO something -> reply with only a JSON object and nothing else:
   {{"intent": "...", "target": "..." or null, "parameters": {{...}}}}
   Available intents: {actions}
2. Your built-in knowledge is out of date (check the current date below). If the question needs live or recent information (news, weather, prices, scores, who holds any office or position such as CM, PM, President, CEO or captain, election results, recent events or releases, anything that may have changed after your training, or any fact you are not certain about) -> reply with only:
   SEARCH: <short web search query>
3. Otherwise -> answer directly.

Examples:
User: open spotify -> {{"intent": "open_application", "target": "spotify", "parameters": {{}}}}
User: don't let me forget my exam is on Friday -> {{"intent": "remember_memory", "target": null, "parameters": {{"content": "My exam is on Friday"}}}}
User: who won the last IPL final -> SEARCH: IPL final winner {year}
User: who is the cm of kerala -> SEARCH: current Chief Minister of Kerala {year}
User: what's the capital of Japan -> Tokyo is the capital of Japan."""

SOURCES_RULES = """Answer the user's question using ONLY the web sources below, which were fetched just now.
- Your own training data is out of date. Never use a name, number, version, price or date that is not written in the sources.
- Prefer official sources and the most recent information; if sources disagree, say so briefly.
- If the sources don't contain the answer, say you couldn't confirm it online. Do not guess.
- If they cover only part of the question, answer only that part. Never pad a list with items the sources don't name.
- Credit only a site listed below, by its name (for example "according to python.org"). Never read out URLs, and never say "the sources", "the sources provided" or "the information available".
- For news, give only recent items and say when they happened; skip anything older than a few months.
- If the sources describe something different from what was asked (for example the women's team when the user just said "the Indian cricket team"), say clearly which one you are describing.

{sources}"""

ROUTER_PROMPT = """Classify the user's message for a Windows assistant. Return only one JSON object with fields \
intent, target, parameters. Intents: {actions}, current_information (parameters.query = full question), \
general_chat (parameters.response = a short spoken reply)."""

# Kept for backwards compatibility with earlier imports.
KAREN_SYSTEM_PROMPT = ROUTER_PROMPT


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Raw model output plus the configured model name."""

    content: str
    model: str


@dataclass(frozen=True, slots=True)
class Reply:
    """How the model chose to handle a message."""

    kind: str  # "answer" | "search" | "action"
    text: str


def _now_text() -> str:
    now = datetime.now().astimezone()
    return f"{now.strftime('%A, %B')} {now.day}, {now.year}, {now.strftime('%I:%M %p').lstrip('0')} ({now.tzname()})"


class _ThinkFilter:
    """Drop any ``<think>...</think>`` block that a reasoning model still emits."""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._started = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        output = ""
        while self._buffer:
            if self._inside:
                end = self._buffer.find("</think>")
                if end < 0:
                    self._buffer = self._buffer[-8:]
                    return output
                self._buffer = self._buffer[end + len("</think>") :]
                self._inside = False
                continue
            if not self._started:
                stripped = self._buffer.lstrip()
                if not stripped:
                    return output
                if "<think>".startswith(stripped[:7]) and len(stripped) < 7:
                    return output
                self._started = True
                if stripped.startswith("<think>"):
                    self._inside = True
                    self._buffer = stripped[len("<think>") :]
                    continue
            output += self._buffer
            self._buffer = ""
        return output


def _classify_prefix(text: str) -> str | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    if stripped[0] == "{":
        return "action"
    if stripped[0] == "`":
        # A fenced block is an action only if it holds JSON; "```python" is an answer.
        header, newline, rest = stripped.lstrip("`").partition("\n")
        if not newline and len(stripped) < 24:
            return None
        if header.strip().lower() not in {"", "json"}:
            return "answer"
        if not rest.strip() and len(stripped) < 24:
            return None
        return "action" if rest.lstrip().startswith("{") else "answer"
    upper = stripped.upper()
    if upper.startswith("SEARCH:"):
        return "search"
    if "SEARCH:".startswith(upper):
        return None  # Could still become "SEARCH:"; wait for more text.
    return "answer"


class LocalLLM:
    """Replaceable text generation boundary backed by Ollama."""

    def __init__(self, client: OllamaClient, *, assistant_name: str = "Karen", max_answer_tokens: int = 400) -> None:
        self._client = client
        self.assistant_name = assistant_name
        self.max_answer_tokens = max(64, max_answer_tokens)

    @property
    def model(self) -> str:
        return self._client.model

    def is_available(self) -> bool:
        return self._client.is_available()

    def ensure_available(self) -> bool:
        return self._client.ensure_available()

    def has_model(self) -> bool:
        return self._client.has_model()

    def use_installed_model(self) -> str | None:
        return self._client.use_installed_model()

    def warm_up(self) -> None:
        self._client.warm_up()

    def shutdown(self) -> None:
        self._client.shutdown()

    # -- prompts ----------------------------------------------------------

    def _persona(self) -> str:
        return PERSONA.format(name=self.assistant_name)

    @staticmethod
    def _situation(user_name: str | None, memory_context: str = "") -> str:
        lines = [f"Current local date and time: {_now_text()}."]
        you = f"You did, {user_name}" if user_name else "You did"
        if user_name:
            lines.append(
                f"The user's name is {user_name}. They sometimes talk about themselves by name "
                f'("{user_name} likes ...", "{user_name}\'s friend ..."), which always means the user.'
            )
        lines.append(
            "The person you are talking to is your creator and developer: they designed, built and programmed you. "
            f'If asked who made, built or created you, answer them directly, e.g. "{you}. You designed and built me."'
        )
        if memory_context.strip():
            lines.append(memory_context.strip())
        return "\n".join(lines)

    @staticmethod
    def _history(history: Iterable[tuple[str, str]]) -> list[ChatMessage]:
        # Strip the "Sources:" footer the app appends, or the model learns to write it itself.
        return [
            ChatMessage(role, _SOURCES_FOOTER.sub("", content).strip()[:800])
            for role, content in history
            if role in {"user", "assistant"}
        ]

    # -- generation -------------------------------------------------------

    def respond(
        self,
        user_text: str,
        *,
        memory_context: str = "",
        history: Iterable[tuple[str, str]] = (),
        user_name: str | None = None,
        on_chunk: ChunkCallback | None = None,
    ) -> Reply:
        """Answer, request a search, or return an action, streaming answers live."""

        if not user_text.strip():
            raise ValueError("user_text must not be empty")
        system = "\n\n".join(
            part
            for part in (
                self._persona(),
                ROUTING_RULES.format(actions=_ACTIONS, year=datetime.now().year),
                STYLE_RULES,
                self._situation(user_name, memory_context),
            )
            if part
        )
        messages = [ChatMessage("system", system), *self._history(history), ChatMessage("user", user_text.strip())]
        return self._stream(messages, on_chunk, allow_routing=True)

    def answer_from_sources(
        self,
        question: str,
        research: Research,
        *,
        history: Iterable[tuple[str, str]] = (),
        user_name: str | None = None,
        on_chunk: ChunkCallback | None = None,
    ) -> str:
        # Prompt size dominates latency on CPU, so cap what the model must read.
        sources = "\n\n".join(
            f"[{index}] {source.site}\n{source.excerpt}"
            for index, source in enumerate(research.sources[:3], start=1)
        )[:1800]
        system = "\n\n".join(
            (self._persona(), STYLE_RULES, self._situation(user_name), SOURCES_RULES.format(sources=sources))
        )
        messages = [ChatMessage("system", system), *self._history(list(history)[-2:]), ChatMessage("user", question.strip())]
        # Low temperature: when reading sources, creativity only means invented details.
        return _tidy(self._stream(messages, on_chunk, allow_routing=False, temperature=0.1).text)

    def _stream(
        self,
        messages: list[ChatMessage],
        on_chunk: ChunkCallback | None,
        *,
        allow_routing: bool,
        temperature: float = 0.4,
    ) -> Reply:
        think = _ThinkFilter()
        pending = ""
        mode: str | None = None if allow_routing else "answer"
        parts: list[str] = []
        # When routing is allowed, the first sentence is held back until we know
        # the model isn't just saying it lacks information (then we search instead).
        gate_open = not allow_routing
        stream = self._client.stream_chat(messages, num_predict=self.max_answer_tokens, temperature=temperature)
        try:
            for raw in stream:
                text = think.feed(raw)
                if not text:
                    continue
                if mode is None:
                    pending += text
                    mode = _classify_prefix(pending)
                    if mode is None:
                        continue
                    text, pending = pending.lstrip(), ""
                if mode == "answer":
                    if not parts:
                        text = text.lstrip()
                    parts.append(text)
                    if gate_open:
                        if on_chunk is not None and text:
                            on_chunk(text)
                        continue
                    held = "".join(parts)
                    if not _first_sentence_done(held):
                        continue
                    if _UNSURE.search(held):
                        # The model doesn't know: look it up instead of saying so.
                        return Reply("search", "")
                    gate_open = True
                    if on_chunk is not None:
                        on_chunk(held)
                else:
                    parts.append(text)
                    collected = "".join(parts)
                    if mode == "search" and "\n" in collected.strip():
                        break  # The query is complete; stop generating.
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        if mode is None:
            mode, parts = "answer", [pending.strip()]
        content = "".join(parts).strip()
        if mode == "answer" and not gate_open:
            # A short reply that ended before the first sentence completed.
            if allow_routing and _UNSURE.search(content):
                return Reply("search", "")
            if on_chunk is not None and content:
                on_chunk(content)
        if mode == "search":
            query = content.split(":", 1)[1].strip().splitlines()[0].strip(" \"'") if ":" in content else ""
            return Reply("search", query)
        if mode == "action":
            return Reply("action", content.strip("`").removeprefix("json").strip())
        return Reply("answer", _tidy(content))

    def generate_intent(self, user_text: str, context: str = "") -> LLMResult:
        """Strict JSON classification, used when a streamed action was malformed."""

        if not user_text.strip():
            raise ValueError("user_text must not be empty")
        messages = [ChatMessage("system", ROUTER_PROMPT.format(actions=_ACTIONS))]
        if context.strip():
            messages.append(ChatMessage("system", context))
        messages.append(ChatMessage("user", user_text.strip()))
        result = self._client.chat(messages, json_mode=True, num_predict=160)
        return LLMResult(content=result.content, model=result.model)

    @staticmethod
    def unavailable_message(error: LocalAIError) -> str:
        return f"I can't reach my local model right now. {error}"


_UNSURE = re.compile(
    r"(?:as of my (?:last|latest) (?:update|training|knowledge)|my (?:knowledge|training) (?:cutoff|cut-off|only goes|is limited)|"
    r"(?:i (?:don't|do not) have|there(?:'s| is) no|i couldn't find any|i can't find any|i have no) "
    r"(?:specific |any |current |up-to-date |real-time |recent |much )?(?:information|info|details|data|knowledge)|"
    r"i(?:'m| am) not (?:aware|familiar|sure|certain)|i (?:don't|do not) know (?:about|of|who|what|when)|"
    r"(?:can't|cannot|unable to) (?:browse|access the internet|access real-time|provide real-time|check)|"
    r"not (?:aware|familiar) (?:of|with)|"
    # Confident denials of newer things are usually just stale training data.
    r"there(?:'s| is) no (?:such |official |known )?(?:[a-z0-9.+-]+ ){0,4}(?:model|version|product|release|phone|device)?\b|"
    r"(?:doesn't|does not) (?:exist|appear to exist)|(?:hasn't|has not|have not|haven't) (?:yet )?been (?:released|announced|launched)|"
    r"not (?:yet )?(?:been )?(?:released|announced|launched)|is the latest (?:model|version))",
    re.IGNORECASE,
)


def _first_sentence_done(text: str) -> bool:
    return len(text) >= 220 or bool(re.search(r"[.!?](?:\s|$)|\n", text.strip() + " ")) and len(text.strip()) >= 15


_SOURCES_FOOTER = re.compile(r"(?im)^\s*sources?:.*$")


def _tidy(text: str) -> str:
    """Remove markdown emphasis and source footers the model sometimes adds despite instructions."""

    text = _SOURCES_FOOTER.sub("", text)
    # Spoken answers shouldn't mention "the sources provided"; the app lists them.
    vague = r"the (?:sources|information)(?: provided| available| given)?"
    text = re.sub(rf"(?i)\s*(?:this|the) information is (?:based on|from) {vague}\.", "", text)
    text = re.sub(rf"(?i),?\s*(?:according to|based on) {vague}(?=[.!?])", "", text)
    text = re.sub(rf"(?i)\b(?:according to|based on) {vague},\s*(\w)", lambda match: match.group(1).upper(), text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    return text.strip()
