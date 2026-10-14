"""Shared application service used by the text, voice and graphical interfaces.

Request flow:
1. Clean the utterance (wake word, filler, punctuation).
2. Resolve a pending confirmation, if any.
3. Fast path: deterministic commands run instantly without the model.
4. Local model: one streamed call answers, requests a web search, or returns an action.
5. Web research: fetch sources and let the model answer from them (or summarize
   extractively when the model is offline).
"""

from __future__ import annotations

import ctypes
import logging
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from karen.exceptions import LocalAIError
from karen.memory import MemoryService
from karen.memory.reminders import ReminderService
from karen.skills.base import SkillManifest
from karen.web import CurrentInfoService

from .confirmation import ConfirmationManager, parse_confirmation
from .fast_path import clean_utterance, is_follow_up, needs_live_information, parse_fast_path
from .intent import Intent, parse_intent
from .llm import LocalLLM
from .router import IntentRouter
from .skill_registry import RegisteredSkill

LOGGER = logging.getLogger(__name__)

ChunkCallback = Callable[[str], None]
StatusCallback = Callable[[str], None]

_QUESTION_START = re.compile(
    r"^(?:who|what|what's|whats|which|where|when|why|how|is|are|was|were|do|does|did|can|could|will|should|tell me|explain|define)\b",
    re.IGNORECASE,
)


_CLAIMED_ACTION = re.compile(
    r"\b(?:i(?:'ve| have) (?:noted|saved|stored|added|set|scheduled|opened|sent|created)|"
    r"i(?:'ll| will) remind you|reminder (?:is )?set|noted that|i've made a note)\b",
    re.IGNORECASE,
)


_FIX_REQUEST = re.compile(r"\b(?:fix|debug|error|bug|wrong|broken|not working|doesn't work|does not work)\b", re.IGNORECASE)
_FENCED = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _python_syntax_note(text: str) -> str:
    """For "fix this Python code", report what Python's own parser says.

    A small model easily misses an unclosed bracket; the parser never does. The
    code is only compiled (parsed), never executed.
    """

    if "python" not in text.lower() or not _FIX_REQUEST.search(text):
        return ""
    fenced = _FENCED.search(text)
    code = fenced.group(1) if fenced else text.split(":", 1)[1] if ":" in text else ""
    code = code.strip()
    if not code or len(code) > 4000:
        return ""
    try:
        compile(code, "<user code>", "exec")
    except SyntaxError as error:
        return f"Python's syntax check of the user's code reports: SyntaxError: {error.msg} (line {error.lineno})."
    except (ValueError, OverflowError):
        return ""
    return ""


@contextmanager
def _keep_awake() -> Iterator[None]:
    """Stop Windows idling into sleep mid-answer (a CPU answer can take a minute)."""

    set_state = getattr(getattr(getattr(ctypes, "windll", None), "kernel32", None), "SetThreadExecutionState", None)
    if set_state is None:
        yield
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    set_state(es_continuous | es_system_required)
    try:
        yield
    finally:
        set_state(es_continuous)


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    """User-facing response plus execution metadata for interfaces."""

    message: str
    source: str
    success: bool


def _confirmation_prompt(intent: Intent) -> str:
    if intent.name == "send_whatsapp_message":
        contact = intent.parameters.get("contact", "the selected contact")
        message = intent.parameters.get("message", "")
        return f"Send this exact message to {contact}?\n“{message}”\nSay yes to send or no to cancel."
    descriptions = {
        "clear_memories": "This will erase everything I remember about you.",
        "clear_conversation": "This will clear our conversation history.",
        "clear_reminders": "This will cancel all your pending reminders.",
        "close_application": f"This will close {intent.target or 'that application'}.",
    }
    lead = descriptions.get(intent.name, "This action needs your confirmation.")
    return f"{lead} Should I go ahead? Say yes or no."


class ApplicationService:
    """Coordinate input, policy, local AI, web research and trusted skills."""

    def __init__(
        self,
        llm: LocalLLM,
        router: IntentRouter,
        *,
        memory_path: Path | None = None,
        memory_enabled: bool = True,
        recent_message_limit: int = 6,
        max_memory_results: int = 5,
        memory: MemoryService | None = None,
        live_information: CurrentInfoService | None = None,
        live_information_enabled: bool = True,
        live_information_timeout_seconds: float = 8.0,
        web_max_pages: int = 3,
        reminders: ReminderService | None = None,
    ) -> None:
        self.llm = llm
        self.router = router
        self.confirmations = ConfirmationManager()
        self._lock = threading.Lock()
        self.live_information = live_information or CurrentInfoService(
            enabled=live_information_enabled,
            timeout_seconds=live_information_timeout_seconds,
            max_pages=web_max_pages,
        )
        self.memory = memory or MemoryService(
            memory_path,
            enabled=memory_enabled,
            recent_message_limit=recent_message_limit,
            max_memory_results=max_memory_results,
        )
        self.reminders = reminders or ReminderService(self.memory)
        self._register(
            "current_information", {"current_information"}, {"live_information_lookup"},
            self.live_information.handle_intent,
        )
        self._register(
            "memory",
            {"remember_memory", "forget_memory", "list_memories", "clear_memories", "clear_conversation"},
            {"local_memory"},
            self.memory.handle_intent,
        )
        self._register(
            "reminders", {"set_reminder", "list_reminders", "clear_reminders"}, {"local_reminders"},
            self.reminders.handle_intent,
        )

    def _register(self, name: str, intents: set[str], capabilities: set[str], handler) -> None:  # noqa: ANN001
        if any(self.router.registry.resolve(intent) is not None for intent in intents):
            return
        self.router.registry.register(
            RegisteredSkill(
                SkillManifest(name=name, intents=frozenset(intents), capabilities=frozenset(capabilities), risk_level="low"),
                handler,
            )
        )

    # -- public API -------------------------------------------------------

    def handle_text(
        self,
        text: str,
        on_chunk: ChunkCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> ServiceResponse:
        """Process one message.

        ``on_chunk`` receives the reply text progressively (streamed answers)
        or all at once (commands), so UIs and speech can always rely on it.
        ``on_status`` receives short progress states such as ``"searching"``.
        """

        cleaned = clean_utterance(text)
        if not cleaned:
            if not re.search(r"\w", text):
                return ServiceResponse("I didn't catch that.", "input", False)
            # Only the wake word ("Hey Karen"): acknowledge like a butler would.
            name = self.memory.user_name()
            response = ServiceResponse(f"Yes{', ' + name if name else ''}? How can I help?", "wake", True)
            if on_chunk is not None:
                on_chunk(response.message)
            return response
        streamed: list[bool] = [False]

        def chunk(value: str) -> None:
            streamed[0] = True
            if on_chunk is not None:
                on_chunk(value)

        with self._lock, _keep_awake():
            history = self.memory.recent_turns()
            self.memory.record_message("user", cleaned)
            try:
                response = self._process(cleaned, history, chunk, on_status)
            except (ValueError, LocalAIError) as error:
                response = ServiceResponse(f"Sorry, I couldn't handle that: {error}", "error", False)
            except Exception:  # pragma: no cover - defensive: never break the UI loop
                LOGGER.exception("Unexpected failure while handling a request")
                response = ServiceResponse("Something went wrong on my side. Please try again.", "error", False)
            # Errors are not saved, so they can't mislead later answers.
            if response.source != "error":
                self.memory.record_message("assistant", response.message)
        if on_chunk is not None and not streamed[0]:
            on_chunk(response.message)
        return response

    # -- pipeline ---------------------------------------------------------

    def _process(
        self,
        text: str,
        history: list[tuple[str, str]],
        on_chunk: ChunkCallback,
        on_status: StatusCallback | None,
    ) -> ServiceResponse:
        prefix = ""
        if self.confirmations.peek() is not None:
            answer = parse_confirmation(text)
            if answer is True:
                approved = self.confirmations.consume()
                if approved is not None:
                    result = self.router.route(approved, confirmed=True)
                    return ServiceResponse(result.message, "confirmed", result.success)
            elif answer is False:
                self.confirmations.clear()
                return ServiceResponse("Cancelled.", "confirmation", True)
            else:
                # A new request replaces the pending one instead of blocking.
                self.confirmations.clear()
                prefix = "Okay, I cancelled the pending action. "

        intent = parse_fast_path(text)
        if intent is not None:
            return self._with_prefix(prefix, self._execute(intent, "fast_path", text, history, on_chunk, on_status))
        previous = next((content for role, content in reversed(history) if role == "user"), "")
        if previous and is_follow_up(text) and needs_live_information(text) and isinstance(self.llm, LocalLLM):
            # "Who was the captain of that team?" still needs live data; the earlier
            # question supplies the context for the search, and the answer step sees
            # the conversation to resolve "that team".
            return self._with_prefix(prefix, self._research(f"{previous} {text}", text, history, on_chunk, on_status))
        return self._with_prefix(prefix, self._ask_model(text, history, on_chunk, on_status))

    @staticmethod
    def _with_prefix(prefix: str, response: ServiceResponse) -> ServiceResponse:
        return replace(response, message=prefix + response.message) if prefix else response

    def _ask_model(
        self,
        text: str,
        history: list[tuple[str, str]],
        on_chunk: ChunkCallback,
        on_status: StatusCallback | None,
    ) -> ServiceResponse:
        if on_status is not None:
            on_status("thinking")
        try:
            reply = self.llm.respond(
                text,
                memory_context="\n".join(filter(None, (self.memory.context_for(text), _python_syntax_note(text)))),
                history=history,
                user_name=self.memory.user_name(),
                on_chunk=on_chunk,
            )
        except LocalAIError as error:
            return self._offline(text, error, on_status)

        if reply.kind == "answer":
            if not reply.text:
                return ServiceResponse("I'm not sure how to answer that.", "local_llm", False)
            if _CLAIMED_ACTION.search(reply.text):
                return self._verify_claimed_action(text, reply.text, history, on_chunk, on_status)
            return ServiceResponse(reply.text, "local_llm", True)
        if reply.kind == "search":
            return self._research(reply.text or text, text, history, on_chunk, on_status)

        try:
            intent = parse_intent(reply.text)
        except ValueError:
            try:
                intent = parse_intent(self.llm.generate_intent(text).content)
            except (ValueError, LocalAIError):
                return ServiceResponse("Sorry, I didn't understand that command. Could you rephrase it?", "local_llm", False)
        # Confirmation policy comes from the application, never from the model.
        intent = replace(intent, requires_confirmation=False)
        if intent.name == "general_chat":
            response = intent.parameters.get("response")
            if isinstance(response, str) and response.strip():
                return ServiceResponse(response.strip(), "local_llm", True)
            return ServiceResponse("I'm not sure how to help with that.", "local_llm", False)
        return self._execute(intent, "local_llm", text, history, on_chunk, on_status)

    def _verify_claimed_action(
        self,
        text: str,
        answer: str,
        history: list[tuple[str, str]],
        on_chunk: ChunkCallback,
        on_status: StatusCallback | None,
    ) -> ServiceResponse:
        """Small models sometimes *say* "I've noted that" without acting; make it true."""

        try:
            intent = parse_intent(self.llm.generate_intent(text).content)
        except (ValueError, LocalAIError):
            return ServiceResponse(answer, "local_llm", True)
        if intent.name in {"general_chat", "current_information"}:
            return ServiceResponse(answer, "local_llm", True)
        intent = replace(intent, requires_confirmation=False)
        result = self._execute(intent, "local_llm", text, history, on_chunk, on_status)
        # Keep the model's wording when the action really succeeded.
        return replace(result, message=answer) if result.success and result.source == "local_llm" else result

    def _execute(
        self,
        intent: Intent,
        source: str,
        text: str,
        history: list[tuple[str, str]],
        on_chunk: ChunkCallback,
        on_status: StatusCallback | None,
    ) -> ServiceResponse:
        if intent.name == "current_information" and isinstance(self.llm, LocalLLM):
            query = intent.parameters.get("query")
            return self._research(query if isinstance(query, str) and query.strip() else text, text, history, on_chunk, on_status)
        if self.router.needs_confirmation(intent):
            self.confirmations.request(intent)
            return ServiceResponse(_confirmation_prompt(intent), "confirmation", False)
        result = self.router.route(intent)
        return ServiceResponse(result.message, source, result.success)

    def _research(
        self,
        query: str,
        question: str,
        history: list[tuple[str, str]],
        on_chunk: ChunkCallback,
        on_status: StatusCallback | None,
    ) -> ServiceResponse:
        if not self.live_information.enabled:
            return ServiceResponse("Web lookups are turned off in my settings, so I can't check that.", "web", False)
        if on_status is not None:
            on_status("searching")
        research = self.live_information.research(query)
        if not research.sources:
            return ServiceResponse("I couldn't find anything reliable online about that right now.", "web", False)
        sites = ", ".join(dict.fromkeys(source.site for source in research.sources[:3]))
        try:
            if on_status is not None:
                on_status("thinking")
            answer = self.llm.answer_from_sources(
                question, research, history=history, user_name=self.memory.user_name(), on_chunk=on_chunk
            )
        except LocalAIError:
            return ServiceResponse(self.live_information.summarize(research), "web", True)
        if not answer.strip():
            return ServiceResponse(self.live_information.summarize(research), "web", True)
        return ServiceResponse(f"{answer.strip()}\nSources: {sites}", "web", True)

    def _offline(self, text: str, error: LocalAIError, on_status: StatusCallback | None) -> ServiceResponse:
        """Stay useful when Ollama or the model is missing."""

        LOGGER.info("Local model unavailable: %s", error)
        if self.live_information.enabled and (_QUESTION_START.search(text) or needs_live_information(text)):
            if on_status is not None:
                on_status("searching")
            research = self.live_information.research(text)
            if research.sources:
                return ServiceResponse(self.live_information.summarize(research), "web", True)
        return ServiceResponse(
            "My local model isn't running, so right now I can only handle direct commands, like "
            "open Chrome, what time is it, set a reminder, or search the web for something. "
            f"({error})",
            "offline",
            False,
        )
