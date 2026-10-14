"""Terminal interface for Karen (typed or spoken, with optional spoken replies)."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from karen.ai.ollama import OllamaClient
from karen.config import ConfigurationStore
from karen.core.application_service import ApplicationService
from karen.core.confirmation import ConfirmationManager
from karen.core.llm import LocalLLM
from karen.core.router import IntentRouter
from karen.voice.contracts import VoiceError

EXIT_COMMANDS = frozenset({"exit", "quit", "goodbye", "bye", "shut down karen"})
_print_lock = threading.Lock()


def build_application(base_dir: Path) -> tuple[LocalLLM, IntentRouter]:
    """Compose replaceable services from local configuration (compatibility helper)."""

    settings = ConfigurationStore(base_dir / "config").load_settings()
    client = OllamaClient(
        settings.ollama_url,
        settings.ollama_model,
        settings.request_timeout_seconds,
        context_window=settings.context_window,
        keep_alive=settings.keep_alive,
    )
    return LocalLLM(client, assistant_name=settings.assistant_name), IntentRouter()


def process_text(
    text: str,
    llm: LocalLLM,
    router: IntentRouter,
    confirmations: ConfirmationManager | None = None,
) -> str:
    """Compatibility wrapper around the shared application service."""

    service = ApplicationService(llm, router, memory_enabled=False)
    if confirmations is not None:
        service.confirmations = confirmations
    return service.handle_text(text).message


def _say(text: str) -> None:
    with _print_lock:
        print(text, flush=True)


def main(*, speak: bool | None = None, voice: bool = False) -> None:
    """Run the interactive loop. ``voice`` uses the microphone instead of typing."""

    from karen.app import KarenRuntime

    runtime = KarenRuntime()
    if speak is not None:
        runtime.speaker.enabled = speak and runtime.speaker.available
    service = runtime.service
    name = runtime.settings.assistant_name

    def notify(message: str) -> None:
        _say(f"\n{name}: {message}")
        runtime.speaker.say(message)

    runtime.start_background(notify)
    greeting = f"{name}: Online. " + ("Press Enter and speak." if voice else "What can I help you with?")
    _say(greeting)
    if voice:
        runtime.recognizer.preload()

    try:
        while True:
            try:
                if voice:
                    typed = input("[Enter to talk, or type] ").strip()
                    if typed:
                        user_text = typed
                    else:
                        runtime.speaker.stop()
                        _say("Listening...")
                        try:
                            user_text = runtime.recognizer.transcribe_once()
                        except VoiceError as error:
                            _say(f"{name}: {error}")
                            continue
                        _say(f"You: {user_text}")
                else:
                    user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                _say(f"\n{name}: Goodbye.")
                return
            if not user_text:
                continue
            if user_text.lower().strip(" .!") in EXIT_COMMANDS:
                _say(f"{name}: Goodbye.")
                runtime.speaker.speak("Goodbye.")
                return
            runtime.speaker.stop()
            with _print_lock:
                sys.stdout.write(f"{name}: ")
                sys.stdout.flush()

            def on_chunk(chunk: str) -> None:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                runtime.speaker.feed(chunk)

            def on_status(state: str) -> None:
                if state == "searching":
                    sys.stdout.write("(searching the web) ")
                    sys.stdout.flush()

            response = service.handle_text(user_text, on_chunk=on_chunk, on_status=on_status)
            runtime.speaker.flush()
            sources = response.message.rpartition("\nSources: ")[2] if "\nSources: " in response.message else ""
            _say(f"\n  [sources: {sources}]" if sources else "")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
