"""Text entry point for Karen AI Phase 1.1."""

from __future__ import annotations

from pathlib import Path

from core.config import load_settings
from core.intent import parse_intent
from core.llm import LocalLLM
from core.router import IntentRouter
from karen.ai.ollama import OllamaClient
from karen.exceptions import LocalAIError


EXIT_COMMANDS = frozenset({"exit", "quit", "goodbye"})


def build_application(base_dir: Path) -> tuple[LocalLLM, IntentRouter]:
    """Compose replaceable services from local configuration."""

    settings = load_settings(base_dir / "config" / "settings.json")
    client = OllamaClient(settings.ollama_url, settings.model, settings.timeout_seconds)
    return LocalLLM(client), IntentRouter()


def process_text(text: str, llm: LocalLLM, router: IntentRouter) -> str:
    """Process one text message through LLM, validation, and routing."""

    try:
        raw = llm.generate_intent(text)
        intent = parse_intent(raw.content)
        return router.route(intent).message
    except (ValueError, LocalAIError) as error:
        return f"I couldn't safely process that request: {error}"


def main() -> None:
    """Run the interactive text loop."""

    llm, router = build_application(Path(__file__).resolve().parent)
    print("Karen: What can I help you with?")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKaren: Goodbye.")
            return
        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            print("Karen: Goodbye.")
            return
        print(f"Karen: {process_text(user_text, llm, router)}")


if __name__ == "__main__":
    main()
