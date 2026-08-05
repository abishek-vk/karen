"""Text entry point for Karen AI Phase 1.1."""

from __future__ import annotations

from pathlib import Path

from karen.ai.ollama import OllamaClient
from karen.config import ConfigurationStore
from karen.core.confirmation import ConfirmationManager, parse_confirmation
from karen.core.fast_path import parse_fast_path
from karen.core.intent import Intent, parse_intent
from karen.core.llm import LocalLLM
from karen.core.router import IntentRouter
from karen.exceptions import LocalAIError


EXIT_COMMANDS = frozenset({"exit", "quit", "goodbye"})


def build_application(base_dir: Path) -> tuple[LocalLLM, IntentRouter]:
    """Compose replaceable services from local configuration."""

    settings = ConfigurationStore(base_dir / "config").load_settings()
    client = OllamaClient(
        settings.ollama_url,
        settings.ollama_model,
        settings.request_timeout_seconds,
    )
    return LocalLLM(client), IntentRouter()


def process_text(
    text: str,
    llm: LocalLLM,
    router: IntentRouter,
    confirmations: ConfirmationManager | None = None,
) -> str:
    """Process text through confirmation, fast-path, LLM, and routing."""

    confirmation_state = confirmations or ConfirmationManager()
    try:
        pending = confirmation_state.peek()
        if pending is not None:
            answer = parse_confirmation(text)
            if answer is True:
                approved = confirmation_state.consume()
                if approved is not None:
                    return router.route(approved, confirmed=True).message
            elif answer is False:
                confirmation_state.clear()
                return "Cancelled."
            else:
                return "Please answer yes or no before I continue with the pending action."

        intent = parse_fast_path(text)
        if intent is None:
            raw = llm.generate_intent(text)
            intent = parse_intent(raw.content)
        if router.needs_confirmation(intent):
            confirmation_state.request(intent)
            return "This action requires your confirmation. Say yes to continue or no to cancel."
        return router.route(intent).message
    except (ValueError, LocalAIError) as error:
        return f"I couldn't safely process that request: {error}"


def main() -> None:
    """Run the interactive text loop."""

    project_dir = Path(__file__).resolve().parent.parent
    llm, router = build_application(project_dir)
    confirmations = ConfirmationManager()
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
        print(f"Karen: {process_text(user_text, llm, router, confirmations)}")


if __name__ == "__main__":
    main()
