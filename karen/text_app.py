"""Text entry point for Karen AI Phase 1.1."""

from __future__ import annotations

from pathlib import Path

from karen.ai.ollama import OllamaClient
from karen.config import ConfigurationStore
from karen.core.application_service import ApplicationService
from karen.core.confirmation import ConfirmationManager
from karen.core.llm import LocalLLM
from karen.core.router import IntentRouter



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
    """Compatibility wrapper around the shared application service."""

    service = ApplicationService(llm, router, memory_enabled=False)
    if confirmations is not None:
        service.confirmations = confirmations
    return service.handle_text(text).message


def main() -> None:
    """Run the interactive text loop."""

    project_dir = Path(__file__).resolve().parent.parent
    llm, router = build_application(project_dir)
    settings = ConfigurationStore(project_dir / "config").load_settings()
    service = ApplicationService(
        llm,
        router,
        memory_path=project_dir / "data" / "karen.db",
        memory_enabled=settings.memory_enabled,
        recent_message_limit=settings.recent_message_limit,
        max_memory_results=settings.max_memory_results,
        live_information_enabled=settings.live_information_enabled,
        live_information_timeout_seconds=settings.live_information_timeout_seconds,
    )
    confirmations = service.confirmations
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
        print(f"Karen: {service.handle_text(user_text).message}")


if __name__ == "__main__":
    main()
