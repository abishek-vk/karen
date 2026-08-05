"""Command-line bootstrap for development diagnostics only."""

from pathlib import Path

from .app import KarenApplication


if __name__ == "__main__":
    application = KarenApplication(Path(__file__).resolve().parent.parent / "data")
    print(f"Karen AI initialized with model {application.settings.ollama_model}")
