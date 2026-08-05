"""Phase 1 application composition root."""

from __future__ import annotations

import logging
from pathlib import Path

from .ai.ollama import OllamaClient
from .config import ConfigurationStore, Settings
from .logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


class KarenApplication:
    """Compose Phase 1 services without starting hardware or background workers."""

    def __init__(self, data_dir: Path) -> None:
        self.configuration: ConfigurationStore = ConfigurationStore(data_dir)
        self.settings: Settings = self.configuration.load_settings()
        configure_logging(data_dir, self.settings.log_level)
        self.ollama: OllamaClient = OllamaClient(
            base_url=self.settings.ollama_url,
            model=self.settings.ollama_model,
            timeout_seconds=self.settings.request_timeout_seconds,
        )
        LOGGER.info("Karen initialized in local-only mode")
