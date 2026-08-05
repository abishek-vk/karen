"""Compatibility wrapper for the canonical Karen configuration API.

New code should import ``Settings`` and ``ConfigurationStore`` from
``karen.config`` directly.
"""

from __future__ import annotations

from pathlib import Path

from karen.config import ConfigurationStore, Settings


CoreSettings = Settings


def load_settings(path: Path) -> Settings:
    """Load settings through the canonical configuration store.

    The original text core accepted a file path; retain that call shape while
    delegating all parsing and migration behavior to ``karen.config``.
    """

    return ConfigurationStore(path.parent).load_settings()
