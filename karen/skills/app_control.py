"""Safe launching for catalogued Windows applications and browsers."""

from __future__ import annotations

import difflib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from karen.skills import focus
from karen.skills.base import SkillResult


APPLICATIONS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "chrome": "chrome.exe",
    "edge": "msedge.exe",
    "firefox": "firefox.exe",
    "brave": "brave.exe",
    "opera": "launcher.exe",
    "vs code": "code.exe",
    "vscode": "code.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
}

APP_ALIASES = {
    "chrome": "google chrome",
    "edge": "microsoft edge",
    "vs code": "visual studio code",
    "vscode": "visual studio code",
    "code": "visual studio code",
    "word": "word",
    "ms word": "word",
    "microsoft word": "word",
    "excel": "excel",
    "powerpoint": "powerpoint",
    "teams": "microsoft teams",
    "spotify": "spotify",
    "camera": "camera",
    "paint": "paint",
    "terminal": "terminal",
    "command prompt": "command prompt",
}
# Kept for backwards compatibility with older imports.
BROWSER_ALIASES = APP_ALIASES
_EXCLUDED_SHORTCUT_WORDS = ("uninstall", "readme", "documentation", "help", "release notes", "website")
_REFRESH_INTERVAL_SECONDS = 30.0
_last_refresh = 0.0


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """A user-visible application that Karen is allowed to launch."""

    name: str
    launch_path: Path
    shortcut: bool = False


_catalog: dict[str, CatalogEntry] | None = None


def _application_name(target: str | None) -> str | None:
    if not target:
        return None
    return " ".join(target.strip().lower().removesuffix(".exe").split()) or None


def _catalog_directories() -> tuple[Path, ...]:
    app_data = os.environ.get("APPDATA")
    program_data = os.environ.get("PROGRAMDATA")
    locations = []
    if app_data:
        locations.append(Path(app_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    if program_data:
        locations.append(Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return tuple(locations)


def refresh_application_catalog() -> dict[str, CatalogEntry]:
    """Discover Start Menu shortcuts without executing or importing them."""

    global _catalog, _last_refresh
    catalog: dict[str, CatalogEntry] = {}
    for directory in _catalog_directories():
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.suffix.lower() not in {".lnk", ".url"}:
                continue
            name = _application_name(path.stem)
            if name and any(word in name for word in _EXCLUDED_SHORTCUT_WORDS):
                continue
            if name and name not in catalog:
                catalog[name] = CatalogEntry(name, path, shortcut=True)
    _catalog = catalog
    _last_refresh = time.monotonic()
    return dict(catalog)


def available_applications() -> tuple[str, ...]:
    """Return sorted names from the built-in and discovered app catalog."""

    catalog = _catalog if _catalog is not None else refresh_application_catalog()
    return tuple(sorted(set(APPLICATIONS) | set(catalog)))


def resolve_application_name(target: str | None) -> str | None:
    """Resolve exact or high-confidence spoken names to catalog names."""

    name = _application_name(target)
    if name is None:
        return None
    choices = available_applications()
    if name in choices:
        return name
    alias = APP_ALIASES.get(name)
    if alias in choices:
        return alias
    # A high cutoff avoids launching a different app than the one asked for.
    matches = difflib.get_close_matches(name, choices, n=1, cutoff=0.8)
    return matches[0] if matches else None


def _launch_catalog_entry(name: str) -> SkillResult:
    catalog = _catalog if _catalog is not None else refresh_application_catalog()
    alias = APP_ALIASES.get(name, "")
    discovered = catalog.get(name) or catalog.get(alias)
    if discovered is None and time.monotonic() - _last_refresh > _REFRESH_INTERVAL_SECONDS:
        # Apps can be installed after Karen starts, so refresh (rate-limited)
        # before falling back to PATH or reporting that the app is missing.
        catalog = refresh_application_catalog()
        discovered = catalog.get(name) or catalog.get(alias)
    hints = [name, alias]
    before = focus.snapshot()
    if discovered is not None:
        try:
            os.startfile(str(discovered.launch_path))  # type: ignore[attr-defined]
        except OSError:
            return SkillResult(False, f"I couldn't open {name.title()}.")
        focus.bring_to_front(before, [*hints, discovered.name, discovered.launch_path.stem])
        return SkillResult(True, f"Opening {name.title()}.")

    executable = APPLICATIONS.get(name)
    if executable is None:
        return SkillResult(
            False,
            "I couldn't find that app. Use its exact Start Menu name or refresh the catalog.",
        )
    try:
        subprocess.Popen([executable], shell=False, close_fds=True)
    except OSError:
        return SkillResult(False, f"I couldn't open {name.title()}.")
    focus.bring_to_front(before, [*hints, Path(executable).stem])
    return SkillResult(True, f"Opening {name.title()}.")


def open_application(target: str | None) -> SkillResult:
    """Launch an exact built-in or Start Menu-catalogued application."""

    name = resolve_application_name(target)
    if name is None:
        return SkillResult(False, "I couldn't match that voice command to an installed application.")
    if os.name != "nt":
        return SkillResult(False, "Application control is available on Windows only.")
    return _launch_catalog_entry(name)


def close_application(target: str | None) -> SkillResult:
    """Refuse closing until process identity and graceful shutdown are implemented."""

    _ = target
    return SkillResult(False, "Closing applications is not enabled yet.")
