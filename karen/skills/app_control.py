"""Safe launching for catalogued Windows applications and browsers."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
}

BROWSER_ALIASES = {
    "chrome": "google chrome",
    "edge": "microsoft edge",
}


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

    global _catalog
    catalog: dict[str, CatalogEntry] = {}
    for directory in _catalog_directories():
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.suffix.lower() not in {".lnk", ".url"}:
                continue
            name = _application_name(path.stem)
            if name and name not in catalog:
                catalog[name] = CatalogEntry(name, path, shortcut=True)
    _catalog = catalog
    return dict(catalog)


def available_applications() -> tuple[str, ...]:
    """Return sorted names from the built-in and discovered app catalog."""

    catalog = _catalog if _catalog is not None else refresh_application_catalog()
    return tuple(sorted(set(APPLICATIONS) | set(catalog)))


def _launch_catalog_entry(name: str) -> SkillResult:
    catalog = _catalog if _catalog is not None else refresh_application_catalog()
    discovered = catalog.get(name)
    if discovered is None:
        discovered = catalog.get(BROWSER_ALIASES.get(name, ""))
    if discovered is not None:
        try:
            os.startfile(str(discovered.launch_path))  # type: ignore[attr-defined]
        except OSError:
            return SkillResult(False, f"I couldn't open {name.title()}.")
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
    return SkillResult(True, f"Opening {name.title()}.")


def open_application(target: str | None) -> SkillResult:
    """Launch an exact built-in or Start Menu-catalogued application."""

    name = _application_name(target)
    if name is None:
        return SkillResult(False, "Tell me the exact name of the application to open.")
    if os.name != "nt":
        return SkillResult(False, "Application control is available on Windows only.")
    return _launch_catalog_entry(name)


def close_application(target: str | None) -> SkillResult:
    """Refuse closing until process identity and graceful shutdown are implemented."""

    _ = target
    return SkillResult(False, "Closing applications is not enabled yet.")
