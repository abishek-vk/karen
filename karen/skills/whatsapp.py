"""Confirmation-gated WhatsApp Web automation on a dedicated browser thread."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from . import focus
from .base import SkillResult

LOGGER = logging.getLogger(__name__)


_PROFILE_DIR = Path(__file__).resolve().parents[2] / "data" / "whatsapp_profile"


class WhatsAppAutomationError(Exception):
    """Expected WhatsApp Web setup or automation failure."""


class _WhatsAppWorker:
    """Own Playwright objects on one thread to satisfy Playwright thread affinity."""

    def __init__(self) -> None:
        self._requests: queue.Queue[tuple[Callable[[Any], Any], threading.Event, dict[str, Any]] | None] = queue.Queue()
        self._started = threading.Event()
        self._startup_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="karen-whatsapp", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=20):
            raise WhatsAppAutomationError("WhatsApp browser worker did not start in time.")
        if self._startup_error is not None:
            raise WhatsAppAutomationError(str(self._startup_error)) from self._startup_error

    def _run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[reportMissingImports]

            profile_dir = _PROFILE_DIR
            profile_dir.mkdir(parents=True, exist_ok=True)
            playwright = sync_playwright().start()
            context = None
            # Prefer the browser Windows already has (Edge, then Chrome) so no
            # separate Chromium download is needed; fall back to Playwright's own.
            for channel in ("msedge", "chrome", None):
                try:
                    context = playwright.chromium.launch_persistent_context(
                        str(profile_dir),
                        channel=channel,
                        headless=False,
                        no_viewport=True,
                        args=["--disable-notifications", "--start-maximized"],
                    )
                    break
                except Exception as error:  # pragma: no cover - browser dependent
                    LOGGER.info("WhatsApp browser %s unavailable: %s", channel or "chromium", error)
            if context is None:
                raise WhatsAppAutomationError(
                    "No browser is available for WhatsApp Web. Install Microsoft Edge or run: playwright install chromium"
                )
            self._started.set()
            while True:
                request = self._requests.get()
                if request is None:
                    # Close properly so WhatsApp's login (kept in browser storage) is saved.
                    try:
                        context.close()
                    finally:
                        playwright.stop()
                    return
                operation, finished, result = request
                try:
                    result["value"] = operation(context)
                except Exception as error:  # pragma: no cover - browser dependent
                    result["error"] = error
                finally:
                    finished.set()
        except Exception as error:  # pragma: no cover - browser/runtime dependent
            self._startup_error = error
            self._started.set()

    def execute(self, operation: Callable[[Any], Any], timeout: float = 120.0) -> Any:
        # The timeout must exceed the sum of the page waits inside an operation,
        # otherwise we could report failure while the message is still sent.
        finished = threading.Event()
        result: dict[str, Any] = {}
        self._requests.put((operation, finished, result))
        if not finished.wait(timeout=timeout):
            raise WhatsAppAutomationError(
                "WhatsApp Web is taking too long. Check WhatsApp before retrying, "
                "because the message may still be delivered."
            )
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def close(self, timeout: float = 15.0) -> None:
        self._requests.put(None)
        self._thread.join(timeout)


_worker: _WhatsAppWorker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> _WhatsAppWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _WhatsAppWorker()
        return _worker


def shutdown() -> None:
    """Close the WhatsApp browser cleanly (called when Karen exits)."""

    global _worker
    with _worker_lock:
        worker, _worker = _worker, None
    if worker is not None:
        try:
            worker.close()
        except Exception:  # pragma: no cover - best effort on exit
            LOGGER.debug("WhatsApp browser did not close cleanly", exc_info=True)


def _run(operation: Callable[[Any], Any]) -> Any:
    """Run on the WhatsApp browser, relaunching it once if the user closed its window."""

    global _worker
    try:
        return _get_worker().execute(operation)
    except Exception as error:
        if "closed" not in str(error).lower():
            raise
        with _worker_lock:
            _worker = None
        return _get_worker().execute(operation)


def _get_whatsapp_page(context: Any) -> Any:
    """Reuse the one WhatsApp tab (closing stray blank tabs) and show it."""

    pages = list(context.pages)
    page = next((candidate for candidate in pages if "web.whatsapp.com" in candidate.url), None)
    if page is None:
        page = next((candidate for candidate in pages if candidate.url in {"about:blank", "edge://newtab/", "chrome://newtab/"}), None)
        page = page or context.new_page()
        page.goto("https://web.whatsapp.com", wait_until="domcontentloaded")
    for other in pages:
        if other is not page and other.url in {"about:blank", "edge://newtab/", "chrome://newtab/"}:
            other.close()
    page.bring_to_front()
    # Raise the browser window itself over Karen's fullscreen interface.
    focus.bring_to_front(set(), ["WhatsApp"], grace=0)
    return page


def _open_page(context: Any) -> bool:
    """Show WhatsApp; return True when logged in, False when the QR code is showing."""

    page = _get_whatsapp_page(context)
    try:
        page.locator("#pane-side, canvas").first.wait_for(state="visible", timeout=20_000)
    except Exception:
        return True  # Still loading; nothing more to report.
    return page.locator("#pane-side").count() > 0


def open_whatsapp() -> SkillResult:
    """Open WhatsApp Web using one persistent local browser session."""

    try:
        logged_in = _run(_open_page)
    except WhatsAppAutomationError as error:
        return SkillResult(False, str(error))
    except Exception as error:  # pragma: no cover - browser dependent
        return SkillResult(False, f"I couldn't open WhatsApp Web: {error}")
    if logged_in:
        return SkillResult(True, "Opening WhatsApp.")
    return SkillResult(
        True,
        "WhatsApp needs a one-time login: on your phone open WhatsApp, Linked devices, and scan the code. "
        "I'll keep you logged in after that.",
    )


def _message_parameters(parameters: dict[str, Any]) -> tuple[str | None, str | None]:
    contact = parameters.get("contact")
    message = parameters.get("message")
    if not isinstance(contact, str) or not contact.strip():
        return None, None
    if not isinstance(message, str) or not message.strip():
        return None, None
    return contact.strip(), message


def _first_visible(page: Any, selectors: tuple[str, ...], timeout: int = 30_000) -> Any:
    """Return the first visible locator across WhatsApp Web DOM variants."""

    deadline = page.locator("body")
    deadline.wait_for(timeout=timeout)
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=2_000)
            return locator
        except Exception:
            continue
    raise WhatsAppAutomationError("WhatsApp Web did not expose the expected control.")


def _send_on_page(context: Any, contact: str, message: str) -> None:
    page = _get_whatsapp_page(context)
    page.wait_for_timeout(500)
    if page.get_by_text("Use WhatsApp on your computer", exact=False).count() > 0:
        raise WhatsAppAutomationError("WhatsApp is not logged in. Scan the QR code first.")

    search = _first_visible(
        page,
        (
            'div[contenteditable="true"][data-tab="3"]',
            'div[contenteditable="true"][aria-label*="Search"]',
            'input[placeholder*="Search"]',
            '[role="textbox"][aria-label*="Search"]',
        ),
    )
    search.fill(contact)
    page.wait_for_timeout(1_800)

    resolved = _resolve_contact(page, contact)
    page.locator(f'#pane-side span[title="{_css_string(resolved)}"]').first.click()

    # Safety: only type into the chat whose header shows the resolved contact.
    header = page.locator('#main header span[title], #main header span[dir="auto"]')
    opened = ""
    for index in range(min(header.count(), 10)):
        candidate = header.nth(index)
        if candidate.is_visible():
            opened = (candidate.get_attribute("title") or candidate.inner_text()).strip()
            if _normalize(opened) == _normalize(resolved):
                break
    if _normalize(opened) != _normalize(resolved):
        raise WhatsAppAutomationError(
            f"Safety check failed: WhatsApp opened {opened!r}, not {resolved!r}. No message was sent."
        )

    composer = _first_visible(
        page,
        (
            'footer div[contenteditable="true"]',
            'div[contenteditable="true"][aria-label*="Type a message"]',
            '[role="textbox"][aria-label*="message"]',
        ),
        timeout=15_000,
    )
    composer.fill(message)
    send_button = _first_visible(
        page,
        (
            'button[aria-label="Send"]',
            'button[data-testid="send"]',
            '[data-icon="send"]',
        ),
        timeout=10_000,
    )
    send_button.click()
    # Verify using the first line: multi-line messages render as separate text nodes.
    first_line = next((line.strip() for line in message.splitlines() if line.strip()), message)
    sent_message = page.get_by_text(first_line, exact=False).last
    try:
        sent_message.wait_for(state="visible", timeout=10_000)
    except Exception as error:
        raise WhatsAppAutomationError(
            "I clicked send but couldn't confirm the message appeared. Please check WhatsApp before resending."
        ) from error


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _css_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _resolve_contact(page: Any, contact: str) -> str:
    """Pick the one chat the spoken name refers to, or explain why not.

    "Shanju" matches a chat named "Shanju" exactly, or else a single chat whose
    name starts with those words ("Shanju K"). Several candidates are never guessed.
    """

    names: list[str] = []
    titles = page.locator("#pane-side span[title]")
    for index in range(min(titles.count(), 60)):
        title = titles.nth(index)
        if title.is_visible():
            name = (title.get_attribute("title") or "").strip()
            if name and name not in names:
                names.append(name)
    wanted = _normalize(contact)
    exact = [name for name in names if _normalize(name) == wanted]
    if len(exact) == 1:
        return exact[0]
    words = wanted.split()
    starts = [name for name in names if _normalize(name).split()[: len(words)] == words]
    if len(starts) == 1:
        return starts[0]
    if not exact and not starts:
        raise WhatsAppAutomationError(f"I couldn't find a WhatsApp chat named {contact}. No message was sent.")
    options = ", ".join((exact or starts)[:4])
    raise WhatsAppAutomationError(
        f"More than one chat matches {contact}: {options}. Say the full name. No message was sent."
    )


def send_message(intent_target: str | None, parameters: dict[str, Any]) -> SkillResult:
    """Send one exact message after the router has confirmed the intent."""

    contact, message = _message_parameters(parameters)
    if contact is None or message is None:
        return SkillResult(False, "I need both a contact and the exact message before opening WhatsApp.")
    if intent_target and intent_target.strip().lower() not in {"whatsapp", "whatsapp web"}:
        return SkillResult(False, "This action is only registered for WhatsApp.")

    try:
        _run(lambda context: _send_on_page(context, contact, message))
    except Exception as error:  # pragma: no cover - browser/runtime dependent
        return SkillResult(False, f"WhatsApp could not send the message: {error}")
    return SkillResult(True, f"Message sent to {contact} on WhatsApp.")
