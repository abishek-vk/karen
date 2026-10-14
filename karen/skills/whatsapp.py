"""Confirmation-gated WhatsApp Web automation on a dedicated browser thread."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable

from .base import SkillResult


_PROFILE_DIR = Path(__file__).resolve().parents[2] / "data" / "whatsapp_profile"


class WhatsAppAutomationError(Exception):
    """Expected WhatsApp Web setup or automation failure."""


class _WhatsAppWorker:
    """Own Playwright objects on one thread to satisfy Playwright thread affinity."""

    def __init__(self) -> None:
        self._requests: queue.Queue[tuple[Callable[[Any], Any], threading.Event, dict[str, Any]]] = queue.Queue()
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
            context = playwright.chromium.launch_persistent_context(
                str(profile_dir),
                headless=False,
                args=["--disable-notifications"],
            )
            self._started.set()
            while True:
                operation, finished, result = self._requests.get()
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


_worker: _WhatsAppWorker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> _WhatsAppWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _WhatsAppWorker()
        return _worker


def _get_whatsapp_page(context: Any) -> Any:
    """Return the existing WhatsApp tab, preferring it over blank tabs."""

    for page in context.pages:
        if "web.whatsapp.com" in page.url:
            page.bring_to_front()
            return page
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://web.whatsapp.com", wait_until="domcontentloaded")
    page.bring_to_front()
    return page


def _open_page(context: Any) -> None:
    _get_whatsapp_page(context)


def open_whatsapp() -> SkillResult:
    """Open WhatsApp Web using one persistent local browser session."""

    try:
        _get_worker().execute(_open_page)
    except WhatsAppAutomationError as error:
        return SkillResult(False, str(error))
    except Exception as error:  # pragma: no cover - browser dependent
        return SkillResult(False, f"I couldn't open WhatsApp Web: {error}")
    return SkillResult(
        True,
        "Opening WhatsApp Web. Log in once; Karen will reuse this local session.",
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
    page.wait_for_timeout(1_500)

    normalized_contact = " ".join(contact.strip().lower().split())
    rows = page.locator(
        '[role="listitem"], [data-testid="cell-frame-container"]'
    )
    matching_rows = []
    for index in range(min(rows.count(), 100)):
        candidate = rows.nth(index)
        if not candidate.is_visible():
            continue
        lines = [line.strip() for line in candidate.inner_text().splitlines() if line.strip()]
        if lines and " ".join(lines[0].lower().split()) == normalized_contact:
            matching_rows.append(candidate)
    if len(matching_rows) != 1:
        escaped_contact = contact.replace("\\", "\\\\").replace('"', '\\"')
        title_result = page.locator(f'[title="{escaped_contact}"]').first
        if title_result.is_visible():
            matching_rows.append(title_result)
    if len(matching_rows) != 1:
        raise WhatsAppAutomationError(
            f"I couldn't uniquely identify {contact}. No message was sent."
        )
    matching_rows[0].click()

    normalized_contact = " ".join(contact.strip().lower().split())
    header_candidates = page.locator(
        'header span[title], header span[dir="auto"], '
        '[data-testid="conversation-info-header-chat-title"]'
    )
    actual_contact = ""
    for index in range(min(header_candidates.count(), 20)):
        candidate = header_candidates.nth(index)
        if not candidate.is_visible():
            continue
        value = candidate.get_attribute("title") or candidate.inner_text()
        normalized_value = " ".join(value.strip().lower().split())
        if normalized_value in {"profile details", "search", "menu", "video call", "voice call"}:
            continue
        actual_contact = value
        if normalized_value == normalized_contact:
            break
    if " ".join(actual_contact.strip().lower().split()) != normalized_contact:
        raise WhatsAppAutomationError(
            f"Safety check failed: WhatsApp opened {actual_contact!r}, not {contact!r}. No message was sent."
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


def send_message(intent_target: str | None, parameters: dict[str, Any]) -> SkillResult:
    """Send one exact message after the router has confirmed the intent."""

    contact, message = _message_parameters(parameters)
    if contact is None or message is None:
        return SkillResult(False, "I need both a contact and the exact message before opening WhatsApp.")
    if intent_target and intent_target.strip().lower() not in {"whatsapp", "whatsapp web"}:
        return SkillResult(False, "This action is only registered for WhatsApp.")

    try:
        _get_worker().execute(lambda context: _send_on_page(context, contact, message))
    except Exception as error:  # pragma: no cover - browser/runtime dependent
        return SkillResult(False, f"WhatsApp could not send the message: {error}")
    return SkillResult(True, f"Message sent to {contact}.")
