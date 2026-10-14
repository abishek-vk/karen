"""Open well-known websites (or a YouTube search) in the default browser."""

from __future__ import annotations

import re
import webbrowser
from urllib.parse import quote_plus

from .base import SkillResult

WEBSITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "google drive": "https://drive.google.com",
    "drive": "https://drive.google.com",
    "google maps": "https://maps.google.com",
    "maps": "https://maps.google.com",
    "google calendar": "https://calendar.google.com",
    "calendar": "https://calendar.google.com",
    "github": "https://github.com",
    "stack overflow": "https://stackoverflow.com",
    "stackoverflow": "https://stackoverflow.com",
    "wikipedia": "https://en.wikipedia.org",
    "linkedin": "https://www.linkedin.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "reddit": "https://www.reddit.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "spotify web": "https://open.spotify.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "outlook": "https://outlook.live.com",
    "news": "https://news.google.com",
    "google news": "https://news.google.com",
}
_SEARCH_URLS = {
    "youtube": "https://www.youtube.com/results?search_query={}",
    "google": "https://www.google.com/search?q={}",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search={}",
    "amazon": "https://www.amazon.in/s?k={}",
    "github": "https://github.com/search?q={}",
}
_FILE_SUFFIXES = (".exe", ".lnk", ".bat", ".cmd", ".msi", ".url", ".txt", ".py", ".ps1", ".pdf", ".docx")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,})(/\S*)?$")


def resolve_website(name: str | None) -> str | None:
    """Return a known site key or a bare domain, else ``None``."""

    if not name:
        return None
    key = " ".join(name.strip().lower().split()).replace(" dot ", ".")
    if key in WEBSITES:
        return key
    if _DOMAIN.fullmatch(key) and not key.endswith(_FILE_SUFFIXES):
        return key
    return None


def open_website(target: str | None, parameters: dict | None = None) -> SkillResult:
    site = resolve_website(target)
    if site is None:
        return SkillResult(False, "I don't know that website. Try saying its address, like github dot com.")
    query = (parameters or {}).get("query")
    if isinstance(query, str) and query.strip() and site in _SEARCH_URLS:
        url = _SEARCH_URLS[site].format(quote_plus(query.strip()))
        message = f"Searching {site.title()} for {query.strip()}."
    elif site in WEBSITES:
        url = WEBSITES[site]
        message = f"Opening {site.title()}."
    else:
        url = site if site.startswith("http") else f"https://{site}"
        message = f"Opening {site}."
    try:
        if not webbrowser.open(url, new=2):
            return SkillResult(False, "I couldn't open your web browser.")
    except webbrowser.Error:
        return SkillResult(False, "I couldn't open your web browser.")
    return SkillResult(True, message)
