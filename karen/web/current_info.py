"""Small, user-triggered web lookup for current information."""

from __future__ import annotations

import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from karen.core.intent import Intent
from karen.skills.base import SkillResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    snippet: str
    url: str


class CurrentInfoService:
    """Fetch concise public search results without sending user data to an LLM."""

    def __init__(self, *, enabled: bool = True, timeout_seconds: float = 8.0) -> None:
        self.enabled = enabled
        self.timeout_seconds = max(1.0, timeout_seconds)

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        if not self.enabled or not query.strip():
            return []
        authoritative = self._search_current_office_holder(query)
        if authoritative:
            return authoritative[:limit]
        url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query.strip())
        request = Request(
            url,
            headers={"User-Agent": "KarenAI/1.0 (local assistant current-information lookup)"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                document = response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, OSError, TimeoutError) as error:
            LOGGER.warning("Live information lookup failed: %s", error)
            document = ""
        results = self._parse_results(document, limit)
        if results:
            return results
        return self._search_bing(query, limit)

    def _search_current_office_holder(self, query: str) -> list[SearchResult]:
        match = re.search(
            r"(?:current|present|latest|now).*?(?:cm|chief minister) of ([a-z][a-z ]+)",
            query.lower(),
        )
        if not match:
            return []
        state = " ".join(match.group(1).split()).strip(" ?.!\")")
        page = "Chief_Minister_of_" + "_".join(word.title() for word in state.split())
        url = (
            "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&page="
            + quote_plus(page)
        )
        request = Request(url, headers={"User-Agent": "KarenAI/1.0 current-information"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            wikitext = payload["parse"]["wikitext"]["*"]
            incumbent = re.search(r"incumbent\s*=\s*\[\[([^\]]+)\]\]", wikitext, re.IGNORECASE)
        except (HTTPError, URLError, OSError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as error:
            LOGGER.warning("Authoritative current-information lookup failed: %s", error)
            return []
        if incumbent is None:
            return []
        person = incumbent.group(1).strip()
        return [
            SearchResult(
                f"Current Chief Minister of {state.title()}",
                f"{person} is listed as the incumbent.",
                f"https://en.wikipedia.org/wiki/{page}",
            )
        ]

    def _search_bing(self, query: str, limit: int) -> list[SearchResult]:
        url = "https://www.bing.com/search?format=rss&q=" + quote_plus(query.strip())
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 KarenAI/1.0"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                document = response.read().decode("utf-8", errors="replace")
            root = ET.fromstring(document)
        except (HTTPError, URLError, OSError, TimeoutError, ET.ParseError) as error:
            LOGGER.warning("Fallback live information lookup failed: %s", error)
            return []
        results: list[SearchResult] = []
        for item in root.findall("./channel/item")[:limit]:
            title = item.findtext("title", default="").strip()
            snippet = item.findtext("description", default="").strip()
            url = item.findtext("link", default="").strip()
            if title or snippet:
                results.append(SearchResult(title, _clean_html(snippet), url))
        return results

    @staticmethod
    def _parse_results(document: str, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)',
            document,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for url, title, remainder in blocks[:limit]:
            snippet_match = re.search(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
                remainder,
                flags=re.IGNORECASE | re.DOTALL,
            )
            results.append(
                SearchResult(
                    title=_clean_html(title),
                    snippet=_clean_html(snippet_match.group(1) if snippet_match else ""),
                    url=html.unescape(url),
                )
            )
        return [result for result in results if result.title or result.snippet]

    def handle_intent(self, intent: Intent) -> SkillResult:
        query = intent.parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return SkillResult(False, "What current information would you like me to check?")
        results = self.search(query)
        if not results:
            return SkillResult(False, "Live information is unavailable right now.")
        result = results[0]
        detail = f" {result.snippet}" if result.snippet else ""
        return SkillResult(
            True,
            f"{result.title}.{detail}\nSource: {result.url}",
        )


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()
