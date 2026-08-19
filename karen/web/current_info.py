"""Small, user-triggered web lookup for current information."""

from __future__ import annotations

import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlparse
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
        cricket = self._search_current_cricket_captain(query)
        if cricket:
            return cricket[:limit]
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
        results = self._parse_results(document, 8)
        results.extend(self._search_bing(query, 8))
        return self._rank_results(query, results, limit)

    def _search_current_cricket_captain(self, query: str) -> list[SearchResult]:
        lowered = query.lower()
        if not (
            "captain" in lowered
            and ("cricket" in lowered or "team" in lowered)
            and ("india" in lowered or "indian" in lowered)
        ):
            return []
        page = "India_national_cricket_team"
        url = (
            "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&page="
            + quote_plus(page)
        )
        request = Request(url, headers={"User-Agent": "KarenAI/1.0 current-information"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            wikitext = payload["parse"]["wikitext"]["*"]
        except (HTTPError, URLError, OSError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as error:
            LOGGER.warning("Cricket captain lookup failed: %s", error)
            return []
        captains: list[str] = []
        for format_name in ("test", "od", "t20i"):
            match = re.search(
                rf"{format_name}_captain\s*=\s*\[\[([^\]]+)\]\]",
                wikitext,
                re.IGNORECASE,
            )
            if match:
                label = "ODI" if format_name == "od" else format_name.upper()
                captains.append(f"{label}: {match.group(1).strip()}")
        if not captains:
            return []
        return [
            SearchResult(
                "Current India cricket team captains",
                "; ".join(captains) + ".",
                "https://en.wikipedia.org/wiki/India_national_cricket_team",
            )
        ]

    def _search_current_office_holder(self, query: str) -> list[SearchResult]:
        lowered = query.lower()
        match = re.search(
            r"(?:current|present|latest|now).*?(?:cm|chief minister) of ([a-z][a-z ]+)",
            lowered,
        )
        possessive_match = re.search(
            r"([a-z][a-z ]+?)(?:'s|s')\s+(?:current|present|latest|now).*?(?:cm|chief minister)",
            lowered,
        )
        if match:
            state = " ".join(match.group(1).split()).strip(" ?.!\"")
        elif possessive_match:
            state = " ".join(possessive_match.group(1).split()).strip(" ?.!\"")
        else:
            return []
        state = re.sub(r"^(?:who is|what is|tell me)\s+", "", state).strip()
        state = {"tamilnadu": "Tamil Nadu", "tamil nad": "Tamil Nadu"}.get(state, state)
        page = "Chief_Minister_of_" + "_".join(word.title() for word in state.split())
        urls = (
            "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&page="
            + quote_plus(page),
            "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&formatversion=2&page="
            + quote_plus(page),
        )
        wikitext = ""
        for url in urls:
            request = Request(url, headers={"User-Agent": "KarenAI/1.0 current-information"})
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="replace"))
                value = payload.get("parse", {}).get("wikitext", "")
                wikitext = value.get("*") if isinstance(value, dict) else value
                if isinstance(wikitext, str) and wikitext:
                    break
            except (HTTPError, URLError, OSError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as error:
                LOGGER.warning("Authoritative current-information lookup failed: %s", error)
        if not isinstance(wikitext, str) or not wikitext:
            return []
        incumbent = re.search(r"incumbent\s*=\s*\[\[([^\]]+)\]\]", wikitext, re.IGNORECASE)
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

    @staticmethod
    def _rank_results(query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]
        trusted_weights = {
            ".gov": 5,
            "gov.in": 5,
            "wikipedia.org": 4,
            "bbc.com": 4,
            "thehindu.com": 4,
            "reuters.com": 4,
        }
        unique: dict[str, SearchResult] = {}
        for result in results:
            if result.url and result.url not in unique:
                unique[result.url] = result
        scored: list[tuple[int, SearchResult]] = []
        for result in unique.values():
            searchable = f"{result.title} {result.snippet}".lower()
            if "captain" in query.lower() and (
                "captain" not in searchable
                or not any(term in searchable for term in ("india", "indian", "cricket"))
            ):
                continue
            score = sum(2 for term in terms if term in result.title.lower())
            score += sum(1 for term in terms if term in searchable)
            host = urlparse(result.url).netloc.lower()
            score += max((weight for domain, weight in trusted_weights.items() if domain in host), default=0)
            if score:
                scored.append((score, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _score, result in scored[:limit]]

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
