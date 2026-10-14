"""User-triggered web research for current or uncertain information.

Pipeline: search (ddgs library -> DuckDuckGo HTML -> Bing RSS) -> fetch the top
pages in parallel -> keep only the passages relevant to the question. The local
model then answers from those passages. Everything is bounded (pages, bytes,
characters, time) so a lookup stays light on a CPU-only laptop.
"""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, quote_plus, urlparse
from urllib.request import Request, urlopen

from karen.core.intent import Intent
from karen.core.offices import parse_office_question
from karen.skills.base import SkillResult

LOGGER = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_MAX_PAGE_BYTES = 600_000
# Every character here costs prompt time on a CPU (~25 tokens/s for a 7B model),
# so keep only the most relevant passages.
_MAX_EXCERPT_CHARS = 450
_PAGE_TIMEOUT_SECONDS = 4.0
_CACHE_TTL_SECONDS = 600.0
_SKIP_HOSTS = ("youtube.com", "youtu.be", "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com")
_STOP = frozenset(
    "the a an of in on at to for is are was were be what who whom which when where how why "
    "does did do can could would should will current currently latest today now right "
    "tell me please about and or with from by as it this that".split()
)


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    snippet: str
    url: str


@dataclass(frozen=True, slots=True)
class Source:
    title: str
    url: str
    excerpt: str

    @property
    def site(self) -> str:
        host = urlparse(self.url).hostname or ""
        return host.removeprefix("www.") or self.title


@dataclass(slots=True)
class Research:
    query: str
    sources: list[Source] = field(default_factory=list)


class CurrentInfoService:
    """Fetch and condense public web information for one question."""

    def __init__(self, *, enabled: bool = True, timeout_seconds: float = 8.0, max_pages: int = 3) -> None:
        self.enabled = enabled
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.max_pages = max(0, min(max_pages, 5))
        self._cache: dict[str, tuple[float, Research]] = {}
        self._cache_lock = threading.Lock()

    # -- search -----------------------------------------------------------

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        if not self.enabled or not query.strip():
            return []
        authoritative = self._search_current_office_holder(query)
        if authoritative:
            return authoritative[:limit]
        for provider in (self._search_ddgs, self._search_duckduckgo_html, self._search_bing):
            results = provider(query, limit)
            if results:
                return results
        return []

    def _search_ddgs(self, query: str, limit: int) -> list[SearchResult]:
        try:
            try:
                from ddgs import DDGS  # type: ignore[import-not-found]
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError:
            return []
        try:
            rows = DDGS(timeout=int(self.timeout_seconds)).text(query.strip(), max_results=limit)
        except Exception as error:  # pragma: no cover - network/library dependent
            LOGGER.warning("ddgs search failed: %s", error)
            return []
        return [
            SearchResult(_clean_html(row.get("title", "")), _clean_html(row.get("body", "")), row.get("href", ""))
            for row in rows or []
            if isinstance(row, dict) and (row.get("title") or row.get("body"))
        ]

    def _search_duckduckgo_html(self, query: str, limit: int) -> list[SearchResult]:
        url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query.strip())
        document = self._get_text(url, user_agent=_BROWSER_UA)
        return self._parse_results(document, limit) if document else []

    def _search_bing(self, query: str, limit: int) -> list[SearchResult]:
        url = "https://www.bing.com/search?format=rss&q=" + quote_plus(query.strip())
        document = self._get_text(url, user_agent=_BROWSER_UA)
        if not document:
            return []
        try:
            root = ET.fromstring(document)
        except ET.ParseError as error:
            LOGGER.warning("Fallback live information lookup failed: %s", error)
            return []
        results: list[SearchResult] = []
        for item in root.findall("./channel/item")[:limit]:
            title = item.findtext("title", default="").strip()
            snippet = item.findtext("description", default="").strip()
            link = item.findtext("link", default="").strip()
            if title or snippet:
                results.append(SearchResult(title, _clean_html(snippet), link))
        return results

    def _search_current_office_holder(self, query: str) -> list[SearchResult]:
        """Answer "who is the <role> of <place>" from the Wikipedia infobox ``incumbent``."""

        parsed = parse_office_question(query)
        if parsed is None:
            return []
        role, place = parsed
        title = f"{role} of {place}"
        url = (
            "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json"
            "&formatversion=2&redirects=1&page=" + quote_plus(title.replace(" ", "_"))
        )
        document = self._get_text(url)
        try:
            payload = json.loads(document) if document else {}
        except json.JSONDecodeError:
            return []
        parse = payload.get("parse", {}) if isinstance(payload, dict) else {}
        wikitext = parse.get("wikitext", "") if isinstance(parse, dict) else ""
        if isinstance(wikitext, dict):
            wikitext = wikitext.get("*", "")
        if not isinstance(wikitext, str) or not wikitext:
            return []
        incumbent = re.search(r"\|\s*incumbent\s*=\s*(?:\{\{[^}]*\}\}\s*)?\[\[([^\]]+)\]\]", wikitext, re.IGNORECASE)
        if incumbent is None:
            return []
        person = incumbent.group(1).split("|")[0].strip()
        since = re.search(r"\|\s*incumbentsince\s*=\s*([^\n|]+)", wikitext, re.IGNORECASE)
        since_text = ""
        if since:
            cleaned = re.sub(r"\{\{[^|}]*\|?|\}\}|\[\[|\]\]", " ", since.group(1))
            cleaned = re.sub(r"\b(?:df|mf)=y(?:es)?\b|\|", " ", cleaned)
            cleaned = " ".join(cleaned.split())
            if cleaned:
                since_text = f" (in office since {cleaned})"
        page = parse.get("title", title) if isinstance(parse, dict) else title
        return [
            SearchResult(
                f"Current {title}",
                f"According to Wikipedia, the current {title} is {person}{since_text}.",
                "https://en.wikipedia.org/wiki/" + quote(str(page).replace(" ", "_")),
            )
        ]

    # -- research ---------------------------------------------------------

    def research(self, query: str) -> Research:
        """Search, read the top pages, and return relevant excerpts with sources."""

        query = query.strip()
        if not self.enabled or not query:
            return Research(query)
        key = query.lower()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
                return cached[1]

        research = Research(query)
        weather = self._weather(query)
        if weather is not None:
            research.sources.append(weather)
        else:
            results = self.search(query, limit=max(3, self.max_pages + 2))
            research.sources.extend(self._read_sources(query, results))

        with self._cache_lock:
            if len(self._cache) > 32:
                self._cache.pop(next(iter(self._cache)))
            if research.sources:
                self._cache[key] = (time.monotonic(), research)
        return research

    def _read_sources(self, query: str, results: list[SearchResult]) -> list[Source]:
        terms = _query_terms(query)
        readable = [
            result for result in results
            if result.url.startswith("http") and not any(host in result.url for host in _SKIP_HOSTS)
        ][: self.max_pages]

        def read(result: SearchResult) -> Source:
            excerpt = ""
            if "wikipedia.org/wiki/Chief_Minister" not in result.url:
                page = self._get_text(
                    result.url, user_agent=_BROWSER_UA, html_only=True,
                    timeout=min(self.timeout_seconds, _PAGE_TIMEOUT_SECONDS),
                )
                if page:
                    excerpt = relevant_passages(extract_text(page), terms, _MAX_EXCERPT_CHARS)
            combined = result.snippet if not excerpt else f"{result.snippet}\n{excerpt}" if result.snippet else excerpt
            return Source(result.title or result.url, result.url, combined[: _MAX_EXCERPT_CHARS + 200])

        sources: list[Source] = []
        if readable:
            with ThreadPoolExecutor(max_workers=len(readable)) as pool:
                sources = [source for source in pool.map(read, readable) if source.excerpt]
        # Results we didn't fetch still contribute their search snippets.
        for result in results:
            if result not in readable and result.snippet and len(sources) < self.max_pages:
                sources.append(Source(result.title or result.url, result.url, result.snippet))
        return sources

    def _weather(self, query: str) -> Source | None:
        lowered = query.lower()
        if not re.search(r"\b(?:weather|temperature|forecast|raining|rain|humid|humidity|sunny|cold|hot outside)\b", lowered):
            return None
        place_match = re.search(
            r"\b(?:in|at|for|of)\s+([a-z][a-z .'-]+?)(?:\s+(?:today|tomorrow|now|right now|this week|tonight))?\s*\??$",
            lowered,
        )
        place = place_match.group(1).strip() if place_match else ""
        if place in {"the", "my area", "my city", "here", "my location"}:
            place = ""
        document = self._get_text(f"https://wttr.in/{quote(place)}?format=j1", user_agent="curl/8.0")
        try:
            data = json.loads(document) if document else None
            current = data["current_condition"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return None
        area = data.get("nearest_area", [{}])[0]
        area_name = (area.get("areaName") or [{}])[0].get("value", place.title() or "your area")
        description = (current.get("weatherDesc") or [{}])[0].get("value", "").strip()
        lines = [
            f"Current weather in {area_name}: {description}, {current.get('temp_C')}°C "
            f"(feels like {current.get('FeelsLikeC')}°C), humidity {current.get('humidity')}%, "
            f"wind {current.get('windspeedKmph')} km/h."
        ]
        for label, day in zip(("Today", "Tomorrow", "Day after tomorrow"), data.get("weather", [])[:3]):
            hourly = day.get("hourly") or []
            rain = max((int(hour.get("chanceofrain", 0) or 0) for hour in hourly), default=0)
            lines.append(
                f"{label} ({day.get('date')}): high {day.get('maxtempC')}°C, low {day.get('mintempC')}°C, "
                f"chance of rain up to {rain}%."
            )
        return Source(f"Weather for {area_name}", "https://wttr.in/" + quote(place), "\n".join(lines))

    # -- fallbacks without a model ---------------------------------------

    def summarize(self, research: Research) -> str:
        """Extractive answer used when the local model is unavailable."""

        if not research.sources:
            return "I couldn't find anything reliable online about that right now."
        best = research.sources[0]
        sentences = re.split(r"(?<=[.!?])\s+", best.excerpt.replace("\n", " "))
        text = " ".join(sentences[:3]).strip()
        return f"{text}\nSource: {best.site}"

    def handle_intent(self, intent: Intent) -> SkillResult:
        query = intent.parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return SkillResult(False, "What would you like me to look up?")
        results = self.search(query)
        if not results:
            return SkillResult(False, "I couldn't reach the web right now.")
        result = results[0]
        detail = f" {result.snippet}" if result.snippet else ""
        return SkillResult(True, f"{result.title}.{detail}\nSource: {result.url}")

    # -- HTTP -------------------------------------------------------------

    def _get_text(
        self,
        url: str,
        *,
        user_agent: str = "KarenAI/1.0 (local assistant)",
        html_only: bool = False,
        timeout: float | None = None,
    ) -> str:
        request = Request(url, headers={"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.8"})
        try:
            with urlopen(request, timeout=timeout or self.timeout_seconds) as response:
                if html_only:
                    content_type = response.headers.get("Content-Type", "") if hasattr(response, "headers") else ""
                    if content_type and "html" not in content_type:
                        return ""
                return response.read(_MAX_PAGE_BYTES).decode("utf-8", errors="replace")
        except (HTTPError, URLError, OSError, TimeoutError, ValueError) as error:
            LOGGER.info("Web request failed for %s: %s", urlparse(url).hostname, error)
            return ""

    @staticmethod
    def _parse_results(document: str, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)',
            document,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for url, title, remainder in blocks:
            if len(results) >= limit:
                break
            target = _resolve_redirect(html.unescape(url))
            if "duckduckgo.com/y.js" in target:  # sponsored result
                continue
            snippet_match = re.search(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
                remainder,
                flags=re.IGNORECASE | re.DOTALL,
            )
            results.append(
                SearchResult(
                    title=_clean_html(title),
                    snippet=_clean_html(snippet_match.group(1) if snippet_match else ""),
                    url=target,
                )
            )
        return [result for result in results if result.title or result.snippet]


def _resolve_redirect(url: str) -> str:
    """Turn DuckDuckGo ``/l/?uddg=`` redirect links into the real target URL."""

    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return url


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _query_terms(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9]{2,}", query.lower())
    return [word for word in dict.fromkeys(words) if word not in _STOP]


class _TextExtractor(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "button", "select", "iframe"})
    _BLOCK = frozenset({"p", "div", "li", "h1", "h2", "h3", "h4", "td", "th", "tr", "section", "article", "br", "dd", "dt", "blockquote"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001 - HTMLParser signature
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def extract_text(document: str) -> str:
    """Return readable page text with one paragraph per line."""

    parser = _TextExtractor()
    try:
        parser.feed(document)
        parser.close()
    except Exception:  # pragma: no cover - malformed HTML
        pass
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if len(line) > 40)


def relevant_passages(text: str, terms: list[str], max_chars: int) -> str:
    """Pick the paragraphs that mention the most query terms, in page order."""

    paragraphs = [paragraph for paragraph in text.split("\n") if paragraph]
    if not paragraphs:
        return ""
    if not terms:
        return " ".join(paragraphs)[:max_chars]
    scored = []
    for index, paragraph in enumerate(paragraphs[:400]):
        lowered = paragraph.lower()
        hits = sum(1 for term in terms if term in lowered)
        if hits:
            # Prefer dense, mid-length paragraphs that contain numbers/dates.
            score = hits / len(terms) + (0.15 if re.search(r"\d", paragraph) else 0) - (0.2 if len(paragraph) > 900 else 0)
            scored.append((score, index, paragraph))
    if not scored:
        return ""
    chosen: list[tuple[int, str]] = []
    total = 0
    for _score, index, paragraph in sorted(scored, key=lambda item: (-item[0], item[1])):
        snippet = paragraph[:600]
        if total + len(snippet) > max_chars:
            continue
        chosen.append((index, snippet))
        total += len(snippet)
        if total >= max_chars * 0.8:
            break
    return "\n".join(paragraph for _index, paragraph in sorted(chosen))
