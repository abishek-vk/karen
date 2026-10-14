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
import math
import re
import threading
import time
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
        authoritative = self._search_current_office_holder(query) or self._search_software_version(query)
        if authoritative:
            return authoritative[:limit]
        terms = _query_terms(query)
        query = _with_current_year(query)
        for provider in (self._search_ddgs, self._search_duckduckgo_html, self._search_duckduckgo_lite, self._search_wikipedia):
            # Ask for a few extra results so off-topic ones can be dropped and
            # trusted sites ranked first.
            results = _rank_by_trust(_on_topic(provider(query, limit + 3), terms))
            if results:
                return results[:limit]
        return []

    def _search_ddgs(self, query: str, limit: int) -> list[SearchResult]:
        try:
            try:
                from ddgs import DDGS  # type: ignore[import-not-found]
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError:
            return []
        rows = None
        # "auto" sometimes picks engines that are blocked from this network, so
        # try the reliable ones in order and stop at the first with results.
        for backend in ("duckduckgo", "yahoo", "bing"):
            try:
                rows = DDGS(timeout=int(self.timeout_seconds)).text(query.strip(), max_results=limit, backend=backend)
            except Exception as error:  # pragma: no cover - network/library dependent
                LOGGER.info("ddgs %s search failed: %s", backend, error)
                continue
            if rows:
                break
        return [
            SearchResult(_clean_html(row.get("title", "")), _clean_html(row.get("body", "")), row.get("href", ""))
            for row in rows or []
            if isinstance(row, dict) and (row.get("title") or row.get("body"))
        ]

    def _search_duckduckgo_html(self, query: str, limit: int) -> list[SearchResult]:
        url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query.strip())
        document = self._get_text(url, user_agent=_BROWSER_UA)
        return self._parse_results(document, limit) if document else []

    def _search_duckduckgo_lite(self, query: str, limit: int) -> list[SearchResult]:
        url = "https://lite.duckduckgo.com/lite/?q=" + quote_plus(query.strip())
        document = self._get_text(url, user_agent=_BROWSER_UA)
        if not document:
            return []
        links = re.findall(
            r"<a[^>]+href=\"([^\"]+)\"[^>]*class=['\"]result-link['\"][^>]*>(.*?)</a>", document, re.IGNORECASE | re.DOTALL
        )
        snippets = re.findall(r"class=['\"]result-snippet['\"][^>]*>(.*?)</td>", document, re.IGNORECASE | re.DOTALL)
        results = []
        for index, (link, title) in enumerate(links):
            target = _resolve_redirect(html.unescape(link))
            if (urlparse(target).hostname or "").endswith("duckduckgo.com"):  # ads and help pages
                continue
            snippet = _clean_html(snippets[index]) if index < len(snippets) else ""
            results.append(SearchResult(_clean_html(title), snippet, target))
        return results[:limit]

    def _search_wikipedia(self, query: str, limit: int) -> list[SearchResult]:
        """Last resort: Wikipedia's own search, which is reliable but not always current."""

        url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1"
            f"&srlimit={max(1, limit)}&srsearch=" + quote_plus(query.strip())
        )
        document = self._get_text(url)
        try:
            rows = json.loads(document).get("query", {}).get("search", []) if document else []
        except (json.JSONDecodeError, AttributeError):
            return []
        return [
            SearchResult(
                row["title"], _clean_html(row.get("snippet", "")),
                "https://en.wikipedia.org/wiki/" + quote(row["title"].replace(" ", "_")),
            )
            for row in rows
            if isinstance(row, dict) and row.get("title")
        ]

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

    def _search_software_version(self, query: str) -> list[SearchResult]:
        """Answer "latest version of X" from release data instead of web pages, which
        are often months out of date: endoflife.date first, then the project's GitHub releases."""

        product = parse_version_question(query)
        if product is None:
            return []
        return self._version_from_endoflife(product) or self._version_from_github(product)

    def _version_from_endoflife(self, product: str) -> list[SearchResult]:
        slug = _ENDOFLIFE_SLUGS.get(product, re.sub(r"[^a-z0-9]+", "-", product).strip("-"))
        document = self._get_text(f"https://endoflife.date/api/{quote(slug)}.json")
        try:
            cycles = json.loads(document) if document else []
        except json.JSONDecodeError:
            return []
        if not isinstance(cycles, list) or not cycles or not isinstance(cycles[0], dict) or not cycles[0].get("latest"):
            return []
        name = _display_name(product)
        newest = cycles[0]
        text = f"According to endoflife.date, the latest {name} release is {newest['latest']}"
        if newest.get("latestReleaseDate"):
            text += f", released on {newest['latestReleaseDate']}"
        text += "."
        today = time.strftime("%Y-%m-%d")
        lts = next(
            (
                cycle for cycle in cycles
                if isinstance(cycle, dict) and isinstance(cycle.get("lts"), (str, bool)) and cycle.get("lts")
                and (cycle["lts"] is True or str(cycle["lts"]) <= today)
            ),
            None,
        )
        if lts is not None and lts is not newest:
            text += f" The newest long-term support (LTS) line is {name} {lts.get('cycle')}, currently {lts.get('latest')}."
        return [SearchResult(f"Latest {name} version", text, f"https://endoflife.date/{slug}")]

    def _version_from_github(self, product: str) -> list[SearchResult]:
        repo_name = re.sub(r"[^a-z0-9]+", "-", product).strip("-")
        document = self._get_text(
            "https://api.github.com/search/repositories?sort=stars&per_page=5&q=" + quote_plus(f"{product} in:name")
        )
        try:
            items = json.loads(document).get("items", []) if document else []
        except (json.JSONDecodeError, AttributeError):
            return []
        repo = next(
            (
                item for item in items
                if isinstance(item, dict) and str(item.get("name", "")).lower() == repo_name
                and int(item.get("stargazers_count") or 0) >= 1000
            ),
            None,
        )
        if repo is None:
            return []
        document = self._get_text(f"https://api.github.com/repos/{repo['full_name']}/releases/latest")
        try:
            release = json.loads(document) if document else {}
        except json.JSONDecodeError:
            return []
        tag = release.get("tag_name") if isinstance(release, dict) else None
        if not tag:
            return []
        published = str(release.get("published_at") or "")[:10]
        name = _display_name(product)
        text = f"According to the official {repo['full_name']} GitHub releases page, the latest {name} release is {tag}"
        text += f", published on {published}." if published else "."
        return [SearchResult(f"Latest {name} release", text, str(release.get("html_url") or repo.get("html_url", "")))]

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


_RECENCY_WORDS = re.compile(r"\b(?:last|latest|current|currently|recent|newest|now|today|this year)\b", re.IGNORECASE)


_VERSION_QUESTION = re.compile(
    r"(?:latest|newest|current|most recent)\s+(?:stable\s+|lts\s+)?(?:version|release)\s+of\s+(?:the\s+)?(?P<a>.+)"
    r"|(?:latest|newest|current|most recent)\s+(?:stable\s+)?(?P<b>[a-z0-9.+# -]+?)\s+(?:version|release)\b"
    r"|(?:what|which)\s+version\s+of\s+(?P<c>.+?)\s+is\s+(?:the\s+)?(?:latest|newest|current)",
    re.IGNORECASE,
)
_NOT_SOFTWARE = frozenset({"iphone", "android phone", "samsung", "galaxy", "pixel", "movie", "song", "film", "car", "the"})
# Spoken names -> endoflife.date product ids.
_ENDOFLIFE_SLUGS = {
    "node": "nodejs", "node.js": "nodejs", "node js": "nodejs", "nodejs": "nodejs", "java": "oracle-jdk",
    "jdk": "oracle-jdk", ".net": "dotnet", "dotnet": "dotnet", "net": "dotnet", "golang": "go", "go": "go",
    "postgres": "postgresql", "vue": "vue", "vue.js": "vue", "angular": "angular", "django": "django",
    "react": "react", "react.js": "react", "linux kernel": "linux", "linux": "linux", "next.js": "nextjs", "nextjs": "nextjs", "c#": "dotnet",
}
_DISPLAY_NAMES = {
    "node": "Node.js", "node.js": "Node.js", "node js": "Node.js", "nodejs": "Node.js", "java": "Java",
    "jdk": "Java", ".net": ".NET", "dotnet": ".NET", "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
    "next.js": "Next.js", "nextjs": "Next.js", "vue.js": "Vue.js", "react.js": "React", "php": "PHP",
    "typescript": "TypeScript", "golang": "Go", "pytorch": "PyTorch", "numpy": "NumPy", "vscode": "VS Code",
    "vs code": "VS Code",
}


def parse_version_question(text: str) -> str | None:
    """Return the product in questions like "what's the latest version of Python"."""

    cleaned = re.sub(r"\b(?:19|20)\d{2}\b|[?!]", " ", text.lower())
    match = _VERSION_QUESTION.search(" ".join(cleaned.split()))
    if match is None:
        return None
    product = next(group for group in match.groups() if group)
    product = re.sub(r"\s+(?:now|today|right now|currently|available|out|released)$", "", product.strip(" .,'\""))
    product = re.sub(r"^(?:the|a|an)\s+", "", product).strip()
    if not product or len(product.split()) > 3 or product in _NOT_SOFTWARE:
        return None
    return product


def _display_name(product: str) -> str:
    return _DISPLAY_NAMES.get(product, product if any(char.isupper() for char in product) else product.title())


def _with_current_year(query: str) -> str:
    """Bias "latest/last/current" searches toward this year's pages."""

    if _RECENCY_WORDS.search(query) and not re.search(r"\b(?:19|20)\d{2}\b", query):
        return f"{query} {time.localtime().tm_year}"
    return query


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _on_topic(results: list[SearchResult], terms: list[str]) -> list[SearchResult]:
    """Drop results that share no meaningful word with the question (search engines
    sometimes return unrelated filler, which would otherwise be read as fact)."""

    if not terms:
        return results
    needed = 2 if len(terms) >= 3 else 1
    kept = []
    for result in results:
        haystack = f"{result.title} {result.snippet} {result.url}".lower()
        if sum(1 for term in terms if term in haystack) >= needed:
            kept.append(result)
    return kept


# Primary/official sources and established references; content farms rank last.
_TRUSTED_HOSTS = (
    "wikipedia.org", "endoflife.date", "britannica.com", "python.org", "nodejs.org", "npmjs.com", "pypi.org", "github.com",
    "developer.mozilla.org", "learn.microsoft.com", "microsoft.com", "apple.com", "google.com", "android.com",
    "react.dev", "kotlinlang.org", "go.dev", "rust-lang.org", "oracle.com", "openjdk.org", "docs.",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com", "indianexpress.com", "ndtv.com",
    "timesofindia.indiatimes.com", "hindustantimes.com", "espncricinfo.com", "espn.com", "fifa.com", "iplt20.com",
    "who.int", "cdc.gov", "nih.gov", "mayoclinic.org", "nasa.gov", "rbi.org.in", "imd.gov.in", "un.org",
    "worldbank.org", "imf.org", "nature.com", "sciencedirect.com", "theverge.com", "techcrunch.com", "forbes.com",
    "bloomberg.com", "cnbc.com", "coindesk.com", "coinmarketcap.com", "xe.com",
)
_LOW_QUALITY_HOSTS = ("quora.com", "pinterest.", "answers.com", "brainly.", "chegg.com", "scribd.com", "medium.com")


def _trust(url: str) -> int:
    host = (urlparse(url).hostname or "").lower()
    if any(part in host for part in _LOW_QUALITY_HOSTS):
        return -1
    if host.endswith((".gov", ".gov.in", ".gov.uk", ".edu", ".ac.in", ".ac.uk", ".nic.in", ".int")):
        return 2
    return 1 if any(part in host for part in _TRUSTED_HOSTS) else 0


def _rank_by_trust(results: list[SearchResult]) -> list[SearchResult]:
    """Stable sort: trusted sources first, otherwise keep the search engine's order."""

    return sorted(results, key=lambda result: -_trust(result.url))


class _TextExtractor(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "button", "select", "iframe"})
    _BLOCK = frozenset({"p", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article", "br", "dd", "dt", "blockquote", "caption"})
    # Table cells stay on their row's line so rows like "2026 | Spain | 1-0 | Argentina" survive.
    _CELL = frozenset({"td", "th"})

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


# Pages rarely repeat the question's exact words ("who won" -> "Spain defeated").
_SYNONYM_GROUPS = (
    ("won", "win", "wins", "winner", "winning", "champion", "defeated", "beat", "victory", "lifted"),
    ("last", "latest", "most recent", "newest", "recent"),
    ("latest", "last", "most recent", "newest", "current", "stable"),
    ("price", "cost", "trading at", "rate", "value"),
    ("version", "release", "released"),
    ("ceo", "chief executive"),
    ("richest", "wealthiest", "net worth"),
)
_SYNONYMS = {group[0]: group for group in _SYNONYM_GROUPS}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_WINDOW_CHARS = 280


def _windows(paragraph: str) -> list[str]:
    """Split a paragraph into runs of whole sentences of at most ~280 characters,
    so one long paragraph can't use up (or be dropped from) the excerpt budget."""

    if len(paragraph) <= _WINDOW_CHARS:
        return [paragraph]
    windows: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        sentence = sentence[:_WINDOW_CHARS]
        if current and len(current) + len(sentence) + 1 > _WINDOW_CHARS:
            windows.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        windows.append(current)
    return windows


def relevant_passages(text: str, terms: list[str], max_chars: int) -> str:
    """Pick the sentence runs that best match the question, in page order."""

    paragraphs = [paragraph for paragraph in text.split("\n") if paragraph]
    if not paragraphs:
        return ""
    if not terms:
        return " ".join(paragraphs)[:max_chars]
    year = time.localtime().tm_year
    this_year, last_year = str(year), str(year - 1)
    # A question that names its own year doesn't need a recency bonus.
    recency = not any(re.fullmatch(r"(?:19|20)\d{2}", term) for term in terms)
    windows = [
        (position, window, {term for term in terms if any(word in window.lower() for word in _SYNONYMS.get(term, (term,)))})
        for position, window in enumerate(window for paragraph in paragraphs[:400] for window in _windows(paragraph))
    ]
    # Words found all over the page (its topic, e.g. "IPL", "2026") say little about
    # where the answer is; rarer ones ("won" -> "defeated") do. Weight terms by rarity.
    # Intent words ("won", "latest", "price") name what is being asked, so they
    # count triple compared with topic words.
    weights = {
        term: math.log(1 + len(windows) / (1 + sum(1 for _p, _w, hits in windows if term in hits)))
        * (3 if term in _SYNONYMS else 1)
        for term in terms
    }
    total_weight = sum(weights.values()) or 1.0
    scored = []
    for position, window, hits in windows:
        if not hits:
            continue
        # Prefer passages with numbers/dates; footnote lines are a last resort.
        score = sum(weights[term] for term in hits) / total_weight + (0.15 if re.search(r"\d", window) else 0)
        if window.startswith(("↑", "^")):
            score -= 0.15
        # "Latest"/"who won" questions need the newest facts, not the page intro.
        if recency and this_year in window:
            score += 0.5
        elif recency and last_year in window:
            score += 0.25
        scored.append((score, position, window))
    if not scored:
        return ""
    chosen: list[tuple[int, str]] = []
    total = 0
    seen: set[str] = set()
    for _score, index, window in sorted(scored, key=lambda item: (-item[0], item[1])):
        if total + len(window) > max_chars or window in seen:
            continue
        seen.add(window)
        chosen.append((index, window))
        total += len(window)
        if total >= max_chars * 0.8:
            break
    return "\n".join(window for _index, window in sorted(chosen))
