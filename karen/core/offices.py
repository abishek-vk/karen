"""Recognize "who is the <office> of <place>" questions, which always need live data."""

from __future__ import annotations

import re

ROLES = {
    "cm": "Chief Minister",
    "chief minister": "Chief Minister",
    "pm": "Prime Minister",
    "prime minister": "Prime Minister",
    "president": "President",
    "vice president": "Vice President",
    "vp": "Vice President",
    "governor": "Governor",
    "deputy chief minister": "Deputy Chief Minister",
    "deputy cm": "Deputy Chief Minister",
    "chief justice": "Chief Justice",
    "mayor": "Mayor",
    "chancellor": "Chancellor",
    "monarch": "Monarch",
    "king": "Monarch",
    "queen": "Monarch",
    "leader of the opposition": "Leader of the Opposition",
    "speaker": "Speaker",
}
PLACES = {
    "tn": "Tamil Nadu", "tamilnadu": "Tamil Nadu", "kl": "Kerala", "ka": "Karnataka",
    "ap": "Andhra Pradesh", "ts": "Telangana", "tg": "Telangana", "up": "Uttar Pradesh",
    "mp": "Madhya Pradesh", "mh": "Maharashtra", "wb": "West Bengal", "hp": "Himachal Pradesh",
    "uk": "the United Kingdom", "united kingdom": "the United Kingdom", "britain": "the United Kingdom",
    "us": "the United States", "usa": "the United States", "america": "the United States",
    "united states": "the United States", "uae": "the United Arab Emirates",
    "pondy": "Puducherry", "pondicherry": "Puducherry", "delhi": "Delhi", "nct": "Delhi",
}
_SMALL_WORDS = frozenset({"the", "of", "and"})
_ROLE_PATTERN = "|".join(sorted((re.escape(role) for role in ROLES), key=len, reverse=True))
_QUESTION = re.compile(
    rf"^(?:who(?:'s| is| are)?|tell me|name)\s+(?:is\s+)?(?:the\s+)?(?:current\s+|present\s+|new\s+|sitting\s+)?"
    rf"({_ROLE_PATTERN})\s+(?:of|in|for)\s+(.+?)(?:\s+(?:now|currently|today|right now|at present))?\s*\??$",
    re.IGNORECASE,
)
_POSSESSIVE = re.compile(
    rf"^(?:who(?:'s| is)?\s+)(.+?)(?:'s|s')\s+(?:current\s+|present\s+)?({_ROLE_PATTERN})\s*\??$",
    re.IGNORECASE,
)


def _place_name(raw: str) -> str:
    key = " ".join(raw.lower().strip(" ?.!\"'").split())
    key = re.sub(r"^the\s+", "", key)
    if key in PLACES:
        return PLACES[key]
    return " ".join(word if word in _SMALL_WORDS else word.capitalize() for word in key.split())


def parse_office_question(text: str) -> tuple[str, str] | None:
    """Return ``(office, place)`` such as ``("Chief Minister", "Tamil Nadu")``."""

    cleaned = " ".join(text.strip().split())
    match = _QUESTION.match(cleaned)
    if match:
        role, place = match.group(1), match.group(2)
    else:
        match = _POSSESSIVE.match(cleaned)
        if not match:
            return None
        place, role = match.group(1), match.group(2)
    place_name = _place_name(place)
    if not place_name or len(place_name) > 60:
        return None
    return ROLES[role.lower()], place_name
