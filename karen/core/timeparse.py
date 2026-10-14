"""Parse short spoken time expressions such as "in 10 minutes" or "at 5:30 pm"."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty five": 45, "sixty": 60, "half an": 0.5, "half a": 0.5,
}
_UNITS = {
    "second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400,
}
_RELATIVE = re.compile(
    r"\b(?:in|after)\s+(\d+(?:\.\d+)?|half an|half a|forty five|[a-z]+)\s+"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)
_ABSOLUTE = re.compile(
    r"\b(?:(today|tonight|tomorrow)\s+)?(?:at|by)\s+(\d{1,2})(?::|\.)?(\d{2})?\s*"
    r"(a\.?m\.?|p\.?m\.?)?(?:\s+(today|tonight|tomorrow))?",
    re.IGNORECASE,
)
_TOMORROW_ONLY = re.compile(r"\btomorrow(?:\s+morning)?\b", re.IGNORECASE)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY = re.compile(
    r"\b(?:(on|next|this|coming)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def parse_when(text: str, now: datetime | None = None) -> tuple[datetime | None, str]:
    """Return ``(due_time, text_without_time_phrase)``; ``due_time`` is local and aware."""

    now = now or datetime.now().astimezone()
    match = _RELATIVE.search(text)
    if match:
        raw_amount = match.group(1).lower()
        amount = _NUMBER_WORDS.get(raw_amount)
        if amount is None:
            try:
                amount = float(raw_amount)
            except ValueError:
                amount = None
        if amount is not None:
            unit = match.group(2).lower().rstrip("s")
            seconds = amount * _UNITS.get(unit, _UNITS.get(unit[:3], 60))
            return now + timedelta(seconds=seconds), _remove(text, match)

    weekday = _WEEKDAY.search(text)
    if weekday:
        target = _WEEKDAYS.index(weekday.group(2).lower())
        days_ahead = (target - now.weekday()) % 7
        if days_ahead == 0 and (weekday.group(1) or "").lower().startswith("next"):
            days_ahead = 7
        text = _remove(text, weekday)
        clock, remainder = parse_when(f"today {text}" if _ABSOLUTE.search(text) else text, now)
        hour, minute = (clock.hour, clock.minute) if clock is not None and _ABSOLUTE.search(text) else (9, 0)
        due = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due <= now:
            due += timedelta(days=7)
        return due, re.sub(r"^today\s*", "", remainder) if clock is not None else text

    match = _ABSOLUTE.search(text)
    if match:
        hour = int(match.group(2))
        minute = int(match.group(3) or 0)
        meridiem = (match.group(4) or "").lower().replace(".", "")
        day_word = (match.group(1) or match.group(5) or "").lower()
        if hour > 23 or minute > 59:
            return None, text
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif not meridiem and day_word == "tonight" and hour < 12:
            hour += 12
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if day_word == "tomorrow":
            due += timedelta(days=1)
        elif due <= now:
            # "at 5" without am/pm: prefer the next 5 o'clock that is still ahead.
            if not meridiem and hour < 12 and due + timedelta(hours=12) > now:
                due += timedelta(hours=12)
            else:
                due += timedelta(days=1)
        return due, _remove(text, match)

    match = _TOMORROW_ONLY.search(text)
    if match:
        due = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return due, _remove(text, match)
    return None, text


def _remove(text: str, match: re.Match[str]) -> str:
    remainder = (text[: match.start()] + " " + text[match.end() :]).strip(" ,.")
    return re.sub(r"\s+", " ", remainder)


def describe(due: datetime, now: datetime | None = None) -> str:
    """Return a short spoken description such as "at 5:30 PM tomorrow"."""

    now = now or datetime.now().astimezone()
    delta = due - now
    if delta.total_seconds() < 3600:
        minutes = max(1, round(delta.total_seconds() / 60))
        return f"in {minutes} minute{'s' if minutes != 1 else ''}"
    clock = due.strftime("%I:%M %p").lstrip("0")
    if due.date() == now.date():
        return f"at {clock}"
    if due.date() == (now + timedelta(days=1)).date():
        return f"at {clock} tomorrow"
    return f"at {clock} on {due.strftime('%A, %B %d')}"
