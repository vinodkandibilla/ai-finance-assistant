"""Deterministic historical-window resolution; no provider or LLM calls."""

import calendar
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

MAX_RECORDS = 100


def resolve_timeframe(request: str, today: date) -> dict[str, object]:
    text = request.lower()
    numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    # Normalize natural-language rolling windows before the numeric parser.
    text = re.sub(r"\b(last|past)\s+the\s+", r"\1 ", text)
    text = re.sub(
        r"\b(last|past)\s+(" + "|".join(numbers) + r")\s+(days?|weeks?|months?)\b",
        lambda match: f"{match[1]} {numbers[match[2]]} {match[3]}",
        text,
    )
    text = re.sub(r"\b(last|past)\s+(day|week|month)\b", r"\1 1 \2", text)
    growth = bool(
        re.search(
            r"\bgrowth\b|\breturns?\b|\bpercentage change\b|\bpercent change\b"
            r"|\bperformance\b|\bcompare\b|\bprice\s+change(?:d)?\b"
            r"|\b(?:highest|lowest|maximum|minimum|extrema)\b",
            text,
        )
    )
    interval = "1week" if re.search(r"\bweekly\b|\b1week\b", text) else "1day"
    if re.search(r"\bmonthly\b|\b1month\b", text):
        interval = "1month"
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    note = (
        "Relative month windows roll back from the as-of date, not a calendar month. "
        "Date bounds are inclusive; only available market observations are returned."
    )

    try:
        if len(dates) == 2:
            start, end = map(date.fromisoformat, dates)
        elif dates:
            raise ValueError("Please provide both start and end dates in YYYY-MM-DD format.")
        elif window := re.search(r"\b(?:last|past)\s+(\d+)\s+(days?|weeks?|months?)\b", text):
            count = int(window[1])
            unit = window[2].rstrip("s")
            if not 1 <= count <= MAX_RECORDS:
                raise ValueError("Please request between 1 and 100 observations.")
            if not re.search(r"\bdaily\b|\bweekly\b|\bmonthly\b|\b1(?:day|week|month)\b", text):
                interval = {"day": "1day", "week": "1week", "month": "1month"}[unit]
            if not growth and interval == {"day": "1day", "week": "1week", "month": "1month"}[unit]:
                return {
                    "status": "ready",
                    "parameters": {"interval": interval, "outputsize": count},
                    "as_of": today.isoformat(),
                    "note": "Latest available bars; daily bars are trading observations, "
                    "not calendar days. The current weekly/monthly bar may be incomplete.",
                }
            end = today
            if unit == "month":
                month_index = today.year * 12 + today.month - 1 - count
                year, month_zero = divmod(month_index, 12)
                start = date(
                    year,
                    month_zero + 1,
                    min(today.day, calendar.monthrange(year, month_zero + 1)[1]),
                )
            else:
                start = today - timedelta(days=count * (7 if unit == "week" else 1))
        else:
            raise ValueError(
                "Please provide explicit from/to dates in YYYY-MM-DD format, "
                "or a relative window such as 'last 12 weeks' or 'last 2 months'."
            )
        if start > end or end > today:
            raise ValueError("Use an ordered historical date range ending on or before today.")
        days = (end - start).days + 1
        estimated_records = {
            "1day": days,
            "1week": (days + 6) // 7 + 1,
            "1month": (end.year - start.year) * 12 + end.month - start.month + 1,
        }[interval]
        if not growth and estimated_records > MAX_RECORDS:
            raise ValueError(
                "Please narrow the window or use a coarser interval to stay "
                "within 100 records. Daily windows count calendar days conservatively."
            )
        return {
            "status": "ready",
            "parameters": {
                "interval": "1day" if growth else interval,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "outputsize": MAX_RECORDS,
            },
            "as_of": today.isoformat(),
            "note": note,
        }
    except (ValueError, OverflowError) as error:
        return {"status": "clarify", "clarification_question": str(error)}


@tool
async def resolve_market_timeframe(request: str) -> dict[str, object]:
    """Resolve the user's original historical request into Twelve Data parameters.

    Supports last N days/weeks/months and explicit ISO date pairs.
    Calendar/fiscal quarter names require explicit from/to dates.
    Never compute dates or counts yourself.
    """
    return resolve_timeframe(request, datetime.now(ZoneInfo("America/Phoenix")).date())
