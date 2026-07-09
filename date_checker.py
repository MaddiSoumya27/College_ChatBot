"""
date_checker.py
---------------
Deadline / date checker tool for BVRIT Hyderabad chatbot.

Given a date string and an optional event label, this tool computes:
  - Whether the date is in the past, today, or upcoming
  - How many days ago / how many days remaining
  - A human-readable status message

Also exposes DATE_CHECKER_TOOL — the OpenAI-compatible tool definition
that lets the LLM decide to call this tool via function calling.

Known BVRIT-relevant event dates are pre-loaded so the LLM can ask
"is the TG EAPCET deadline passed?" and get a concrete answer.
"""

from datetime import date, datetime, timedelta
from typing import Optional

# ── Known BVRIT-relevant event dates ─────────────────────────────────────────
# Keep these updated each academic year.
# Format: "event_key": ("Display Name", "YYYY-MM-DD")
KNOWN_EVENTS: dict[str, tuple[str, str]] = {
    # Admissions
    "eapcet":             ("TG EAPCET 2025 (Exam)",               "2025-05-21"),
    "eapcet_results":     ("TG EAPCET 2025 Results",              "2025-06-10"),
    "counselling":        ("TG EAPCET 2025 Counselling Start",    "2025-07-01"),
    "counselling_end":    ("TG EAPCET 2025 Counselling End",      "2025-08-15"),
    "admission_close":    ("BVRIT Admissions Closing Date",       "2025-09-01"),
    "pgecet":             ("PGECET 2025 (M.Tech Admissions)",     "2025-07-10"),

    # Academic calendar
    "sem1_start":         ("Semester 1 Start (AY 2025-26)",       "2025-09-15"),
    "sem1_end":           ("Semester 1 End / Exams",              "2026-01-10"),
    "sem2_start":         ("Semester 2 Start (AY 2025-26)",       "2026-01-20"),
    "sem2_end":           ("Semester 2 End / Exams",              "2026-05-30"),
    "graduation":         ("Graduation Day 2025",                 "2025-11-15"),

    # Events
    "synergia":           ("Synergia 2026 (Tech & Cultural Fest)","2026-03-31"),
    "annual_day":         ("Annual Day 2026",                     "2026-04-05"),
    "tedx":               ("TedX at BVRIT Hyderabad 2026",        "2026-02-15"),
    "milan":              ("Milan 2026",                          "2026-03-20"),

    # Scholarships
    "scholarship":        ("Merit Scholarship Application",       "2025-10-31"),
    "fee_reimbursement":  ("Fee Reimbursement Application",       "2025-11-30"),
}


def check_date(
    event_date: str,
    event_name: Optional[str] = None,
    reference_date: Optional[str] = None,
) -> dict:
    """
    Compare an event date against today (or a reference date).

    Parameters
    ----------
    event_date : str
        The date to check. Accepts:
          - ISO format: "YYYY-MM-DD"
          - Known event key: e.g. "eapcet", "counselling", "synergia"
    event_name : str, optional
        Human-readable label for the event. Auto-filled if event_date is a key.
    reference_date : str, optional
        "Today" for comparison. Defaults to actual today. Format: "YYYY-MM-DD"

    Returns
    -------
    dict with keys:
        event_name, event_date, reference_date, status, days_diff,
        message, is_past, is_today, is_upcoming
    """
    # Resolve known event keys
    if event_date.lower().replace(" ", "_") in KNOWN_EVENTS:
        key = event_date.lower().replace(" ", "_")
        auto_name, date_str = KNOWN_EVENTS[key]
        event_name  = event_name or auto_name
        event_date  = date_str
    else:
        event_name = event_name or "Event"

    # Parse event date
    try:
        evt_date = datetime.strptime(event_date, "%Y-%m-%d").date()
    except ValueError:
        # Try common alternative formats
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y"):
            try:
                evt_date = datetime.strptime(event_date, fmt).date()
                event_date = evt_date.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
        else:
            return {
                "event_name":     event_name,
                "event_date":     event_date,
                "reference_date": str(date.today()),
                "status":         "error",
                "days_diff":      0,
                "message":        f"Could not parse date '{event_date}'. Use YYYY-MM-DD format.",
                "is_past":        False,
                "is_today":       False,
                "is_upcoming":    False,
            }

    # Reference date (default = today)
    today = date.today()
    if reference_date:
        try:
            today = datetime.strptime(reference_date, "%Y-%m-%d").date()
        except ValueError:
            pass  # fall back to actual today

    delta = (evt_date - today).days  # positive = future, negative = past

    if delta < 0:
        status    = "past"
        days_diff = abs(delta)
        if days_diff == 1:
            time_str = "yesterday"
        elif days_diff < 30:
            time_str = f"{days_diff} days ago"
        elif days_diff < 365:
            months = days_diff // 30
            time_str = f"about {months} month{'s' if months > 1 else ''} ago"
        else:
            years = days_diff // 365
            time_str = f"about {years} year{'s' if years > 1 else ''} ago"
        message = f"⏰ **{event_name}** has already passed ({time_str}, on {evt_date.strftime('%d %B %Y')})."
    elif delta == 0:
        status    = "today"
        days_diff = 0
        message   = f"🎯 **{event_name}** is **TODAY** ({evt_date.strftime('%d %B %Y')})!"
    else:
        status    = "upcoming"
        days_diff = delta
        if days_diff == 1:
            time_str = "tomorrow"
        elif days_diff < 7:
            time_str = f"in {days_diff} days"
        elif days_diff < 30:
            weeks = days_diff // 7
            time_str = f"in {weeks} week{'s' if weeks > 1 else ''} ({days_diff} days)"
        elif days_diff < 365:
            months = days_diff // 30
            time_str = f"in about {months} month{'s' if months > 1 else ''} ({days_diff} days)"
        else:
            years  = days_diff // 365
            months = (days_diff % 365) // 30
            time_str = f"in {years} year{'s' if years > 1 else ''} and {months} month{'s' if months > 1 else ''}"
        message = f"📅 **{event_name}** is coming up **{time_str}** (on {evt_date.strftime('%d %B %Y')})."

    return {
        "event_name":     event_name,
        "event_date":     str(evt_date),
        "reference_date": str(today),
        "status":         status,
        "days_diff":      days_diff,
        "message":        message,
        "is_past":        status == "past",
        "is_today":       status == "today",
        "is_upcoming":    status == "upcoming",
    }


def list_upcoming_events(days_ahead: int = 90) -> list[dict]:
    """Return all known events occurring within the next N days."""
    today = date.today()
    results = []
    for key, (name, date_str) in KNOWN_EVENTS.items():
        evt = datetime.strptime(date_str, "%Y-%m-%d").date()
        delta = (evt - today).days
        if 0 <= delta <= days_ahead:
            results.append({
                "key":        key,
                "event_name": name,
                "event_date": date_str,
                "days_away":  delta,
            })
    return sorted(results, key=lambda x: x["days_away"])


def format_date_result(result: dict) -> str:
    """Format the date_checker result as a chat-ready markdown string."""
    lines = [result["message"], ""]
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Event | {result['event_name']} |")
    lines.append(f"| Event Date | {datetime.strptime(result['event_date'], '%Y-%m-%d').strftime('%d %B %Y')} |")
    lines.append(f"| Today | {datetime.strptime(result['reference_date'], '%Y-%m-%d').strftime('%d %B %Y')} |")
    lines.append(f"| Status | {'✅ Past' if result['is_past'] else '🎯 Today' if result['is_today'] else '⏳ Upcoming'} |")
    if result["status"] != "error":
        if result["is_past"]:
            lines.append(f"| Days Since | {result['days_diff']} days |")
        elif result["is_upcoming"]:
            lines.append(f"| Days Remaining | {result['days_diff']} days |")
    lines.append("")
    lines.append("---")
    lines.append(
        "📋 *For the latest dates, check the official BVRIT Hyderabad website:* "
        "**bvrithyderabad.edu.in** or contact **+91 40 4241 7773**"
    )
    return "\n".join(lines)


# ── Tool definition for LLM function/tool calling ────────────────────────────

DATE_CHECKER_TOOL = {
    "type": "function",
    "function": {
        "name": "check_date",
        "description": (
            "Check whether a BVRIT Hyderabad event date or academic deadline is "
            "in the past, today, or upcoming — and how many days away it is. "
            "Use this when the user asks about: admission deadlines, exam dates, "
            "counselling dates, fee reimbursement deadlines, fest dates, semester "
            "start/end dates, graduation day, or any specific calendar date. "
            "Do NOT use for fee calculations or general college information questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_date": {
                    "type": "string",
                    "description": (
                        "The date to check. Either:\n"
                        "  - A known event key: 'eapcet', 'counselling', 'counselling_end', "
                        "'admission_close', 'pgecet', 'sem1_start', 'sem1_end', 'sem2_start', "
                        "'sem2_end', 'graduation', 'synergia', 'annual_day', 'tedx', 'milan', "
                        "'scholarship', 'fee_reimbursement'\n"
                        "  - Or an ISO date string: 'YYYY-MM-DD'"
                    ),
                },
                "event_name": {
                    "type": "string",
                    "description": (
                        "Optional human-readable label for the event. "
                        "Auto-filled for known event keys."
                    ),
                },
                "reference_date": {
                    "type": "string",
                    "description": (
                        "Optional reference date for comparison (default = today). "
                        "Format: YYYY-MM-DD"
                    ),
                },
            },
            "required": ["event_date"],
        },
    },
}
