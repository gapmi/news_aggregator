from __future__ import annotations

from datetime import date, datetime, timezone


MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def gmt_date_from_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError("Run timestamp must be a datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Run timestamp must include a timezone")

    return value.astimezone(timezone.utc).date().isoformat()


def validate_issue_date(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Issue date must be an ISO date string")

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Issue date must use YYYY-MM-DD") from exc

    if parsed.isoformat() != value:
        raise ValueError("Issue date must use YYYY-MM-DD")

    return value


def format_issue_date(value: str) -> str:
    parsed = date.fromisoformat(validate_issue_date(value))
    month = MONTH_NAMES[parsed.month - 1]

    return f"{month} {parsed.day}, {parsed.year} \u00b7 GMT"


def build_date_style(subtitle_style: str) -> str:
    fields = subtitle_style.split(",")
    alignment_fields = [
        index
        for index, field in enumerate(fields)
        if field.startswith("Alignment=")
    ]

    if len(alignment_fields) != 1:
        raise ValueError(
            "Subtitle style must contain exactly one Alignment field"
        )

    fields[alignment_fields[0]] = "Alignment=8"
    return ",".join(fields)