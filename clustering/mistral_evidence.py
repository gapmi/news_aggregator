from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup


DEFAULT_MAX_SUMMARY_CHARS = 4000


_BOILERPLATE_LINK_RE = re.compile(
    r"^(?:"
    r"continue reading(?:\.{3}|…)?"
    r"|read more(?:\.{3}|…)?"
    r"|leer m[aá]s(?:\.{3}|…)?"
    r"|seguir leyendo(?:\.{3}|…)?"
    r"|читать далее(?:\.{3}|…)?"
    r"|send us your questions\b.*"
    r")$",
    flags=re.IGNORECASE,
)

_BOILERPLATE_BLOCK_RE = re.compile(
    r"^(?:"
    r"send us your questions\b"
    r"|sign up for (?:our|the)\b"
    r"|subscribe to (?:our|the)\b"
    r"|the post\b.*\bappeared first on\b"
    r")",
    flags=re.IGNORECASE,
)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _comparison_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum()
    )


def _title_variants(
    title: str,
    source_name: str | None,
) -> set[str]:
    normalized_title = _normalize_whitespace(html.unescape(title))
    variants = {normalized_title}

    if " - " in normalized_title:
        variants.add(normalized_title.rsplit(" - ", 1)[0].rstrip(" -"))

    if source_name:
        normalized_source = _normalize_whitespace(
            html.unescape(source_name)
        )

        for variant in tuple(variants):
            variants.add(f"{variant} {normalized_source}")
            variants.add(f"{variant} - {normalized_source}")

    return {
        _comparison_key(variant)
        for variant in variants
        if variant
    }


def _extract_description_text(raw_description: str) -> str:
    soup = BeautifulSoup(raw_description, "html.parser")

    for tag in soup.find_all(
        ["script", "style", "noscript", "iframe", "form"]
    ):
        tag.decompose()

    for block in list(soup.find_all(["p", "li"])):
        if block.parent is None:
            continue

        text = _normalize_whitespace(block.get_text(" ", strip=True))

        if _BOILERPLATE_BLOCK_RE.search(text):
            block.decompose()

    for link in list(soup.find_all("a")):
        text = _normalize_whitespace(link.get_text(" ", strip=True))

        if _BOILERPLATE_LINK_RE.fullmatch(text):
            link.decompose()

    for tag in soup.find_all("font"):
        tag.decompose()

    return _normalize_whitespace(
        html.unescape(soup.get_text(" ", strip=True))
    )


def _limit_summary(
    text: str,
    max_chars: int,
) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False

    prefix = text[:max_chars]

    boundaries = list(
        re.finditer(
            r"""[.!?](?:["'”’)]*)(?=\s|$)""",
            prefix,
        )
    )

    if not boundaries:
        return "", True

    return prefix[:boundaries[-1].end()].strip(), True


def prepare_rss_evidence(
    *,
    title: str,
    rss_description: str | None,
    source_name: str | None = None,
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
) -> dict[str, Any]:
    """
    Prepare RSS summary text without inventing or completing facts.

    Status describes text availability, not factual verification.
    """
    if not isinstance(title, str):
        raise ValueError("title must be a string")

    if (
        not isinstance(max_summary_chars, int)
        or isinstance(max_summary_chars, bool)
        or max_summary_chars < 100
    ):
        raise ValueError("max_summary_chars must be at least 100")

    if (
        rss_description is not None
        and not isinstance(rss_description, str)
    ):
        raise ValueError("rss_description must be a string or None")

    raw = rss_description or ""

    result: dict[str, Any] = {
        "summary_text": None,
        "summary_status": "empty",
        "summary_origin": "rss_description",
        "raw_description_chars": len(raw),
        "cleaned_description_chars": 0,
        "summary_truncated": False,
    }

    if not raw.strip():
        return result

    cleaned = _extract_description_text(raw)
    result["cleaned_description_chars"] = len(cleaned)

    if not cleaned:
        result["summary_status"] = "empty_after_cleanup"
        return result

    if _comparison_key(cleaned) in _title_variants(
        title,
        source_name,
    ):
        result["summary_status"] = "headline_only"
        return result

    summary, truncated = _limit_summary(
        cleaned,
        max_summary_chars,
    )
    result["summary_truncated"] = truncated

    if not summary:
        result["summary_status"] = "too_long_without_sentence_boundary"
        return result

    result["summary_text"] = summary
    result["summary_status"] = "available"

    return result