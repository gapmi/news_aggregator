"""Conservative country-name matching for scene narration.

Matches are mentions, not proof of the location of an event.
Multiple matches must not be reduced to the first country.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


NAME_FIELDS = ("NAME_EN", "ADMIN", "NAME", "FORMAL_EN")

EXTRA_ALIASES = {
    "GTM": ("Guatemala",),
    "CHN": ("China",),
    "USA": (
        "United States",
        "United States of America",
        "U.S.",
        "U.S.A.",
        "USA",
    ),
}

# These names can identify something other than a country.
# Do not highlight automatically when one occurs.
AMBIGUOUS_NAMES = {
    "georgia",
    "jordan",
    "chad",
}


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains(text: str, name: str) -> bool:
    return re.search(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)",
        text,
    ) is not None


def resolve_scene_country(
    *,
    narration: str,
    geojson_path: Path,
) -> dict:
    """Return one unambiguous country code, or None.

    Only explicit names in narration are considered.
    No inference from visual prompts, adjectives or word order.
    """
    data = json.loads(
        Path(geojson_path).read_text(encoding="utf-8-sig")
    )
    text = _normalize(narration)

    aliases_by_code: dict[str, set[str]] = {}

    for feature in data["features"]:
        properties = feature.get("properties", {})
        code = str(properties.get("ADM0_A3") or "").strip()

        if not re.fullmatch(r"[A-Z]{3}", code):
            continue

        aliases = aliases_by_code.setdefault(code, set())

        for field in NAME_FIELDS:
            value = properties.get(field)
            if isinstance(value, str):
                name = _normalize(value)
                if len(name) >= 4:
                    aliases.add(name)

        for value in EXTRA_ALIASES.get(code, ()):
            aliases.add(_normalize(value))

    matches: dict[str, list[str]] = {}

    for code, aliases in aliases_by_code.items():
        found = sorted(
            name for name in aliases if _contains(text, name)
        )
        if found:
            matches[code] = found

    ambiguous = sorted(
        name for name in AMBIGUOUS_NAMES
        if _contains(text, name)
    )

    if ambiguous:
        status = "ambiguous_name"
        country_code = None
    elif len(matches) == 1:
        status = "single_country_mention"
        country_code = next(iter(matches))
    elif matches:
        status = "multiple_countries"
        country_code = None
    else:
        status = "no_country"
        country_code = None

    return {
        "country_code": country_code,
        "status": status,
        "matched_names": matches,
        "ambiguous_names": ambiguous,
    }