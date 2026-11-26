"""Regional map layouts and English labels.

Bounds use (west, south, east, north).
Label coordinates use (longitude, latitude).

This module only defines layouts. It does not change the renderer.
"""

from __future__ import annotations

from typing import Any


CENTRAL_AMERICA: dict[str, Any] = {
    "id": "central_america",
    "bounds": (-119.0, -5.0, -59.0, 36.0),
    "context_country_codes": (
        "MEX",
        "CUB",
        "COL",
        "VEN",
    ),
    "water_labels": (
        {
            "text": "Pacific Ocean",
            "longitude": -106.0,
            "latitude": 5.0,
        },
        {
            "text": "Caribbean Sea",
            "longitude": -75.0,
            "latitude": 16.0,
        },
        {
            "text": "Gulf of\nMexico",
            "longitude": -91.0,
            "latitude": 25.0,
        },
    ),
    "country_label_overrides": {
        "GTM": {
            "longitude": -101.0,
            "latitude": 16.0,
            "leader_line": True,
        },
    },
}


KOREAN_PENINSULA: dict[str, Any] = {
    "id": "korean_peninsula",
    "bounds": (112.0, 27.0, 145.0, 47.0),
    "context_country_codes": (
        "CHN",
        "PRK",
        "KOR",
        "JPN",
    ),
    "water_labels": (
        {
            "text": "Yellow\nSea",
            "longitude": 123.5,
            "latitude": 35.0,
        },
        {
            "text": "Sea of Japan\n(East Sea)",
            "longitude": 135.0,
            "latitude": 40.0,
        },
        {
            "text": "Korea Strait",
            "longitude": 129.0,
            "latitude": 33.5,
        },
    ),
    "country_label_overrides": {
        "KOR": {
            "longitude": 138.0,
            "latitude": 35.5,
            "leader_line": True,
        },
        "PRK": {
            "longitude": 119.0,
            "latitude": 42.0,
            "leader_line": True,
        },
    },
}


def get_region_layout(properties: dict[str, Any]) -> dict[str, Any] | None:
    """Return a supported regional layout for a country."""
    code = str(properties.get("ADM0_A3") or "").strip().upper()

    if code in {"KOR", "PRK"}:
        return KOREAN_PENINSULA

    if properties.get("SUBREGION") == "Central America":
        return CENTRAL_AMERICA

    return None


def country_label(properties: dict[str, Any]) -> str:
    """Return an English country name without using localized name fields."""
    for field in ("NAME_EN", "NAME_LONG", "NAME", "ADM0_A3"):
        value = properties.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError("Country has no usable English name or country code")