"""Standalone country-map renderer. Not connected to the pipeline yet."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


BACKGROUND = (15, 25, 38)
LAND = (39, 57, 74)
BORDER = (81, 105, 123)
HIGHLIGHT = (70, 177, 250)
HIGHLIGHT_BORDER = (157, 220, 250)


def _polygons(geometry: dict) -> list:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if kind == "Polygon":
        return [coordinates]
    if kind == "MultiPolygon":
        return coordinates

    raise ValueError(f"Unsupported geometry: {kind!r}")


def _ring_area(ring: list) -> float:
    return abs(
        sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(ring, ring[1:] + ring[:1])
        )
    ) / 2.0


def render_country_map(
    *,
    geojson_path: Path,
    country_code: str,
    output_path: Path,
    width: int = 2400,
    height: int = 1350,
    caption_fraction: float = 0.22,
) -> dict:
    """Render a country without labels and reserve an empty caption band."""
    if width < 100 or height < 100:
        raise ValueError("Image dimensions must be at least 100 pixels.")

    if not 0 <= caption_fraction < 0.5:
        raise ValueError("caption_fraction must be in [0, 0.5).")

    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    features = data["features"]
    code = country_code.strip().upper()

    selected = [
        feature
        for feature in features
        if feature.get("properties", {}).get("ADM0_A3") == code
    ]

    if not selected:
        raise ValueError(f"Country not found by ADM0_A3: {code!r}")

    target_polygons = [
        polygon
        for feature in selected
        for polygon in _polygons(feature["geometry"])
    ]

    points = [
        point
        for polygon in target_polygons
        for ring in polygon
        for point in ring
    ]

    west = min(point[0] for point in points)
    east = max(point[0] for point in points)
    south = min(point[1] for point in points)
    north = max(point[1] for point in points)

    if east - west > 180:
        raise ValueError(
            "Countries crossing the date line are not supported yet."
        )

    latitude = (south + north) / 2
    longitude = (west + east) / 2
    correction = max(math.cos(math.radians(latitude)), 0.01)

    def project(point):
        return (
            (point[0] - longitude) * correction,
            point[1] - latitude,
        )

    projected = [
        [
            [project(point) for point in ring]
            for ring in polygon
        ]
        for polygon in target_polygons
    ]

    area = sum(
        _ring_area(polygon[0])
        - sum(_ring_area(hole) for hole in polygon[1:])
        for polygon in projected
    )

    if area <= 0:
        raise ValueError("Selected country has no positive projected area.")

    map_height = round(height * (1 - caption_fraction))
    map_pixels = width * map_height
    target_width = max((east - west) * correction, 1e-9)
    target_height = max(north - south, 1e-9)

    world_scale = min(
        width / (360 * correction),
        map_height / 180,
    )
    initial_coverage = area * world_scale**2 / map_pixels

    requested_scale = (
        math.sqrt(0.10 * map_pixels / area)
        if initial_coverage <= 0.05
        else world_scale
    )

    # The full outline has priority over the target coverage.
    fit_scale = min(
        width * 0.88 / target_width,
        map_height * 0.88 / target_height,
    )
    scale = min(requested_scale, fit_scale)

    def pixels(ring):
        return [
            (
                round(width / 2 + x * scale),
                round(map_height / 2 - y * scale),
            )
            for x, y in ring
        ]

    canvas = Image.new("RGB", (width, map_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    def paint(polygons, fill, outline, line_width):
        for polygon in polygons:
            outer = pixels(polygon[0])

            if len(outer) < 3:
                continue

            draw.polygon(outer, fill=fill)

            for hole in polygon[1:]:
                hole_pixels = pixels(hole)
                if len(hole_pixels) >= 3:
                    draw.polygon(hole_pixels, fill=BACKGROUND)

            for ring in polygon:
                line = pixels(ring)
                if len(line) >= 2:
                    draw.line(
                        line + line[:1],
                        fill=outline,
                        width=line_width,
                    )

    for feature in features:
        geometry = feature.get("geometry")
        if not geometry:
            continue

        polygons = [
            [
                [project(point) for point in ring]
                for ring in polygon
            ]
            for polygon in _polygons(geometry)
        ]
        paint(polygons, LAND, BORDER, 1)

    paint(
        projected,
        HIGHLIGHT,
        HIGHLIGHT_BORDER,
        max(2, round(width / 900)),
    )

    image = Image.new("RGB", (width, height), BACKGROUND)
    image.paste(canvas, (0, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)

    return {
        "country_code": code,
        "output_path": str(output_path),
        "map_size": [width, map_height],
        "caption_height": height - map_height,
        "initial_coverage_percent": round(
            initial_coverage * 100, 3
        ),
        "final_coverage_percent": round(
            area * scale**2 / map_pixels * 100, 3
        ),
        "limited_by_full_outline": requested_scale > fit_scale,
    }