"""Standalone country-map renderer. Not connected to the pipeline yet."""
from PIL import Image, ImageDraw, ImageFont
from scripts.map_regions import country_label, get_region_layout
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


def render_world_map(
    *,
    geojson_path: Path,
    output_path: Path,
    width: int = 2400,
    height: int = 1350,
    caption_fraction: float = 0.22,
    highlighted_country_codes: list[str] | None = None,
) -> dict:
    """Render the world map with highlighted countries and labels."""
    if width < 100 or height < 100:
        raise ValueError("Image dimensions must be at least 100 pixels.")

    if not 0 <= caption_fraction < 0.5:
        raise ValueError("caption_fraction must be in [0, 0.5).")

    data = json.loads(
        Path(geojson_path).read_text(encoding="utf-8-sig")
    )

    map_height = round(height * (1 - caption_fraction))
    scale = min(
        width * 0.94 / 360,
        map_height * 0.94 / 180,
    )
    
    highlighted_codes = {
        str(code).strip().upper()
        for code in (highlighted_country_codes or [])
        if str(code).strip()
    }

    features_by_code = {
        feature["properties"].get("ADM0_A3"): feature
        for feature in data["features"]
        if feature.get("geometry")
        and feature.get("properties", {}).get("ADM0_A3")
    }

    canvas = Image.new("RGB", (width, map_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    def pixels(ring):
        return [
            (
                round(width / 2 + longitude * scale),
                round(map_height / 2 - latitude * scale),
            )
            for longitude, latitude, *_ in ring
        ]

    for feature in data["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue

        for polygon in _polygons(geometry):
            if not polygon:
                continue

            outer = pixels(polygon[0])
            if len(outer) < 3:
                continue

            draw.polygon(outer, fill=LAND)

            for hole in polygon[1:]:
                hole_pixels = pixels(hole)
                if len(hole_pixels) >= 3:
                    draw.polygon(hole_pixels, fill=BACKGROUND)

            for ring in polygon:
                line = pixels(ring)
                if len(line) >= 2:
                    draw.line(
                        line + line[:1],
                        fill=BORDER,
                        width=1,
                    )

    highlighted_boxes: list[tuple[str, tuple[int, int, int, int]]] = []

    for country_code in highlighted_codes:
        feature = features_by_code.get(country_code)
        if feature is None:
            continue

        polygons = _polygons(feature["geometry"])
        all_pixels = [
            point
            for polygon in polygons
            for ring in polygon
            for point in pixels(ring)
        ]

        if not all_pixels:
            continue

        country_box = (
            min(point[0] for point in all_pixels),
            min(point[1] for point in all_pixels),
            max(point[0] for point in all_pixels),
            max(point[1] for point in all_pixels),
        )

        for polygon in polygons:
            outer = pixels(polygon[0])
            if len(outer) < 3:
                continue

            draw.polygon(outer, fill=HIGHLIGHT)

            for hole in polygon[1:]:
                hole_pixels = pixels(hole)
                if len(hole_pixels) >= 3:
                    draw.polygon(hole_pixels, fill=BACKGROUND)

            for ring in polygon:
                line = pixels(ring)
                if len(line) >= 2:
                    draw.line(
                        line + line[:1],
                        fill=HIGHLIGHT_BORDER,
                        width=max(2, round(width / 800)),
                    )

        highlighted_boxes.append(
            (country_label(feature["properties"]), country_box)
        )

    label_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        max(18, round(42 * (width / 2400))),
    )
    placed_boxes: list[tuple[int, int, int, int]] = []

    def boxes_overlap(first, second, margin):
        return (
            first[0] - margin < second[2]
            and first[2] + margin > second[0]
            and first[1] - margin < second[3]
            and first[3] + margin > second[1]
        )

    def draw_country_label(text, country_box):
        left, top, right, bottom = draw.textbbox(
            (0, 0),
            text,
            font=label_font,
        )
        label_width = right - left
        label_height = bottom - top

        country_width = country_box[2] - country_box[0]
        country_height = country_box[3] - country_box[1]
        fits_inside = (
            label_width <= country_width
            and label_height <= country_height
        )

        if fits_inside:
            candidates = [
                (
                    (country_box[0] + country_box[2]) / 2,
                    (country_box[1] + country_box[3]) / 2,
                )
            ]
        else:
            gap = max(18, round(28 * (width / 2400)))
            candidates = [
                (
                    (country_box[0] + country_box[2]) / 2,
                    country_box[1] - label_height / 2 - gap,
                ),
                (
                    (country_box[0] + country_box[2]) / 2,
                    country_box[3] + label_height / 2 + gap,
                ),
                (
                    country_box[0] - label_width / 2 - gap,
                    (country_box[1] + country_box[3]) / 2,
                ),
                (
                    country_box[2] + label_width / 2 + gap,
                    (country_box[1] + country_box[3]) / 2,
                ),
            ]

        margin = max(12, round(20 * (width / 2400)))

        for center_x, center_y in candidates:
            box = (
                round(center_x - label_width / 2),
                round(center_y - label_height / 2),
                round(center_x + label_width / 2),
                round(center_y + label_height / 2),
            )

            if not (
                margin <= box[0]
                and margin <= box[1]
                and box[2] <= width - margin
                and box[3] <= map_height - margin
            ):
                continue

            if any(
                boxes_overlap(box, other, margin)
                for other in placed_boxes
            ):
                continue

            if not fits_inside and boxes_overlap(
                box,
                country_box,
                margin,
            ):
                continue

            draw.text(
                (center_x, center_y),
                text,
                font=label_font,
                fill=(244, 249, 255),
                anchor="mm",
                stroke_width=max(2, round(4 * (width / 2400))),
                stroke_fill=BACKGROUND,
            )
            placed_boxes.append(box)
            return True

        return False

    for text, country_box in highlighted_boxes:
        draw_country_label(text, country_box)

    image = Image.new("RGB", (width, height), BACKGROUND)
    image.paste(canvas, (0, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)

    return {
        "country_code": (
            next(iter(highlighted_codes)) if highlighted_codes else None
        ),
        "highlighted_country_codes": sorted(highlighted_codes),
        "output_path": str(output_path),
        "map_size": [width, map_height],
        "caption_height": height - map_height,
        "mode": "world",
    }