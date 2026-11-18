"""Standalone country-map renderer. Not connected to the pipeline yet."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.map_regions import country_label, get_region_layout


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


def _render_country_map_closeup(
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

def render_country_map(
    *,
    geojson_path: Path,
    country_code: str,
    output_path: Path,
    width: int = 2400,
    height: int = 1350,
    caption_fraction: float = 0.22,
) -> dict:
    """Render a regional map when configured, otherwise use the close-up."""
    if width < 100 or height < 100:
        raise ValueError("Image dimensions must be at least 100 pixels.")

    if not 0 <= caption_fraction < 0.5:
        raise ValueError("caption_fraction must be in [0, 0.5).")

    data = json.loads(
        Path(geojson_path).read_text(encoding="utf-8-sig")
    )
    features = [
        feature
        for feature in data["features"]
        if feature.get("geometry")
    ]
    code = country_code.strip().upper()

    selected = [
        feature
        for feature in features
        if feature.get("properties", {}).get("ADM0_A3") == code
    ]

    if not selected:
        raise ValueError(f"Country not found by ADM0_A3: {code!r}")

    properties = selected[0]["properties"]
    layout = get_region_layout(properties)

    if layout is None:
        return _render_country_map_closeup(
            geojson_path=geojson_path,
            country_code=code,
            output_path=output_path,
            width=width,
            height=height,
            caption_fraction=caption_fraction,
        )

    map_height = round(height * (1 - caption_fraction))
    west, south, east, north = layout["bounds"]

    longitude = (west + east) / 2
    latitude = (south + north) / 2
    correction = max(math.cos(math.radians(latitude)), 0.01)

    def project(point):
        return (
            (point[0] - longitude) * correction,
            point[1] - latitude,
        )

    target_polygons = [
        polygon
        for feature in selected
        for polygon in _polygons(feature["geometry"])
    ]
    target_points = [
        point
        for polygon in target_polygons
        for ring in polygon
        for point in ring
    ]

    target_west = min(point[0] for point in target_points)
    target_east = max(point[0] for point in target_points)

    if target_east - target_west > 180:
        raise ValueError(
            "Countries crossing the date line are not supported yet."
        )

    scale = min(
        width * 0.94 / ((east - west) * correction),
        map_height * 0.94 / (north - south),
    )

    # Widen the view if needed to keep the selected country's full outline.
    max_x = max(abs(project(point)[0]) for point in target_points)
    max_y = max(abs(project(point)[1]) for point in target_points)

    scale = min(
        scale,
        width * 0.47 / max(max_x, 1e-9),
        map_height * 0.47 / max(max_y, 1e-9),
    )

    def pixel(lon, lat):
        x, y = project((lon, lat))
        return (
            round(width / 2 + x * scale),
            round(map_height / 2 - y * scale),
        )

    canvas = Image.new("RGB", (width, map_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    size_factor = min(width / 2400, height / 1350)

    def paint(feature, fill, outline, line_width):
        for polygon in _polygons(feature["geometry"]):
            outer = [pixel(*point[:2]) for point in polygon[0]]
            if len(outer) < 3:
                continue

            draw.polygon(outer, fill=fill)

            for hole in polygon[1:]:
                points = [pixel(*point[:2]) for point in hole]
                if len(points) >= 3:
                    draw.polygon(points, fill=BACKGROUND)

            for ring in polygon:
                points = [pixel(*point[:2]) for point in ring]
                if len(points) >= 2:
                    draw.line(
                        points + points[:1],
                        fill=outline,
                        width=line_width,
                    )

    for feature in features:
        paint(
            feature,
            LAND,
            BORDER,
            max(1, round(2 * size_factor)),
        )

    for feature in selected:
        paint(
            feature,
            HIGHLIGHT,
            HIGHLIGHT_BORDER,
            max(2, round(3 * size_factor)),
        )

    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    regular = ImageFont.truetype(
        str(font_dir / "DejaVuSans.ttf"),
        max(12, round(30 * size_factor)),
    )
    bold = ImageFont.truetype(
        str(font_dir / "DejaVuSans-Bold.ttf"),
        max(14, round(38 * size_factor)),
    )
    water_font = ImageFont.truetype(
        str(font_dir / "DejaVuSans.ttf"),
        max(12, round(34 * size_factor)),
    )

    label_boxes = []

    def label(text, lon, lat, *, font, color, required=False):
        position = pixel(lon, lat)
        stroke = max(1, round(3 * size_factor))
        box = draw.multiline_textbbox(
            position,
            text,
            font=font,
            anchor="mm",
            align="center",
            spacing=max(1, round(5 * size_factor)),
            stroke_width=stroke,
        )

        margin = max(2, round(8 * size_factor))
        if not (
            margin <= box[0]
            and margin <= box[1]
            and box[2] <= width - margin
            and box[3] <= map_height - margin
        ):
            if required:
                raise ValueError("Selected country label does not fit.")
            return False

        padded = (
            box[0] - margin,
            box[1] - margin,
            box[2] + margin,
            box[3] + margin,
        )
        for other in label_boxes:
            if (
                padded[0] < other[2]
                and padded[2] > other[0]
                and padded[1] < other[3]
                and padded[3] > other[1]
            ):
                if required:
                    raise ValueError("Selected country label overlaps.")
                return False

        draw.multiline_text(
            position,
            text,
            font=font,
            fill=color,
            anchor="mm",
            align="center",
            spacing=max(1, round(5 * size_factor)),
            stroke_width=stroke,
            stroke_fill=BACKGROUND,
        )
        label_boxes.append(box)
        return box

    target_lon = float(properties["LABEL_X"])
    target_lat = float(properties["LABEL_Y"])
    override = layout.get("country_label_overrides", {}).get(code, {})

    label_lon = float(override.get("longitude", target_lon))
    label_lat = float(override.get("latitude", target_lat))

    # Reserve the selected country's label before optional context labels.
    target_box = label(
        country_label(properties),
        label_lon,
        label_lat,
        font=bold,
        color=(244, 249, 255),
        required=True,
    )

    if override.get("leader_line"):
        country_point = pixel(target_lon, target_lat)
        label_point = pixel(label_lon, label_lat)
        edge_x = (
            target_box[2] if country_point[0] >= label_point[0]
            else target_box[0]
        )
        gap = max(2, round(8 * size_factor))
        edge_x += gap if country_point[0] >= label_point[0] else -gap

        draw.line(
            [country_point, (edge_x, label_point[1])],
            fill=HIGHLIGHT_BORDER,
            width=max(1, round(3 * size_factor)),
        )
        x, y = country_point
        radius = max(2, round(6 * size_factor))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=HIGHLIGHT_BORDER,
        )

    for item in layout["water_labels"]:
        label(
            item["text"],
            item["longitude"],
            item["latitude"],
            font=water_font,
            color=(133, 179, 209),
        )

    by_code = {
        feature["properties"].get("ADM0_A3"): feature
        for feature in features
    }

    for context_code in layout["context_country_codes"]:
        if context_code == code:
            continue

        feature = by_code.get(context_code)
        if feature is None:
            continue

        props = feature["properties"]
        if props.get("LABEL_X") is None or props.get("LABEL_Y") is None:
            continue

        label(
            country_label(props),
            float(props["LABEL_X"]),
            float(props["LABEL_Y"]),
            font=regular,
            color=(206, 218, 229),
        )

    image = Image.new("RGB", (width, height), BACKGROUND)
    image.paste(canvas, (0, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)

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
    map_pixels = width * map_height
    world_scale = min(
        width / (360 * correction),
        map_height / 180,
    )

    return {
        "country_code": code,
        "output_path": str(output_path),
        "map_size": [width, map_height],
        "caption_height": height - map_height,
        "initial_coverage_percent": round(
            area * world_scale**2 / map_pixels * 100, 3
        ),
        "final_coverage_percent": round(
            area * scale**2 / map_pixels * 100, 3
        ),
        "limited_by_full_outline": False,
        "mode": "country",
        "layout_mode": "regional",
        "region_id": layout["id"],
        "label_language": "en",
    }


def render_world_map(
    *,
    geojson_path: Path,
    output_path: Path,
    width: int = 2400,
    height: int = 1350,
    caption_fraction: float = 0.22,
) -> dict:
    """Render the whole world without highlighting or labels."""
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

    image = Image.new("RGB", (width, height), BACKGROUND)
    image.paste(canvas, (0, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)

    return {
        "country_code": None,
        "output_path": str(output_path),
        "map_size": [width, map_height],
        "caption_height": height - map_height,
        "mode": "world",
    }