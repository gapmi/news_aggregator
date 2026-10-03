from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 2400
HEIGHT = 1350
FPS = 30

MAX_ASSETS_PER_SCENE = 3

RENDER_NATIVE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

ASSET_SOURCE_PRIORITY = {
    "rss_image": 0,
    "og_image": 1,
    "twitter_image": 2,
    "page_image": 3,
    "viewport_screenshot": 4,
}

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
)

REGULAR_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select collected visual assets per scene and create the "
            "render manifest required by render_news_video_v2.py."
        )
    )
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument(
        "--audio-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--assets-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-assets-per-scene",
        type=int,
        default=MAX_ASSETS_PER_SCENE,
    )
    return parser.parse_args()


def required_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} was not found: {path}")

    if path.stat().st_size == 0:
        raise RuntimeError(f"{label} is empty: {path}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    required_file(path, label)

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise RuntimeError(f"{label} root must be a JSON object")

    return data


def load_asset_rows(path: Path) -> list[dict[str, str]]:
    required_file(path, "assets manifest")

    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"Assets manifest contains no rows: {path}")

    required_columns = {
        "scene_number",
        "candidate_rank",
        "is_representative",
        "asset_type",
        "asset_source",
        "asset_path",
        "status",
    }

    actual_columns = set(rows[0].keys())
    missing_columns = required_columns - actual_columns

    if missing_columns:
        raise RuntimeError(
            "Assets manifest misses required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return rows


def integer_value(value: str | None, fallback: int) -> int:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError):
        return fallback


def is_true(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "y",
    }


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").split())


def image_quality_metrics(path: Path) -> dict[str, float]:
    """
    Calculate lightweight visual-information metrics on a small image copy.

    This does not decide whether an image is editorially useful. It detects
    likely empty, flat, monochrome, or visually uninformative frames.
    """
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((320, 180))

        pixels = list(image.getdata())
        total = len(pixels)

        if total == 0:
            return {
                "dominant_share": 1.0,
                "entropy": 0.0,
                "edge_share": 0.0,
                "saturation_mean": 0.0,
            }

        quantized = [
            (red // 32, green // 32, blue // 32)
            for red, green, blue in pixels
        ]

        color_counts = Counter(quantized)
        dominant_share = max(color_counts.values()) / total

        gray = image.convert("L")
        histogram = gray.histogram()

        entropy = 0.0

        for count in histogram:
            if count <= 0:
                continue

            probability = count / total
            entropy -= probability * math.log2(probability)

        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_share = sum(
            value >= 35
            for value in edges.getdata()
        ) / total

        saturation_total = 0.0

        for red, green, blue in pixels:
            maximum = max(red, green, blue)
            minimum = min(red, green, blue)

            if maximum:
                saturation_total += (maximum - minimum) / maximum

        saturation_mean = saturation_total / total

    return {
        "dominant_share": dominant_share,
        "entropy": entropy,
        "edge_share": edge_share,
        "saturation_mean": saturation_mean,
    }


def is_low_information_visual(path: Path) -> tuple[bool, dict[str, float]]:
    """
    Reject an image only when at least two independent weak signals appear.

    This avoids discarding a legitimate photo merely because it is dark,
    low-saturation, simple, or dominated by one broad color.
    """
    metrics = image_quality_metrics(path)

    weak_signals = (
        metrics["dominant_share"] >= 0.78,
        metrics["entropy"] <= 3.2,
        metrics["edge_share"] <= 0.015,
        metrics["saturation_mean"] <= 0.08,
    )

    return sum(weak_signals) >= 2, metrics


def asset_sort_key(row: dict[str, str]) -> tuple[int, int, int, int, int, str]:
    asset_type = str(row.get("asset_type") or "").strip()
    source = str(row.get("asset_source") or "").strip()

    is_real_image = asset_type == "image"
    is_screenshot = asset_type == "screenshot"
    representative = is_true(row.get("is_representative"))

    return (
        0 if representative else 1,
        0 if is_real_image else 1,
        1 if is_screenshot else 0,
        integer_value(row.get("candidate_rank"), 9999),
        ASSET_SOURCE_PRIORITY.get(source, 9999),
        str(row.get("asset_path") or ""),
    )


def select_assets(
    *,
    scene_number: int,
    rows: list[dict[str, str]],
    max_assets: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    eligible: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []

    for row in rows:
        if integer_value(row.get("scene_number"), -1) != scene_number:
            continue

        if str(row.get("status") or "").strip() != "ok":
            continue

        if str(row.get("asset_type") or "").strip() != "image":
            continue

        asset_path = Path(str(row.get("asset_path") or "").strip())

        if not asset_path.is_file():
            continue

        if asset_path.stat().st_size == 0:
            continue

        try:
            low_information, metrics = is_low_information_visual(asset_path)
        except Exception as exc:
            rejected.append(
                {
                    "scene_number": scene_number,
                    "asset_path": str(asset_path),
                    "reason": f"Could not analyze visual quality: {exc}",
                    "metrics": None,
                }
            )
            continue

        if low_information:
            rejected.append(
                {
                    "scene_number": scene_number,
                    "asset_path": str(asset_path),
                    "reason": "Low-information visual",
                    "metrics": {
                        key: round(value, 4)
                        for key, value in metrics.items()
                    },
                }
            )
            continue

        eligible.append(row)

    eligible.sort(key=asset_sort_key)

    selected: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for row in eligible:
        asset_path = str(row["asset_path"])

        if asset_path in seen_paths:
            continue

        seen_paths.add(asset_path)
        selected.append(row)

        if len(selected) >= max_assets:
            break

    return selected, rejected


def copy_asset(
    *,
    source_path: Path,
    target_dir: Path,
    index: int,
) -> Path:
    source_suffix = source_path.suffix.lower()

    if source_suffix in RENDER_NATIVE_EXTENSIONS:
        output_suffix = ".jpg" if source_suffix == ".jpeg" else source_suffix
        target_path = target_dir / f"asset_{index:02d}{output_suffix}"

        shutil.copy2(source_path, target_path)

    else:
        target_path = target_dir / f"asset_{index:02d}.png"

        try:
            with Image.open(source_path) as image:
                if image.mode in {"RGBA", "LA"}:
                    background = Image.new(
                        "RGB",
                        image.size,
                        (0, 0, 0),
                    )
                    alpha = image.getchannel("A")
                    background.paste(
                        image,
                        mask=alpha,
                    )
                    image = background
                else:
                    image = image.convert("RGB")

                image.save(
                    target_path,
                    format="PNG",
                    optimize=True,
                )

        except Exception as exc:
            raise RuntimeError(
                "Could not normalize renderer asset: "
                f"{source_path} → {target_path}: {exc}"
            ) from exc

    if not target_path.is_file() or target_path.stat().st_size == 0:
        raise RuntimeError(
            "Could not create selected renderer asset: "
            f"{source_path} → {target_path}"
        )

    return target_path


def load_font(
    candidates: tuple[str, ...],
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in candidates:
        path = Path(candidate)

        if path.is_file():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


def split_words(
    text: str,
    *,
    max_words: int,
) -> str:
    words = normalize_text(text).split()

    if len(words) <= max_words:
        return " ".join(words)

    return " ".join(words[:max_words]).rstrip(",.;:") + "…"


def narration_to_card_text(
    narration: str,
) -> tuple[str, str]:
    """
    Create a compact card headline and secondary line using only narration.

    No event, statistic, source, or interpretation is invented here.
    """
    text = normalize_text(narration)

    if not text:
        return (
            "News update",
            "A news development is being reviewed.",
        )

    sentences = [
        sentence.strip()
        for sentence in (
            text.replace("!", ".")
            .replace("?", ".")
            .split(".")
        )
        if sentence.strip()
    ]

    headline_source = sentences[0] if sentences else text

    detail_source = (
        sentences[1]
        if len(sentences) > 1
        else headline_source
    )

    headline = split_words(
        headline_source,
        max_words=11,
    )

    detail = split_words(
        detail_source,
        max_words=16,
    )

    return headline, detail


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = normalize_text(text).split()

    if not words:
        return []

    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])

        left, _, right, _ = draw.textbbox(
            (0, 0),
            candidate,
            font=font,
        )

        if current and right - left > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        lines.append(" ".join(current))

    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    *,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int,
) -> int:
    x, y = position

    for line in wrap_text(
        draw,
        text,
        font,
        max_width,
    ):
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
        )

        _, top, _, bottom = draw.textbbox(
            (x, y),
            line,
            font=font,
        )

        y += (bottom - top) + line_gap

    return y


def create_fallback_asset(
    *,
    target_dir: Path,
    scene_number: int,
    narration: str,
) -> Path:
    """
    Create a readable editorial fallback card.

    The card contains only a compact version of the actual scene narration.
    It deliberately contains no AI-generated event imagery or invented facts.
    """
    target_path = target_dir / "asset_01_fallback.png"

    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (15, 25, 38),
    )

    pixels = image.load()

    for y in range(HEIGHT):
        vertical = y / max(HEIGHT - 1, 1)

        for x in range(WIDTH):
            horizontal = x / max(WIDTH - 1, 1)

            center_distance = (
                (horizontal - 0.38) ** 2
                + (vertical - 0.42) ** 2
            ) ** 0.5

            glow = max(0.0, 1.0 - center_distance * 2.15)
            vignette = abs(horizontal - 0.5) * 18

            red = int(12 + glow * 18 - vignette * 0.20)
            green = int(23 + glow * 34 - vignette * 0.32)
            blue = int(38 + glow * 58 - vignette * 0.45)

            pixels[x, y] = (
                max(0, min(255, red)),
                max(0, min(255, green)),
                max(0, min(255, blue)),
            )

    draw = ImageDraw.Draw(image)

    margin_left = 220
    margin_right = 220
    content_width = WIDTH - margin_left - margin_right

    label_font = load_font(
        FONT_CANDIDATES,
        42,
    )
    headline_font = load_font(
        FONT_CANDIDATES,
        96,
    )
    detail_font = load_font(
        REGULAR_FONT_CANDIDATES,
        54,
    )
    footer_font = load_font(
        REGULAR_FONT_CANDIDATES,
        30,
    )

    accent_width = 12
    accent_top = 260
    accent_bottom = 905

    draw.rounded_rectangle(
        (
            margin_left,
            accent_top,
            margin_left + accent_width,
            accent_bottom,
        ),
        radius=6,
        fill=(75, 182, 255),
    )

    headline, detail = narration_to_card_text(narration)

    text_left = margin_left + 65
    label_top = 275

    draw.text(
        (text_left, label_top),
        "NEWS IN FOCUS",
        font=label_font,
        fill=(122, 205, 255),
    )

    headline_top = label_top + 110

    detail_top = draw_wrapped_text(
        draw,
        position=(text_left, headline_top),
        text=headline,
        font=headline_font,
        fill=(246, 249, 252),
        max_width=content_width - 65,
        line_gap=24,
    )

    detail_top += 45

    draw_wrapped_text(
        draw,
        position=(text_left, detail_top),
        text=detail,
        font=detail_font,
        fill=(191, 206, 220),
        max_width=content_width - 65,
        line_gap=18,
    )

    footer = f"Scene {scene_number:02d} • News analysis"
    footer_box_top = HEIGHT - 160

    draw.rounded_rectangle(
        (
            margin_left,
            footer_box_top,
            WIDTH - margin_right,
            HEIGHT - 92,
        ),
        radius=18,
        fill=(8, 16, 26),
    )

    draw.text(
        (margin_left + 30, footer_box_top + 18),
        footer,
        font=footer_font,
        fill=(151, 177, 199),
    )

    image.save(
        target_path,
        format="PNG",
        optimize=True,
    )

    if not target_path.is_file() or target_path.stat().st_size == 0:
        raise RuntimeError(
            f"Could not create fallback asset: {target_path}"
        )

    return target_path


def main() -> int:
    args = parse_args()

    if args.run_id < 1:
        raise ValueError("--run-id must be positive")

    if args.max_assets_per_scene < 1:
        raise ValueError("--max-assets-per-scene must be at least 1")

    audio_manifest = load_json(
        args.audio_manifest,
        "audio manifest",
    )

    manifest_run_id = audio_manifest.get("run_id")

    if manifest_run_id != args.run_id:
        raise RuntimeError(
            "audio manifest run_id does not match requested run_id: "
            f"audio_manifest={manifest_run_id!r}, requested={args.run_id}"
        )

    audio_scenes = audio_manifest.get("scenes")

    if not isinstance(audio_scenes, list) or not audio_scenes:
        raise RuntimeError("audio manifest has no scenes")

    assets_rows = load_asset_rows(args.assets_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    render_scenes: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    ordered_audio_scenes = sorted(
        audio_scenes,
        key=lambda item: int(item["scene_number"]),
    )

    expected_scene_numbers = list(
        range(1, len(ordered_audio_scenes) + 1)
    )

    actual_scene_numbers = [
        int(item["scene_number"])
        for item in ordered_audio_scenes
    ]

    if actual_scene_numbers != expected_scene_numbers:
        raise RuntimeError(
            "Invalid audio scene order: "
            f"expected={expected_scene_numbers}, "
            f"actual={actual_scene_numbers}"
        )

    for audio_scene in ordered_audio_scenes:
        scene_number = int(audio_scene["scene_number"])

        selected, rejected = select_assets(
            scene_number=scene_number,
            rows=assets_rows,
            max_assets=args.max_assets_per_scene,
        )

        rejected_rows.extend(rejected)

        use_fallback = not selected

        audio_file = str(audio_scene.get("audio_file") or "").strip()

        if not audio_file:
            raise RuntimeError(
                f"audio_file is missing for scene {scene_number}"
            )

        audio_path = args.audio_manifest.parent / audio_file

        if not audio_path.is_file():
            raise RuntimeError(
                f"Audio file is missing for scene {scene_number}: {audio_path}"
            )

        actual_duration = audio_scene.get("actual_duration_seconds")

        if not isinstance(actual_duration, (int, float)):
            raise RuntimeError(
                "actual_duration_seconds is invalid for scene "
                f"{scene_number}: {actual_duration!r}"
            )

        scene_dir = args.output_dir / f"scene_{scene_number:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        copied_assets: list[str] = []

        if use_fallback:
            fallback_path = create_fallback_asset(
                target_dir=scene_dir,
                scene_number=scene_number,
                narration=str(audio_scene.get("narration") or ""),
            )

            copied_assets.append(str(fallback_path.resolve()))

            selection_rows.append(
                {
                    "scene_number": scene_number,
                    "asset_index": 1,
                    "source_asset_path": None,
                    "render_asset_path": str(fallback_path.resolve()),
                    "article_id": None,
                    "article_title": None,
                    "article_url": None,
                    "candidate_rank": None,
                    "is_representative": None,
                    "asset_type": "generated_editorial_card",
                    "asset_source": "narration_editorial_card",
                    "asset_url": None,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "fallback_reason": (
                        "No valid high-information source image was collected "
                        "for this scene."
                    ),
                }
            )

        else:
            for index, row in enumerate(selected, start=1):
                source_path = Path(str(row["asset_path"]))

                copied_path = copy_asset(
                    source_path=source_path,
                    target_dir=scene_dir,
                    index=index,
                )

                copied_assets.append(str(copied_path.resolve()))

                selection_rows.append(
                    {
                        "scene_number": scene_number,
                        "asset_index": index,
                        "source_asset_path": str(source_path),
                        "render_asset_path": str(copied_path.resolve()),
                        "article_id": row.get("article_id"),
                        "article_title": row.get("article_title"),
                        "article_url": row.get("article_url"),
                        "candidate_rank": row.get("candidate_rank"),
                        "is_representative": row.get("is_representative"),
                        "asset_type": row.get("asset_type"),
                        "asset_source": row.get("asset_source"),
                        "asset_url": row.get("asset_url"),
                        "width": row.get("width"),
                        "height": row.get("height"),
                        "fallback_reason": None,
                    }
                )

        render_scenes.append(
            {
                "scene_number": scene_number,
                "actual_duration_seconds": round(
                    float(actual_duration),
                    3,
                ),
                "audio_path": str(audio_path.resolve()),
                "assets": copied_assets,
            }
        )

    render_manifest = {
        "run_id": args.run_id,
        "canvas": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
        },
        "scene_count": len(render_scenes),
        "scenes": render_scenes,
    }

    render_manifest_path = args.output_dir / "render_manifest.json"

    render_manifest_path.write_text(
        json.dumps(
            render_manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    selection_path = args.output_dir / "selected_assets.json"

    selection_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "assets": selection_rows,
                "rejected_low_information_assets": rejected_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "render_manifest": str(render_manifest_path),
                "selected_assets": str(selection_path),
                "scene_count": len(render_scenes),
                "asset_count": len(selection_rows),
                "rejected_low_information_asset_count": len(rejected_rows),
            },
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())