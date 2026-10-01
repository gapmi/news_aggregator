from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


WIDTH = 1920
HEIGHT = 1080
FPS = 30

MAX_ASSETS_PER_SCENE = 3


ASSET_SOURCE_PRIORITY = {
    "rss_image": 0,
    "og_image": 1,
    "twitter_image": 2,
    "page_image": 3,
    "viewport_screenshot": 4,
}


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


def asset_sort_key(row: dict[str, str]) -> tuple[int, int, int, int, str]:
    asset_type = str(row.get("asset_type") or "").strip()
    source = str(row.get("asset_source") or "").strip()

    is_image = asset_type == "image"
    representative = is_true(row.get("is_representative"))

    return (
        0 if representative else 1,
        0 if is_image else 1,
        integer_value(row.get("candidate_rank"), 9999),
        ASSET_SOURCE_PRIORITY.get(source, 9999),
        str(row.get("asset_path") or ""),
    )


def select_assets(
    *,
    scene_number: int,
    rows: list[dict[str, str]],
    max_assets: int,
) -> list[dict[str, str]]:
    eligible: list[dict[str, str]] = []

    for row in rows:
        if integer_value(row.get("scene_number"), -1) != scene_number:
            continue

        if str(row.get("status") or "").strip() != "ok":
            continue

        if str(row.get("asset_type") or "").strip() not in {
            "image",
            "screenshot",
        }:
            continue

        asset_path = Path(str(row.get("asset_path") or "").strip())

        if not asset_path.is_file():
            continue

        if asset_path.stat().st_size == 0:
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

    return selected


def copy_asset(
    *,
    source_path: Path,
    target_dir: Path,
    index: int,
) -> Path:
    suffix = source_path.suffix.lower() or ".jpg"
    target_path = target_dir / f"asset_{index:02d}{suffix}"

    shutil.copy2(source_path, target_path)

    if not target_path.is_file() or target_path.stat().st_size == 0:
        raise RuntimeError(
            f"Could not copy selected asset: {source_path} → {target_path}"
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

        selected = select_assets(
            scene_number=scene_number,
            rows=assets_rows,
            max_assets=args.max_assets_per_scene,
        )

        if not selected:
            raise RuntimeError(
                "No valid collected asset exists for scene "
                f"{scene_number}. Check source_assets/attempts.csv."
            )

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
            },
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())