from __future__ import annotations
from scripts.video_profiles import VideoProfile

import hashlib
import json
import csv
import subprocess

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

FINGERPRINT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_fingerprint(
    *,
    settings: Mapping[str, Any],
    files: Mapping[str, Path],
) -> dict[str, Any]:
    """
    Build a read-only snapshot of settings and file contents.

    File keys are logical names, not absolute filesystem paths.
    Missing files are recorded explicitly. Other read errors propagate.
    """
    file_records: dict[str, dict[str, Any]] = {}

    for name, path in sorted(files.items()):
        try:
            digest = sha256_file(path)
        except FileNotFoundError:
            file_records[name] = {
                "status": "missing",
            }
        else:
            file_records[name] = {
                "status": "present",
                "sha256": digest,
            }

    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "settings": dict(settings),
        "files": file_records,
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return {
        "sha256": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
        "payload": json.loads(serialized),
    }


def changed_components(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    changes: list[str] = []

    for section in ("settings", "files"):
        old_values = previous["payload"][section]
        new_values = current["payload"][section]

        for name in sorted(set(old_values) | set(new_values)):
            if (
                (name in old_values) != (name in new_values)
                or old_values.get(name) != new_values.get(name)
            ):
                changes.append(f"{section}.{name}")

    if (
        previous["payload"]["schema_version"]
        != current["payload"]["schema_version"]
    ):
        changes.append("schema_version")

    return changes

def build_news_fingerprint(
    *,
    run_id: int,
    profile: VideoProfile,
    project_root: Path,
    scene_manifest_path: Path,
    audio_manifest_path: Path,
    assets_manifest_path: Path,
    media_dir: Path,
    max_assets_per_scene: int,
) -> dict[str, Any]:
    """
    Snapshot news inputs conservatively.

    Includes all source-image candidates for this run's scenes, not only
    selected photos, because changes can affect quality filtering/selection.
    """
    audio = json.loads(
        audio_manifest_path.read_text(encoding="utf-8")
    )
    scenes = audio["scenes"]
    scene_numbers = {
        int(scene["scene_number"])
        for scene in scenes
    }

    files: dict[str, Path] = {
        "manifest.script": scene_manifest_path,
        "manifest.audio": audio_manifest_path,
        "manifest.source_assets": assets_manifest_path,
        "maps.geojson": (
            project_root
            / "media"
            / "maps"
            / "ne_110m_admin_0_countries.geojson"
        ),
    }

    for scene in scenes:
        number = int(scene["scene_number"])
        audio_file = scene["audio_file"]

        files[f"audio.scene_{number:03d}"] = (
            audio_manifest_path.parent / audio_file
        )
        files[f"tts.scene_{number:03d}"] = (
            audio_manifest_path.parent
            / f"scene_{number:03d}_tts.json"
        )

    with assets_manifest_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        for row_number, row in enumerate(
            csv.DictReader(file),
            start=1,
        ):
            try:
                scene_number = int(row.get("scene_number") or "")
            except (TypeError, ValueError):
                continue

            if scene_number not in scene_numbers:
                continue

            if str(row.get("status") or "").strip() != "ok":
                continue

            if str(row.get("asset_type") or "").strip() != "image":
                continue

            raw_path = str(row.get("asset_path") or "").strip()

            if not raw_path:
                continue

            path = Path(raw_path)

            if not path.is_absolute():
                path = project_root / path

            files[f"source.row_{row_number:06d}"] = path

    for filename in (
        "video.mp4",
        "news_intro.wav",
        "news_transition.wav",
        "news_ending.wav",
    ):
        files[f"branding.{filename}"] = media_dir / filename

    for filename in (
        "build_render_manifest.py",
        "render_news_video_v2.py",
        "video_branding.py",
        "video_profiles.py",
        "map_fallback.py",
        "map_regions.py",
        "scene_geography.py",
        "video_fingerprint.py",
    ):
        files[f"code.{filename}"] = (
            project_root / "scripts" / filename
        )

    font_root = Path("/usr/share/fonts")

    if font_root.is_dir():
        for path in sorted(font_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".ttf",
                ".otf",
                ".ttc",
            }:
                name = path.relative_to(font_root).as_posix()
                files[f"font.{name}"] = path

    ffmpeg = subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        capture_output=True,
        text=True,
    )

    settings = {
        "content_type": "news",
        "run_id": run_id,
        "profile": asdict(profile),
        "max_assets_per_scene": max_assets_per_scene,
        "branding_enabled": True,
        "pillow_version": version("Pillow"),
        "ffmpeg_version": ffmpeg.stdout.strip(),
    }

    return build_fingerprint(
        settings=settings,
        files=files,
    )
def check_saved_render(
    *,
    record_path: Path,
    final_path: Path,
    run_id: int,
    profile_name: str,
    current_inputs: dict[str, Any],
) -> tuple[bool, str]:
    """
    Check saved input identity and MP4 content identity.

    This does not replace the render-report and media quality checks.
    """
    try:
        record = json.loads(
            record_path.read_text(encoding="utf-8")
        )

        if not isinstance(record, dict):
            return False, "Fingerprint record is not an object"

        if record.get("record_version") != 1:
            return False, "Unsupported fingerprint record version"

        if (
            record.get("run_id") != run_id
            or record.get("profile") != profile_name
        ):
            return False, "Fingerprint run or profile mismatch"

        if record.get("inputs") != current_inputs:
            return False, "Render inputs differ from saved fingerprint"

        output = record.get("output")

        if not isinstance(output, dict):
            return False, "Fingerprint output record is invalid"

        if output.get("filename") != final_path.name:
            return False, "Final video filename mismatch"

        if not final_path.is_file():
            return False, "Final video is missing"

        size_bytes = final_path.stat().st_size

        if size_bytes <= 0:
            return False, "Final video is empty"

        if output.get("size_bytes") != size_bytes:
            return False, "Final video size mismatch"

        if output.get("sha256") != sha256_file(final_path):
            return False, "Final video SHA256 mismatch"

    except (OSError, ValueError) as exc:
        return False, (
            "Cannot verify saved render: "
            f"{type(exc).__name__}: {exc}"
        )

    return True, "Saved inputs and final video match"