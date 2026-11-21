from __future__ import annotations

import argparse
import hashlib
import json
import logging

import math
import os
import shutil
import subprocess

import sys
import time
import csv
import psycopg2.extras

from pathlib import Path
from typing import Any
from scripts.video_date import (
    format_issue_date,
    gmt_date_from_timestamp,
)

from scripts.video_branding import (
    MEDIA_DIR,
    TRANSITION_SECONDS,
    media_duration,
)
from scripts.video_profiles import (
    DEFAULT_VIDEO_PROFILE,
    VIDEO_PROFILES,
    get_video_profile,
)
from fractions import Fraction
from scripts.video_fingerprint import (
    build_news_fingerprint,
    changed_components,
    check_saved_render,
)

log = logging.getLogger(__name__)

VIDEO_PIPELINE_ADVISORY_LOCK_KEY = 917_244_612

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

DEFAULT_TARGET_DURATION_SECONDS = 60
DEFAULT_MAX_TOPICS = 8
DEFAULT_HEADLINES_PER_TOPIC = 3
DEFAULT_MAX_ASSETS_PER_SCENE = 3

FINAL_WIDTH = 2400
FINAL_HEIGHT = 1350


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local news video pipeline for one completed "
            "clustering run."
        )
    )

    parser.add_argument("--run-id", type=int, required=True)

    parser.add_argument(
        "--profile",
        choices=sorted(VIDEO_PROFILES),
        default=DEFAULT_VIDEO_PROFILE,
        help="Video profile; horizontal is the default.",
    )

    parser.add_argument(
        "--existing-inputs-only",
        action="store_true",
        help=(
            "Use saved scene, audio and source-asset manifests only. "
            "Never generate scripts, TTS or collect source assets."
        ),
    )
    parser.add_argument(
        "--target-duration-seconds",
        type=int,
        default=DEFAULT_TARGET_DURATION_SECONDS,
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=DEFAULT_MAX_TOPICS,
    )
    parser.add_argument(
        "--headlines-per-topic",
        type=int,
        default=DEFAULT_HEADLINES_PER_TOPIC,
    )
    parser.add_argument(
        "--max-assets-per-scene",
        type=int,
        default=DEFAULT_MAX_ASSETS_PER_SCENE,
    )
    parser.add_argument(
        "--force-script",
        action="store_true",
        help=(
            "Generate a new Mistral script and regenerate downstream "
            "audio, assets and video."
        ),
    )
    parser.add_argument(
        "--force-tts",
        action="store_true",
        help="Regenerate Cartesia WAV files and final video.",
    )
    parser.add_argument(
        "--force-assets",
        action="store_true",
        help="Regenerate visual assets and final video.",
    )
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Rebuild final MP4.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Stop after render_manifest.json is successfully created.",
    )

    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    log.info("$ %s", " ".join(command))
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def probe_mp4(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration,size:"
                "stream=codec_name,codec_type,width,height,"
                "r_frame_rate,time_base"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def load_branding_duration(
    *,
    render_dir: Path,
    run_id: int,
) -> float | None:
    """Return added duration only for a matching branded render report."""
    report_path = render_dir / "render_report_v2.json"
    final_path = render_dir / f"final_{run_id}.mp4"

    try:
        if not report_path.is_file() or not final_path.is_file():
            return None

        if final_path.stat().st_size == 0:
            return None

        if report_path.stat().st_mtime_ns < final_path.stat().st_mtime_ns:
            return None

        report = json.loads(report_path.read_text(encoding="utf-8"))

        if not isinstance(report, dict):
            return None

        if report.get("run_id") != run_id:
            return None

        if report.get("final_video") != str(final_path):
            return None

        branding = report.get("branding")
        if not isinstance(branding, dict):
            return None

        if branding.get("enabled") is not True:
            return None

        scene_files = report.get("scene_files")
        sequence_files = report.get("sequence_files")

        if not isinstance(scene_files, list) or not scene_files:
            return None

        if not isinstance(sequence_files, list):
            return None

        scene_count = len(scene_files)
        transition_count = scene_count - 1

        if branding.get("transition_count") != transition_count:
            return None

        if len(sequence_files) != scene_count + transition_count + 1:
            return None

        intro_seconds = 0.0
        ending_seconds = round(
            media_duration(MEDIA_DIR / "news_ending.wav") * 30
        ) / 30

        expected_extra = (
            ending_seconds
            + transition_count * TRANSITION_SECONDS
        )

        checks = (
            (
                float(branding["intro_duration_seconds"]),
                intro_seconds,
            ),
            (
                float(branding["ending_duration_seconds"]),
                ending_seconds,
            ),
            (
                float(branding["transition_duration_seconds"]),
                TRANSITION_SECONDS,
            ),
            (
                float(branding["extra_duration_seconds"]),
                expected_extra,
            ),
            (
                float(report["branding_duration_seconds"]),
                expected_extra,
            ),
            (
                float(report["expected_duration_seconds"]),
                float(report["news_duration_seconds"]) + expected_extra,
            ),
        )

        for actual, expected in checks:
            if not math.isfinite(actual) or not math.isfinite(expected):
                return None
            if abs(actual - expected) > 0.01:
                return None

        return expected_extra

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        log.warning(
            "Cannot reuse branded render for run_id=%s: %s",
            run_id,
            exc,
        )
        return None


def validate_final_video(
    *,
    final_path: Path,
    expected_duration_seconds: float,
    expected_width: int = FINAL_WIDTH,
    expected_height: int = FINAL_HEIGHT,
    expected_fps: int = 30,
) -> dict[str, Any]:
    if not final_path.is_file():
        raise RuntimeError(f"Final video was not created: {final_path}")

    size_bytes = final_path.stat().st_size
    if size_bytes < 1_000_000:
        raise RuntimeError(
            f"Final MP4 is unexpectedly small: {size_bytes} bytes"
        )

    probe = probe_mp4(final_path)
    streams = probe.get("streams")

    if not isinstance(streams, list):
        raise RuntimeError("ffprobe did not return a streams array")

    video_stream = next(
        (
            stream for stream in streams
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    audio_stream = next(
        (
            stream for stream in streams
            if stream.get("codec_type") == "audio"
        ),
        None,
    )

    if video_stream is None:
        raise RuntimeError("Final MP4 has no video stream")
    if audio_stream is None:
        raise RuntimeError("Final MP4 has no audio stream")

    if video_stream.get("codec_name") != "h264":
        raise RuntimeError(
            "Final MP4 video codec must be h264; got "
            f"{video_stream.get('codec_name')!r}"
        )

    if audio_stream.get("codec_name") != "aac":
        raise RuntimeError(
            "Final MP4 audio codec must be aac; got "
            f"{audio_stream.get('codec_name')!r}"
        )

    if video_stream.get("width") != expected_width:
        raise RuntimeError(
            f"Final MP4 width must be {expected_width}; got "
            f"{video_stream.get('width')}"
        )

    if video_stream.get("height") != expected_height:
        raise RuntimeError(
            f"Final MP4 height must be {expected_height}; got "
            f"{video_stream.get('height')}"
        )
    try:
        actual_fps = Fraction(str(video_stream.get("r_frame_rate")))
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(
            "Final MP4 has an invalid frame rate"
        ) from exc

    if actual_fps != Fraction(expected_fps, 1):
        raise RuntimeError(
            f"Final MP4 frame rate must be {expected_fps}; "
            f"got {actual_fps}"
        )

    format_data = probe.get("format")
    if not isinstance(format_data, dict):
        raise RuntimeError("ffprobe did not return a format object")

    actual_duration = float(format_data.get("duration") or 0.0)

    if not math.isfinite(actual_duration) or actual_duration <= 0:
        raise RuntimeError("Final MP4 duration is invalid")

    if (
        not math.isfinite(expected_duration_seconds)
        or expected_duration_seconds <= 0
    ):
        raise RuntimeError("Expected video duration is invalid")

    duration_delta = abs(actual_duration - expected_duration_seconds)

    if duration_delta > 1.0:
        raise RuntimeError(
            "Final MP4 duration differs too much from audio duration: "
            f"actual={actual_duration:.3f}, "
            f"expected={expected_duration_seconds:.3f}, "
            f"delta={duration_delta:.3f}"
        )

    return {
        "probe": probe,
        "duration_seconds": round(actual_duration, 6),
        "duration_delta_seconds": round(duration_delta, 6),
        "size_bytes": size_bytes,
        "sha256": sha256_file(final_path),
    }

def get_conn():
    from clustering.offline import get_conn as open_connection

    return open_connection()

def try_acquire_lock(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(%s)",
            (VIDEO_PIPELINE_ADVISORY_LOCK_KEY,),
        )
        row = cur.fetchone()
    return bool(row[0])


def release_lock(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(%s)",
            (VIDEO_PIPELINE_ADVISORY_LOCK_KEY,),
        )
    conn.commit()


def validate_clustering_run(conn, run_id: int) -> str:
    with conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(
            """
            SELECT
                id,
                status,
                finished_at AT TIME ZONE 'UTC' AS finished_at
            FROM clustering_runs
            WHERE id = %s
            LIMIT 1
            """,
            (run_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"Clustering run was not found: run_id={run_id}"
        )

    if row["status"] not in {"success", "completed", "degraded"}:
        raise RuntimeError(
            "Clustering run is not eligible for video generation: "
            f"run_id={run_id}, status={row['status']!r}"
        )

    if row["finished_at"] is None:
        raise RuntimeError(
            f"Clustering run has no finished_at: run_id={run_id}"
        )

    return gmt_date_from_timestamp(row["finished_at"])


def find_existing_success(
    conn,
    *,
    run_id: int,
    profile_name: str,
) -> dict[str, Any] | None:
    with conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(
            """
            SELECT id, meta
            FROM pipeline_runs
            WHERE job_type = 'video_pipeline'
              AND related_run_id = %s
              AND status = 'success'
              AND COALESCE(
                    meta->>'profile',
                    'news_horizontal'
                  ) = %s
              AND meta->>'stage' IN (
                    'rendered',
                    'youtube_uploaded'
                  )
            ORDER BY finished_at DESC NULLS LAST, id DESC
            LIMIT 1
            """,
            (run_id, profile_name),
        )
        row = cur.fetchone()

    return dict(row) if row else None


def start_pipeline_record(
    conn,
    *,
    run_id: int,
    args: argparse.Namespace,
    artifact_root: Path,
) -> int:
    meta = {
        "stage": "starting",
        "run_id": run_id,
        "profile": args.profile,
        "artifact_root": str(artifact_root),
        "target_duration_seconds": args.target_duration_seconds,
        "max_topics": args.max_topics,
        "headlines_per_topic": args.headlines_per_topic,
        "max_assets_per_scene": args.max_assets_per_scene,
        "force_script": args.force_script,
        "force_tts": args.force_tts,
        "force_assets": args.force_assets,
        "force_render": args.force_render,
        "upload_mode": "none",
        "existing_inputs_only": args.existing_inputs_only,
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_runs (
                job_type,
                status,
                related_run_id,
                meta
            )
            VALUES ('video_pipeline', 'running', %s, %s)
            RETURNING id
            """,
            (run_id, psycopg2.extras.Json(meta)),
        )
        row = cur.fetchone()

    conn.commit()
    return int(row[0])


def update_pipeline_meta(
    conn,
    *,
    pipeline_run_id: int,
    patch: dict[str, Any],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb
            WHERE id = %s
            """,
            (json.dumps(patch), pipeline_run_id),
        )
    conn.commit()


def complete_pipeline_record(
    conn,
    *,
    pipeline_run_id: int,
    run_id: int,
    status: str,
    error: str | None,
    patch: dict[str, Any],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET
                status = %s,
                related_run_id = %s,
                error = %s,
                finished_at = NOW(),
                meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb
            WHERE id = %s
              AND status = 'running'
            """,
            (
                status,
                run_id,
                error,
                json.dumps(patch),
                pipeline_run_id,
            ),
        )

        if cur.rowcount != 1:
            raise RuntimeError(
                "Could not complete pipeline record: "
                f"pipeline_run_id={pipeline_run_id}"
            )

    conn.commit()


def write_pipeline_state(
    *,
    artifact_root: Path,
    payload: dict[str, Any],
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    path = artifact_root / ".pipeline_state.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def has_valid_mistral_script(
    conn,
    *,
    run_id: int,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM mistral_video_scripts
            WHERE run_id = %s
              AND status = 'success'
              AND validation_status = 'passed'
              AND script_json IS NOT NULL
            LIMIT 1
            """,
            (run_id,),
        )
        return cur.fetchone() is not None


def load_scene_count(scene_manifest_path: Path) -> int:
    if not scene_manifest_path.is_file():
        raise RuntimeError(
            f"Scene manifest was not created: {scene_manifest_path}"
        )

    data = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes")

    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError(
            f"Scene manifest has no scenes: {scene_manifest_path}"
        )

    actual_numbers = [
        int(scene["scene_number"]) for scene in scenes
    ]
    expected_numbers = list(range(1, len(scenes) + 1))

    if actual_numbers != expected_numbers:
        raise RuntimeError(
            "Scene manifest scene numbers are invalid: "
            f"expected={expected_numbers}, actual={actual_numbers}"
        )

    return len(scenes)


def has_valid_tts_output(
    *,
    audio_dir: Path,
    scene_number: int,
) -> bool:
    wav_path = audio_dir / f"scene_{scene_number:03d}.wav"
    metadata_path = audio_dir / f"scene_{scene_number:03d}_tts.json"

    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        return False
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        duration = float(metadata["actual_duration_seconds"])
    except Exception:
        return False

    return math.isfinite(duration) and duration > 0


def load_audio_duration(audio_manifest_path: Path) -> float:
    if not audio_manifest_path.is_file():
        raise RuntimeError(
            f"Audio manifest was not created: {audio_manifest_path}"
        )

    data = json.loads(audio_manifest_path.read_text(encoding="utf-8"))
    duration = data.get("total_actual_duration_seconds")

    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise RuntimeError(
            "audio_manifest total_actual_duration_seconds "
            "must be finite and positive"
        )

    return float(duration)

def validate_existing_inputs(
    *,
    run_id: int,
    scene_manifest_path: Path,
    audio_manifest_path: Path,
    assets_manifest_path: Path,
) -> dict[str, Any]:
    def read_object(path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required input missing or empty: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            raise RuntimeError(f"JSON root must be an object: {path}")

        return data

    def positive_number(value: Any, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise RuntimeError(f"{label} must be finite and positive")

        return float(value)

    def checked_scenes(
        data: dict[str, Any],
        label: str,
    ) -> list[dict[str, Any]]:
        if data.get("run_id") != run_id:
            raise RuntimeError(f"{label}: run_id mismatch")

        scenes = data.get("scenes")

        if not isinstance(scenes, list) or not scenes:
            raise RuntimeError(f"{label}: no scenes")

        if not all(isinstance(scene, dict) for scene in scenes):
            raise RuntimeError(f"{label}: invalid scene entries")

        numbers = [scene.get("scene_number") for scene in scenes]

        if (
            any(type(number) is not int for number in numbers)
            or numbers != list(range(1, len(scenes) + 1))
        ):
            raise RuntimeError(f"{label}: invalid scene order")

        if data.get("scene_count") != len(scenes):
            raise RuntimeError(f"{label}: scene_count mismatch")

        return scenes

    script = read_object(scene_manifest_path)
    audio = read_object(audio_manifest_path)

    script_scenes = checked_scenes(script, "scene manifest")
    audio_scenes = checked_scenes(audio, "audio manifest")

    if len(script_scenes) != len(audio_scenes):
        raise RuntimeError("Script and audio scene counts differ")

    audio_dir = audio_manifest_path.parent.resolve()
    total_duration = 0.0

    for script_scene, audio_scene in zip(script_scenes, audio_scenes):
        number = script_scene["scene_number"]
        label = f"Scene {number}"

        metadata_path = audio_dir / f"scene_{number:03d}_tts.json"
        metadata = read_object(metadata_path)

        for source in (audio_scene, metadata):
            if (
                source.get("run_id") != run_id
                or type(source.get("scene_number")) is not int
                or source["scene_number"] != number
            ):
                raise RuntimeError(f"{label}: TTS identity mismatch")

        narration = script_scene.get("narration")

        if not isinstance(narration, str) or not narration.strip():
            raise RuntimeError(f"{label}: missing narration")

        if (
            audio_scene.get("narration") != narration
            or metadata.get("narration") != narration
        ):
            raise RuntimeError(f"{label}: narration mismatch")

        target = positive_number(
            script_scene.get("target_duration_seconds"),
            f"{label} target duration",
        )

        for source in (audio_scene, metadata):
            source_target = positive_number(
                source.get("target_duration_seconds"),
                f"{label} TTS target duration",
            )
            if abs(source_target - target) > 0.001:
                raise RuntimeError(f"{label}: target duration mismatch")

        audio_file = audio_scene.get("audio_file")
        expected_file = f"scene_{number:03d}.wav"

        if (
            audio_file != expected_file
            or metadata.get("audio_file") != expected_file
        ):
            raise RuntimeError(f"{label}: unexpected audio filename")

        wav_path = audio_dir / expected_file

        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            raise RuntimeError(f"{label}: WAV missing or empty")

        duration = positive_number(
            audio_scene.get("actual_duration_seconds"),
            f"{label} audio duration",
        )
        metadata_duration = positive_number(
            metadata.get("actual_duration_seconds"),
            f"{label} metadata duration",
        )

        if abs(duration - metadata_duration) > 0.001:
            raise RuntimeError(f"{label}: metadata duration mismatch")

        wav_probe = probe_mp4(wav_path)

        if not any(
            stream.get("codec_type") == "audio"
            for stream in wav_probe.get("streams", [])
        ):
            raise RuntimeError(f"{label}: no audio stream in WAV")

        measured_duration = float(
            wav_probe.get("format", {}).get("duration") or 0
        )

        if (
            not math.isfinite(measured_duration)
            or measured_duration <= 0
            or abs(measured_duration - duration) > 0.1
        ):
            raise RuntimeError(
                f"{label}: WAV duration differs from metadata: "
                f"measured={measured_duration}, metadata={duration}"
            )

        total_duration += duration

    reported_total = positive_number(
        audio.get("total_actual_duration_seconds"),
        "Total audio duration",
    )

    if abs(total_duration - reported_total) > 0.01:
        raise RuntimeError("Audio total duration mismatch")

    if (
        not assets_manifest_path.is_file()
        or assets_manifest_path.stat().st_size == 0
    ):
        raise RuntimeError("Source assets manifest missing or empty")

    required_columns = {
        "scene_number",
        "candidate_rank",
        "is_representative",
        "asset_type",
        "asset_source",
        "asset_path",
        "status",
    }

    with assets_manifest_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise RuntimeError("Source assets manifest columns invalid")

        if next(reader, None) is None:
            raise RuntimeError("Source assets manifest contains no rows")

    return {
        "scene_count": len(script_scenes),
        "audio_duration_seconds": reported_total,
        "scene_manifest_sha256": sha256_file(scene_manifest_path),
        "audio_manifest_sha256": sha256_file(audio_manifest_path),
        "source_assets_manifest_sha256": sha256_file(
            assets_manifest_path
        ),
    }

def render_report_has_issue_date(
    *,
    render_dir: Path,
    issue_date_gmt: str,
) -> bool:
    try:
        report = json.loads(
            (render_dir / "render_report_v2.json").read_text(
                encoding="utf-8"
            )
        )

        return (
            isinstance(report, dict)
            and report.get("issue_date_gmt") == issue_date_gmt
            and report.get("issue_date_text")
            == format_issue_date(issue_date_gmt)
        )
    except (OSError, ValueError, TypeError):
        return False

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    args = parse_args()

    profile = get_video_profile(args.profile)

    if args.run_id < 1:
        raise ValueError("--run-id must be positive")
    if args.target_duration_seconds < 60:
        raise ValueError("--target-duration-seconds must be at least 60")
    if args.max_topics < 1:
        raise ValueError("--max-topics must be at least 1")
    if args.headlines_per_topic < 2:
        raise ValueError("--headlines-per-topic must be at least 2")
    if args.max_assets_per_scene < 1:
        raise ValueError("--max-assets-per-scene must be at least 1")
    if args.existing_inputs_only and (
        args.force_script
        or args.force_tts
        or args.force_assets
    ):
        raise ValueError(
            "--existing-inputs-only cannot be combined with "
            "--force-script, --force-tts or --force-assets"
        )

    if args.force_script:
        args.force_tts = True
        args.force_assets = True

    if (
        profile.name != DEFAULT_VIDEO_PROFILE
        and not args.existing_inputs_only
    ):
        raise ValueError(
            "Vertical rendering currently requires "
            "--existing-inputs-only"
        )
    
    if args.force_tts or args.force_assets:
        args.force_render = True

    started_monotonic = time.monotonic()

    run_root = ARTIFACTS_ROOT / f"run_{args.run_id}"

    artifact_root = (
        run_root
        if profile.name == DEFAULT_VIDEO_PROFILE
        else run_root / "vertical"
    )

    script_dir = run_root / "script"
    audio_dir = run_root / "audio"
    source_assets_dir = run_root / "source_assets"

    render_assets_dir = artifact_root / "render_assets"
    render_dir = artifact_root / "render"

    scene_manifest_path = script_dir / f"scene_manifest_{args.run_id}.json"
    audio_manifest_path = audio_dir / "audio_manifest.json"
    candidates_path = source_assets_dir / "candidates.csv"
    collected_assets_manifest_path = source_assets_dir / "manifest.csv"
    render_manifest_path = render_assets_dir / "render_manifest.json"
    final_video_path = render_dir / f"final_{args.run_id}.mp4"

    conn = get_conn()
    lock_acquired = False
    pipeline_run_id: int | None = None

    try:
        if not try_acquire_lock(conn):
            raise RuntimeError(
                "Video pipeline cannot start: "
                "another worker holds the video lock"
            )

        lock_acquired = True

        issue_date_gmt = validate_clustering_run(
            conn, args.run_id,
        )
        issue_date_text = format_issue_date(issue_date_gmt)

        if (
            args.existing_inputs_only
            and not args.skip_render
            and render_dir.exists()
            and any(render_dir.iterdir())
            and not args.force_render
        ):
            try:
                saved_fingerprint_path = (
                    render_dir / "render_fingerprint.json"
                )

                if not saved_fingerprint_path.is_file():
                    raise RuntimeError(
                        "Saved render fingerprint is missing"
                    )

                reuse_inputs = validate_existing_inputs(
                    run_id=args.run_id,
                    scene_manifest_path=scene_manifest_path,
                    audio_manifest_path=audio_manifest_path,
                    assets_manifest_path=collected_assets_manifest_path,
                )

                def current_reuse_fingerprint():
                    return build_news_fingerprint(
                        run_id=args.run_id,
                        issue_date_gmt=issue_date_gmt,
                        profile=profile,
                        project_root=PROJECT_ROOT,
                        scene_manifest_path=scene_manifest_path,
                        audio_manifest_path=audio_manifest_path,
                        assets_manifest_path=collected_assets_manifest_path,
                        media_dir=MEDIA_DIR,
                        max_assets_per_scene=args.max_assets_per_scene,
                    )

                reuse_fingerprint = current_reuse_fingerprint()

                matched, reason = check_saved_render(
                    record_path=saved_fingerprint_path,
                    final_path=final_video_path,
                    run_id=args.run_id,
                    profile_name=profile.name,
                    current_inputs=reuse_fingerprint,
                )

                if not matched:
                    raise RuntimeError(reason)

                reuse_branding_duration = load_branding_duration(
                    render_dir=render_dir,
                    run_id=args.run_id,
                )

                if reuse_branding_duration is None:
                    raise RuntimeError(
                        "Final video has no valid matching branding report"
                    )

                reuse_report = json.loads(
                    (render_dir / "render_report_v2.json").read_text(
                        encoding="utf-8"
                    )
                )

                if not isinstance(reuse_report, dict):
                    raise RuntimeError(
                        "Render report root must be an object"
                    )

                if (
                    reuse_report.get("run_id") != args.run_id
                    or reuse_report.get("profile") != profile.name
                ):
                    raise RuntimeError(
                        "Render report run or profile mismatch"
                    )

                if reuse_report.get("canvas") != {
                    "width": profile.width,
                    "height": profile.height,
                    "fps": profile.fps,
                }:
                    raise RuntimeError(
                        "Render report canvas mismatch"
                    )

                reuse_scene_files = reuse_report.get("scene_files")

                if (
                    not isinstance(reuse_scene_files, list)
                    or len(reuse_scene_files)
                    != reuse_inputs["scene_count"]
                ):
                    raise RuntimeError(
                        "Render report scene count mismatch"
                    )

                if not render_report_has_issue_date(
                    render_dir=render_dir,
                    issue_date_gmt=issue_date_gmt,
                ):
                    raise RuntimeError(
                        "Saved render issue date is missing or mismatched"
                    )

                reuse_news_duration = float(
                    reuse_report["news_duration_seconds"]
                )

                if (
                    not math.isfinite(reuse_news_duration)
                    or abs(
                        reuse_news_duration
                        - reuse_inputs["audio_duration_seconds"]
                    ) > 0.1
                ):
                    raise RuntimeError(
                        "Render report news duration mismatch"
                    )

                reuse_video_report = validate_final_video(
                    final_path=final_video_path,
                    expected_duration_seconds=(
                        reuse_inputs["audio_duration_seconds"]
                        + reuse_branding_duration
                    ),
                    expected_width=profile.width,
                    expected_height=profile.height,
                    expected_fps=profile.fps,
                )

                completed_reuse_fingerprint = (
                    current_reuse_fingerprint()
                )

                if completed_reuse_fingerprint != reuse_fingerprint:
                    raise RuntimeError(
                        "Render inputs changed during reuse checks"
                    )

                matched, reason = check_saved_render(
                    record_path=saved_fingerprint_path,
                    final_path=final_video_path,
                    run_id=args.run_id,
                    profile_name=profile.name,
                    current_inputs=completed_reuse_fingerprint,
                )

                if not matched:
                    raise RuntimeError(reason)

                if (
                    sha256_file(final_video_path)
                    != reuse_video_report["sha256"]
                ):
                    raise RuntimeError(
                        "Final video changed during reuse checks"
                    )

            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                raise RuntimeError(
                    f"Render directory is not empty: {render_dir}. "
                    f"Cannot reuse saved render: {exc}. "
                    "Use --force-render only if replacing this "
                    "profile's render output is intended."
                ) from exc

            log.info(
                "Reusing verified final video: run_id=%s "
                "profile=%s final_video=%s",
                args.run_id,
                profile.name,
                final_video_path,
            )
            return 0

        existing_success = find_existing_success(
            conn,
            run_id=args.run_id,
            profile_name=profile.name,
        )

        if (
            not args.existing_inputs_only
            and existing_success is not None
            and final_video_path.is_file()
            and final_video_path.stat().st_size > 0
            and not args.force_script
            and not args.force_tts
            and not args.force_assets
            and not args.force_render
            and load_branding_duration(
                render_dir=render_dir,
                run_id=args.run_id,
            ) is not None
        ):
            if not render_report_has_issue_date(
                render_dir=render_dir,
                issue_date_gmt=issue_date_gmt,
            ):
                raise RuntimeError(
                    "Existing final video has no matching GMT date. "
                    "Output was preserved. Rebuild explicitly only "
                    "after backing up this profile's render directory."
                )
            log.info(
                "Final video already exists for run_id=%s; "
                "pipeline_run_id=%s",
                args.run_id,
                existing_success["id"],
            )
            return 0

        for directory in (
            artifact_root,
            script_dir,
            audio_dir,
            source_assets_dir,
            render_assets_dir,
            render_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        pipeline_run_id = start_pipeline_record(
            conn,
            run_id=args.run_id,
            args=args,
            artifact_root=artifact_root,
        )

        write_pipeline_state(
            artifact_root=artifact_root,
            payload={
                "run_id": args.run_id,
                "profile": profile.name,
                "existing_inputs_only": args.existing_inputs_only,
                "pipeline_run_id": pipeline_run_id,
                "status": "running",
                "stage": "starting",
            },
        )

        if args.existing_inputs_only:
            update_pipeline_meta(
                conn,
                pipeline_run_id=pipeline_run_id,
                patch={"stage": "validating_existing_inputs"},
            )

            existing_inputs = validate_existing_inputs(
                run_id=args.run_id,
                scene_manifest_path=scene_manifest_path,
                audio_manifest_path=audio_manifest_path,
                assets_manifest_path=collected_assets_manifest_path,
            )

            scene_count = existing_inputs["scene_count"]
            audio_duration_seconds = existing_inputs[
                "audio_duration_seconds"
            ]
            scene_manifest_sha256 = existing_inputs[
                "scene_manifest_sha256"
            ]

            update_pipeline_meta(
                conn,
                pipeline_run_id=pipeline_run_id,
                patch={
                    "existing_inputs_only": True,
                    "inputs_validation": "passed",
                    "scene_count": scene_count,
                    "audio_duration_seconds": audio_duration_seconds,
                    "scene_manifest_path": str(scene_manifest_path),
                    "audio_manifest_path": str(audio_manifest_path),
                    "source_assets_manifest_path": str(
                        collected_assets_manifest_path
                    ),
                    "scene_manifest_sha256": scene_manifest_sha256,
                    "audio_manifest_sha256": existing_inputs[
                        "audio_manifest_sha256"
                    ],
                    "source_assets_manifest_sha256": existing_inputs[
                        "source_assets_manifest_sha256"
                    ],
                },
            )

        else:
            update_pipeline_meta(
                conn,
                pipeline_run_id=pipeline_run_id,
                patch={"stage": "mistral"},
            )

            script_generated = False

            if (
                args.force_script
                or not has_valid_mistral_script(
                    conn,
                    run_id=args.run_id,
                )
            ):
                run_command([
                    sys.executable,
                    "-m",
                    "clustering.mistral_generate",
                    "--run-id",
                    str(args.run_id),
                    "--target-duration-seconds",
                    str(args.target_duration_seconds),
                    "--max-topics",
                    str(args.max_topics),
                    "--headlines-per-topic",
                    str(args.headlines_per_topic),
                ])
                script_generated = True
            else:
                log.info(
                    "Using existing validated Mistral script: run_id=%s",
                    args.run_id,
                )

            update_pipeline_meta(
                conn,
                pipeline_run_id=pipeline_run_id,
                patch={"stage": "scene_manifest"},
            )

            previous_scene_manifest_sha256 = (
                sha256_file(scene_manifest_path)
                if scene_manifest_path.is_file()
                else None
            )

            run_command([
                sys.executable,
                "-m",
                "scripts.export_scene_manifest",
                "--run-id",
                str(args.run_id),
                "--output",
                str(scene_manifest_path),
            ])

            scene_count = load_scene_count(scene_manifest_path)
            scene_manifest_sha256 = sha256_file(scene_manifest_path)
            scene_manifest_changed = (
                previous_scene_manifest_sha256 != scene_manifest_sha256
            )

            if script_generated or scene_manifest_changed:
                args.force_tts = True
                args.force_assets = True
                args.force_render = True

                log.info(
                    "Invalidating downstream artifacts: run_id=%s "
                    "script_generated=%s scene_manifest_changed=%s",
                    args.run_id,
                    script_generated,
                    scene_manifest_changed,
                )

            update_pipeline_meta(
                conn,
                pipeline_run_id=pipeline_run_id,
                patch={
                    "stage": "tts",
                    "scene_count": scene_count,
                    "scene_manifest_path": str(scene_manifest_path),
                    "scene_manifest_sha256": scene_manifest_sha256,
                    "script_generated": script_generated,
                    "scene_manifest_changed": scene_manifest_changed,
                    "effective_force_tts": args.force_tts,
                    "effective_force_assets": args.force_assets,
                    "effective_force_render": args.force_render,
                },
            )

            for scene_number in range(1, scene_count + 1):
                if (
                    not args.force_tts
                    and has_valid_tts_output(
                        audio_dir=audio_dir,
                        scene_number=scene_number,
                    )
                ):
                    log.info(
                        "Using existing TTS output: run_id=%s scene=%s",
                        args.run_id,
                        scene_number,
                    )
                    continue

                run_command([
                    sys.executable,
                    "-m",
                    "scripts.cartesia_tts_pilot",
                    "--manifest",
                    str(scene_manifest_path),
                    "--scene-number",
                    str(scene_number),
                    "--output-dir",
                    str(audio_dir),
                ])

            run_command([
                sys.executable,
                "-m",
                "scripts.build_audio_manifest",
                "--run-id",
                str(args.run_id),
                "--audio-dir",
                str(audio_dir),
            ])

            audio_duration_seconds = load_audio_duration(
                audio_manifest_path
            )

            update_pipeline_meta(
                conn,
                pipeline_run_id=pipeline_run_id,
                patch={
                    "stage": "asset_candidates",
                    "audio_manifest_path": str(audio_manifest_path),
                    "audio_duration_seconds": audio_duration_seconds,
                },
            )

            if args.force_assets or not candidates_path.is_file():
                run_command([
                    sys.executable,
                    "-m",
                    "scripts.export_scene_asset_candidates",
                    "--run-id",
                    str(args.run_id),
                    "--output-csv",
                    str(candidates_path),
                ])
            else:
                log.info(
                    "Using existing asset candidates: run_id=%s",
                    args.run_id,
                )

            update_pipeline_meta(
                conn,
                pipeline_run_id=pipeline_run_id,
                patch={"stage": "asset_collection"},
            )

            if (
                args.force_assets
                or not collected_assets_manifest_path.is_file()
                or collected_assets_manifest_path.stat().st_size == 0
            ):
                asset_env = os.environ.copy()
                asset_env["ASSETS_INPUT_CSV"] = str(candidates_path)
                asset_env["ASSETS_OUTPUT_DIR"] = str(source_assets_dir)

                run_command(
                    [
                        sys.executable,
                        "-m",
                        "clustering.collect_cluster_assets",
                    ],
                    env=asset_env,
                )
            else:
                log.info(
                    "Using existing asset collection manifest: run_id=%s",
                    args.run_id,
                )

        input_fingerprint = None

        if args.existing_inputs_only:
            input_fingerprint = build_news_fingerprint(
                run_id=args.run_id,
                issue_date_gmt=issue_date_gmt,
                profile=profile,
                project_root=PROJECT_ROOT,
                scene_manifest_path=scene_manifest_path,
                audio_manifest_path=audio_manifest_path,
                assets_manifest_path=collected_assets_manifest_path,
                media_dir=MEDIA_DIR,
                max_assets_per_scene=args.max_assets_per_scene,
            )

        update_pipeline_meta(
            conn,
            pipeline_run_id=pipeline_run_id,
            patch={"stage": "render_manifest"},
        )

        if args.force_assets and render_assets_dir.exists():
            shutil.rmtree(render_assets_dir)
            render_assets_dir.mkdir(parents=True, exist_ok=True)

        run_command([
            sys.executable,
            "-m",
            "scripts.build_render_manifest",
            "--run-id",
            str(args.run_id),
            "--issue-date-gmt",
            issue_date_gmt,
            "--profile",
            profile.name,
            "--audio-manifest",
            str(audio_manifest_path),
            "--assets-manifest",
            str(collected_assets_manifest_path),
            "--output-dir",
            str(render_assets_dir),
            "--max-assets-per-scene",
            str(args.max_assets_per_scene),
        ])

        if not render_manifest_path.is_file():
            raise RuntimeError(
                f"Render manifest was not created: {render_manifest_path}"
            )

        if args.skip_render:
            complete_pipeline_record(
                conn,
                pipeline_run_id=pipeline_run_id,
                run_id=args.run_id,
                status="success",
                error=None,
                patch={
                    "stage": "render_manifest_ready",
                    "render_manifest_path": str(render_manifest_path),
                    "render_skipped": True,
                    "elapsed_seconds": round(
                        time.monotonic() - started_monotonic, 3
                    ),
                },
            )
            write_pipeline_state(
                artifact_root=artifact_root,
                payload={
                    "run_id": args.run_id,
                    "profile": profile.name,
                    "existing_inputs_only": args.existing_inputs_only,
                    "pipeline_run_id": pipeline_run_id,
                    "status": "success",
                    "stage": "render_manifest_ready",
                    "render_skipped": True,
                },
            )
            log.info(
                "Video pipeline stopped after render manifest: run_id=%s",
                args.run_id,
            )
            return 0

        update_pipeline_meta(
            conn,
            pipeline_run_id=pipeline_run_id,
            patch={"stage": "render"},
        )

        if args.force_render and render_dir.exists():
            shutil.rmtree(render_dir)
            render_dir.mkdir(parents=True, exist_ok=True)

        rendered_now = False

        if (
            args.existing_inputs_only
            or args.force_render
            or not final_video_path.is_file()
            or final_video_path.stat().st_size == 0
            or not render_report_has_issue_date(
                render_dir=render_dir,
                issue_date_gmt=issue_date_gmt,
            )
            or load_branding_duration(
                render_dir=render_dir,
                run_id=args.run_id,
            ) is None
        ):
            run_command([
                sys.executable,
                "-m",
                "scripts.render_news_video_v2",
                "--manifest",
                str(render_manifest_path),
                "--output-dir",
                str(render_dir),
            ])
            rendered_now = True
        else:
            log.info(
                "Using existing final MP4: run_id=%s",
                args.run_id,
            )

        update_pipeline_meta(
            conn,
            pipeline_run_id=pipeline_run_id,
            patch={"stage": "quality_gate"},
        )

        branding_duration_seconds = load_branding_duration(
            render_dir=render_dir,
            run_id=args.run_id,
        )

        if branding_duration_seconds is None:
            raise RuntimeError(
                "Final video has no valid matching branding report"
            )

        render_report_path = render_dir / "render_report_v2.json"
        render_report = json.loads(
            render_report_path.read_text(encoding="utf-8")
        )

        if not isinstance(render_report, dict):
            raise RuntimeError("Render report root must be an object")
        if not render_report_has_issue_date(
            render_dir=render_dir,
            issue_date_gmt=issue_date_gmt,
        ):
            raise RuntimeError(
                "Final render report issue date is missing or mismatched"
            )
        report_profile = render_report.get(
            "profile",
            DEFAULT_VIDEO_PROFILE,
        )

        if report_profile != profile.name:
            raise RuntimeError(
                "Render report profile mismatch: "
                f"report={report_profile!r}, "
                f"expected={profile.name!r}"
            )

        expected_canvas = {
            "width": profile.width,
            "height": profile.height,
            "fps": profile.fps,
        }

        report_canvas = render_report.get("canvas")

        if (
            report_canvas is not None
            or profile.name != DEFAULT_VIDEO_PROFILE
        ):
            if report_canvas != expected_canvas:
                raise RuntimeError(
                    "Render report canvas mismatch: "
                    f"report={report_canvas!r}, "
                    f"expected={expected_canvas!r}"
                )

        reported_scene_count = len(render_report["scene_files"])
        if reported_scene_count != scene_count:
            raise RuntimeError(
                "Render report scene count differs from pipeline: "
                f"report={reported_scene_count}, pipeline={scene_count}"
            )

        reported_news_duration = float(
            render_report["news_duration_seconds"]
        )

        if (
            not math.isfinite(reported_news_duration)
            or abs(reported_news_duration - audio_duration_seconds) > 0.1
        ):
            raise RuntimeError(
                "Render report news duration differs from audio manifest: "
                f"report={reported_news_duration:.6f}, "
                f"audio={audio_duration_seconds:.6f}"
            )

        expected_final_duration_seconds = (
            audio_duration_seconds + branding_duration_seconds
        )

        final_video_report = validate_final_video(
            final_path=final_video_path,
            expected_duration_seconds=expected_final_duration_seconds,
            expected_width=profile.width,
            expected_height=profile.height,
            expected_fps=profile.fps,
        )

        if input_fingerprint is not None and rendered_now:
            completed_fingerprint = build_news_fingerprint(
                run_id=args.run_id,
                issue_date_gmt=issue_date_gmt,
                profile=profile,
                project_root=PROJECT_ROOT,
                scene_manifest_path=scene_manifest_path,
                audio_manifest_path=audio_manifest_path,
                assets_manifest_path=collected_assets_manifest_path,
                media_dir=MEDIA_DIR,
                max_assets_per_scene=args.max_assets_per_scene,
            )

            if (
                completed_fingerprint["sha256"]
                != input_fingerprint["sha256"]
            ):
                changes = changed_components(
                    input_fingerprint,
                    completed_fingerprint,
                )
                raise RuntimeError(
                    "Render inputs changed during production: "
                    + ", ".join(changes)
                )

            fingerprint_record = {
                "record_version": 1,
                "run_id": args.run_id,
                "profile": profile.name,
                "inputs": input_fingerprint,
                "output": {
                    "filename": final_video_path.name,
                    "sha256": final_video_report["sha256"],
                    "size_bytes": final_video_report["size_bytes"],
                    "duration_seconds": final_video_report[
                        "duration_seconds"
                    ],
                },
            }

            fingerprint_path = render_dir / "render_fingerprint.json"
            temporary_fingerprint_path = (
                render_dir / ".render_fingerprint.tmp.json"
            )

            try:
                temporary_fingerprint_path.write_text(
                    json.dumps(
                        fingerprint_record,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                temporary_fingerprint_path.replace(fingerprint_path)
            finally:
                temporary_fingerprint_path.unlink(missing_ok=True)

        success_meta = {
            "stage": "rendered",
            "profile": profile.name,
            "existing_inputs_only": args.existing_inputs_only,
            "scene_manifest_sha256": scene_manifest_sha256,
            "branding_duration_seconds": branding_duration_seconds,
            "expected_final_duration_seconds": expected_final_duration_seconds,
            "render_report_path": str(render_report_path),
            "scene_manifest_path": str(scene_manifest_path),
            "audio_manifest_path": str(audio_manifest_path),
            "source_assets_manifest_path": str(collected_assets_manifest_path),
            "render_manifest_path": str(render_manifest_path),
            "final_video_path": str(final_video_path),
            "final_duration_seconds": final_video_report["duration_seconds"],
            "duration_delta_seconds": final_video_report["duration_delta_seconds"],
            "final_video_size_bytes": final_video_report["size_bytes"],
            "final_video_sha256": final_video_report["sha256"],
            "upload_mode": "none",
            "youtube_video_id": None,
            "youtube_url": None,
            "elapsed_seconds": round(
                time.monotonic() - started_monotonic, 3
            ),
            "issue_date_gmt": issue_date_gmt,
            "issue_date_text": issue_date_text,
        }

        complete_pipeline_record(
            conn,
            pipeline_run_id=pipeline_run_id,
            run_id=args.run_id,
            status="success",
            error=None,
            patch=success_meta,
        )

        write_pipeline_state(
            artifact_root=artifact_root,
            payload={
                "run_id": args.run_id,
                "profile": profile.name,
                "existing_inputs_only": args.existing_inputs_only,
                "pipeline_run_id": pipeline_run_id,
                "status": "success",
                "stage": "rendered",
                "final_video_path": str(final_video_path),
                "final_duration_seconds": final_video_report[
                    "duration_seconds"
                ],
                "final_video_sha256": final_video_report["sha256"],
                "scene_manifest_sha256": scene_manifest_sha256,
                "upload_mode": "none",
            },
        )

        log.info(
            "Video pipeline finished: run_id=%s final_video=%s",
            args.run_id,
            final_video_path,
        )
        return 0

    except Exception as exc:
        log.exception("Video pipeline failed: run_id=%s", args.run_id)

        if pipeline_run_id is not None:
            try:
                conn.rollback()
                complete_pipeline_record(
                    conn,
                    pipeline_run_id=pipeline_run_id,
                    run_id=args.run_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    patch={
                        "stage": "failed",
                        "error_type": type(exc).__name__,
                        "elapsed_seconds": round(
                            time.monotonic() - started_monotonic, 3
                        ),
                    },
                )
            except Exception:
                log.exception("Could not write failed pipeline status")

        if pipeline_run_id is not None:
            write_pipeline_state(
                artifact_root=artifact_root,
                payload={
                    "run_id": args.run_id,
                    "profile": profile.name,
                    "existing_inputs_only": args.existing_inputs_only,
                    "pipeline_run_id": pipeline_run_id,
                    "status": "failed",
                    "stage": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        return 1

    finally:
        if lock_acquired:
            try:
                conn.rollback()
                release_lock(conn)
            except Exception:
                log.exception(
                    "Could not release video pipeline advisory lock"
                )
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
