from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psycopg2.extras

from clustering.offline import get_conn


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
        help="Generate a new Mistral script even if a validated script exists.",
    )

    parser.add_argument(
        "--force-tts",
        action="store_true",
        help="Regenerate Cartesia WAV files for every scene.",
    )

    parser.add_argument(
        "--force-assets",
        action="store_true",
        help="Regenerate article candidates and recollect visual assets.",
    )

    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Rebuild render assets and final MP4.",
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


def validate_final_video(
    *,
    final_path: Path,
    expected_duration_seconds: float,
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
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
        ),
        None,
    )

    audio_stream = next(
        (
            stream
            for stream in streams
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

    if video_stream.get("width") != FINAL_WIDTH:
        raise RuntimeError(
            f"Final MP4 width must be {FINAL_WIDTH}; got "
            f"{video_stream.get('width')}"
        )

    if video_stream.get("height") != FINAL_HEIGHT:
        raise RuntimeError(
            f"Final MP4 height must be {FINAL_HEIGHT}; got "
            f"{video_stream.get('height')}"
        )

    format_data = probe.get("format")

    if not isinstance(format_data, dict):
        raise RuntimeError("ffprobe did not return a format object")

    actual_duration = float(format_data.get("duration") or 0.0)

    if actual_duration <= 0:
        raise RuntimeError("Final MP4 duration is invalid")

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


def validate_clustering_run(conn, run_id: int) -> None:
    with conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(
            """
            SELECT id, status, finished_at
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


def find_existing_success(
    conn,
    *,
    run_id: int,
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
            ORDER BY finished_at DESC NULLS LAST, id DESC
            LIMIT 1
            """,
            (run_id,),
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
        "artifact_root": str(artifact_root),
        "target_duration_seconds": args.target_duration_seconds,
        "max_topics": args.max_topics,
        "headlines_per_topic": args.headlines_per_topic,
        "max_assets_per_scene": args.max_assets_per_scene,
        "upload_mode": "none",
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
            VALUES (
                'video_pipeline',
                'running',
                %s,
                %s
            )
            RETURNING id
            """,
            (
                run_id,
                psycopg2.extras.Json(meta),
            ),
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
            (
                json.dumps(patch),
                pipeline_run_id,
            ),
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
        int(scene["scene_number"])
        for scene in scenes
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
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        duration = float(metadata["actual_duration_seconds"])
    except Exception:
        return False

    return duration > 0


def load_audio_duration(audio_manifest_path: Path) -> float:
    if not audio_manifest_path.is_file():
        raise RuntimeError(
            f"Audio manifest was not created: {audio_manifest_path}"
        )

    data = json.loads(audio_manifest_path.read_text(encoding="utf-8"))
    duration = data.get("total_actual_duration_seconds")

    if not isinstance(duration, (int, float)):
        raise RuntimeError(
            "audio_manifest total_actual_duration_seconds is invalid"
        )

    if duration <= 0:
        raise RuntimeError(
            "audio_manifest total_actual_duration_seconds must be positive"
        )

    return float(duration)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    args = parse_args()

    if args.run_id < 1:
        raise ValueError("--run-id must be positive")

    if args.target_duration_seconds < 60:
        raise ValueError(
            "--target-duration-seconds must be at least 60"
        )

    if args.max_topics < 1:
        raise ValueError("--max-topics must be at least 1")

    if args.headlines_per_topic < 2:
        raise ValueError(
            "--headlines-per-topic must be at least 2"
        )

    if args.max_assets_per_scene < 1:
        raise ValueError(
            "--max-assets-per-scene must be at least 1"
        )

    started_monotonic = time.monotonic()

    artifact_root = ARTIFACTS_ROOT / f"run_{args.run_id}"

    script_dir = artifact_root / "script"
    audio_dir = artifact_root / "audio"
    source_assets_dir = artifact_root / "source_assets"
    render_assets_dir = artifact_root / "render_assets"
    render_dir = artifact_root / "render"

    scene_manifest_path = (
        script_dir / f"scene_manifest_{args.run_id}.json"
    )

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
            log.warning(
                "Video pipeline skipped: another worker holds the video lock"
            )
            return 0

        lock_acquired = True

        validate_clustering_run(conn, args.run_id)

        existing_success = find_existing_success(
            conn,
            run_id=args.run_id,
        )

        if (
            existing_success is not None
            and final_video_path.is_file()
            and final_video_path.stat().st_size > 0
            and not args.force_render
        ):
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
                "pipeline_run_id": pipeline_run_id,
                "status": "running",
                "stage": "starting",
            },
        )

        update_pipeline_meta(
            conn,
            pipeline_run_id=pipeline_run_id,
            patch={"stage": "mistral"},
        )

        if (
            args.force_script
            or not has_valid_mistral_script(
                conn,
                run_id=args.run_id,
            )
        ):
            run_command(
                [
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
                ]
            )
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

        run_command(
            [
                sys.executable,
                "-m",
                "scripts.export_scene_manifest",
                "--run-id",
                str(args.run_id),
                "--output",
                str(scene_manifest_path),
            ]
        )

        scene_count = load_scene_count(scene_manifest_path)

        update_pipeline_meta(
            conn,
            pipeline_run_id=pipeline_run_id,
            patch={
                "stage": "tts",
                "scene_count": scene_count,
                "scene_manifest_path": str(scene_manifest_path),
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

            run_command(
                [
                    sys.executable,
                    "-m",
                    "scripts.cartesia_tts_pilot",
                    "--manifest",
                    str(scene_manifest_path),
                    "--scene-number",
                    str(scene_number),
                    "--output-dir",
                    str(audio_dir),
                ]
            )

        run_command(
            [
                sys.executable,
                "-m",
                "scripts.build_audio_manifest",
                "--run-id",
                str(args.run_id),
                "--audio-dir",
                str(audio_dir),
            ]
        )

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
            run_command(
                [
                    sys.executable,
                    "-m",
                    "scripts.export_scene_asset_candidates",
                    "--run-id",
                    str(args.run_id),
                    "--output-csv",
                    str(candidates_path),
                ]
            )
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

        update_pipeline_meta(
            conn,
            pipeline_run_id=pipeline_run_id,
            patch={"stage": "render_manifest"},
        )

        if args.force_assets and render_assets_dir.exists():
            shutil.rmtree(render_assets_dir)
            render_assets_dir.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                sys.executable,
                "-m",
                "scripts.build_render_manifest",
                "--run-id",
                str(args.run_id),
                "--audio-manifest",
                str(audio_manifest_path),
                "--assets-manifest",
                str(collected_assets_manifest_path),
                "--output-dir",
                str(render_assets_dir),
                "--max-assets-per-scene",
                str(args.max_assets_per_scene),
            ]
        )

        if not render_manifest_path.is_file():
            raise RuntimeError(
                "Render manifest was not created: "
                f"{render_manifest_path}"
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
                        time.monotonic() - started_monotonic,
                        3,
                    ),
                },
            )

            write_pipeline_state(
                artifact_root=artifact_root,
                payload={
                    "run_id": args.run_id,
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

        if (
            args.force_render
            or not final_video_path.is_file()
            or final_video_path.stat().st_size == 0
        ):
            run_command(
                [
                    sys.executable,
                    "-m",
                    "scripts.render_news_video_v2",
                    "--manifest",
                    str(render_manifest_path),
                    "--output-dir",
                    str(render_dir),
                ]
            )
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

        final_video_report = validate_final_video(
            final_path=final_video_path,
            expected_duration_seconds=audio_duration_seconds,
        )

        success_meta = {
            "stage": "rendered",
            "scene_count": scene_count,
            "scene_manifest_path": str(scene_manifest_path),
            "audio_manifest_path": str(audio_manifest_path),
            "source_assets_manifest_path": str(
                collected_assets_manifest_path
            ),
            "render_manifest_path": str(render_manifest_path),
            "final_video_path": str(final_video_path),
            "final_duration_seconds": (
                final_video_report["duration_seconds"]
            ),
            "duration_delta_seconds": (
                final_video_report["duration_delta_seconds"]
            ),
            "final_video_size_bytes": (
                final_video_report["size_bytes"]
            ),
            "final_video_sha256": final_video_report["sha256"],
            "upload_mode": "none",
            "youtube_video_id": None,
            "youtube_url": None,
            "elapsed_seconds": round(
                time.monotonic() - started_monotonic,
                3,
            ),
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
                "pipeline_run_id": pipeline_run_id,
                "status": "success",
                "stage": "rendered",
                "final_video_path": str(final_video_path),
                "final_duration_seconds": (
                    final_video_report["duration_seconds"]
                ),
                "final_video_sha256": final_video_report["sha256"],
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
        log.exception(
            "Video pipeline failed: run_id=%s",
            args.run_id,
        )

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
                            time.monotonic() - started_monotonic,
                            3,
                        ),
                    },
                )
            except Exception:
                log.exception(
                    "Could not write failed pipeline status"
                )

        write_pipeline_state(
            artifact_root=artifact_root,
            payload={
                "run_id": args.run_id,
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
                release_lock(conn)
            except Exception:
                log.exception(
                    "Could not release video pipeline advisory lock"
                )

        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())