from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2.extras

from clustering.offline import get_conn


log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_LOCK_KEY = 917_244_613

SCHEDULE_HOURS_UTC = {6, 18}

POLL_SECONDS = int(
    os.getenv("VIDEO_WORKER_POLL_SECONDS", "60")
)

TARGET_DURATION_SECONDS = int(
    os.getenv("VIDEO_TARGET_DURATION_SECONDS", "120")
)

MAX_TOPICS = int(
    os.getenv("VIDEO_MAX_TOPICS", "8")
)

HEADLINES_PER_TOPIC = int(
    os.getenv("VIDEO_HEADLINES_PER_TOPIC", "3")
)

UPLOAD_ENABLED = (
    os.getenv("YOUTUBE_UPLOAD_ENABLED", "false").casefold()
    == "true"
)

PRIVACY_STATUS = os.getenv(
    "YOUTUBE_PRIVACY_STATUS",
    "unlisted",
).strip()

RETENTION_HOURS = int(
    os.getenv("VIDEO_ARTIFACT_RETENTION_HOURS", "48")
)


def acquire_lock(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(%s)",
            (WORKER_LOCK_KEY,),
        )
        row = cur.fetchone()

    return bool(row[0])


def release_lock(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(%s)",
            (WORKER_LOCK_KEY,),
        )

    conn.commit()


def schedule_slot(now: datetime) -> str | None:
    if now.minute != 0:
        return None

    if now.hour not in SCHEDULE_HOURS_UTC:
        return None

    return now.strftime("%Y-%m-%dT%H:00:00Z")


def slot_already_processed(
    conn,
    *,
    slot: str,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM pipeline_runs
            WHERE job_type = 'video_schedule'
              AND meta->>'schedule_slot' = %s
              AND status IN ('success', 'no_run')
            LIMIT 1
            """,
            (slot,),
        )

        return cur.fetchone() is not None


def start_schedule_record(
    conn,
    *,
    slot: str,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_runs (
                job_type,
                status,
                meta
            )
            VALUES (
                'video_schedule',
                'running',
                %s
            )
            RETURNING id
            """,
            (
                psycopg2.extras.Json(
                    {
                        "schedule_slot": slot,
                        "upload_enabled": UPLOAD_ENABLED,
                        "privacy_status": PRIVACY_STATUS,
                    }
                ),
            ),
        )
        row = cur.fetchone()

    conn.commit()
    return int(row[0])


def finish_schedule_record(
    conn,
    *,
    record_id: int,
    status: str,
    run_id: int | None,
    error: str | None,
    meta: dict[str, Any],
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
                json.dumps(meta),
                record_id,
            ),
        )

    conn.commit()


def find_latest_eligible_run(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id
            FROM clustering_runs r
            WHERE r.status IN ('success', 'completed', 'degraded')
              AND r.finished_at IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM pipeline_runs p
                  WHERE p.job_type = 'video_pipeline'
                    AND p.related_run_id = r.id
                    AND p.status = 'success'
                    AND COALESCE(
                        p.meta->>'youtube_video_id',
                        ''
                    ) <> ''
              )
            ORDER BY r.finished_at DESC, r.id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()

    return int(row[0]) if row else None


def run_command(command: list[str]) -> None:
    log.info("$ %s", " ".join(command))

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_video_pipeline(run_id: int) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "scripts.run_video_pipeline",
            "--run-id",
            str(run_id),
            "--target-duration-seconds",
            str(TARGET_DURATION_SECONDS),
            "--max-topics",
            str(MAX_TOPICS),
            "--headlines-per-topic",
            str(HEADLINES_PER_TOPIC),
        ]
    )


def run_upload(run_id: int) -> None:
    final_path = (
        PROJECT_ROOT
        / "artifacts"
        / f"run_{run_id}"
        / "render"
        / f"final_{run_id}.mp4"
    )

    command = [
        sys.executable,
        "-m",
        "scripts.upload_youtube_video",
        "--run-id",
        str(run_id),
        "--video-path",
        str(final_path),
        "--privacy-status",
        PRIVACY_STATUS,
    ]

    if not UPLOAD_ENABLED:
        command.append("--dry-run")

    run_command(command)


def run_cleanup() -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "scripts.cleanup_video_artifacts",
            "--retention-hours",
            str(RETENTION_HOURS),
        ]
    )


def process_slot(slot: str) -> None:
    conn = get_conn()
    lock_acquired = False
    schedule_record_id: int | None = None
    run_id: int | None = None

    try:
        if not acquire_lock(conn):
            log.info("Video worker skipped: worker lock is busy")
            return

        lock_acquired = True

        if slot_already_processed(conn, slot=slot):
            log.info("Schedule slot already processed: %s", slot)
            return

        schedule_record_id = start_schedule_record(
            conn,
            slot=slot,
        )

        run_id = find_latest_eligible_run(conn)

        if run_id is None:
            finish_schedule_record(
                conn,
                record_id=schedule_record_id,
                status="no_run",
                run_id=None,
                error=None,
                meta={
                    "stage": "selection",
                    "reason": "No eligible completed clustering run",
                },
            )
            return

        run_video_pipeline(run_id)
        run_upload(run_id)
        run_cleanup()

        finish_schedule_record(
            conn,
            record_id=schedule_record_id,
            status="success",
            run_id=run_id,
            error=None,
            meta={
                "stage": "completed",
                "upload_enabled": UPLOAD_ENABLED,
            },
        )

    except Exception as exc:
        log.exception("Video worker slot failed: %s", slot)

        if schedule_record_id is not None:
            try:
                conn.rollback()

                finish_schedule_record(
                    conn,
                    record_id=schedule_record_id,
                    status="failed",
                    run_id=run_id,
                    error=f"{type(exc).__name__}: {exc}",
                    meta={"stage": "failed"},
                )
            except Exception:
                log.exception("Could not persist worker failure")

    finally:
        if lock_acquired:
            try:
                release_lock(conn)
            except Exception:
                log.exception("Could not release video worker lock")

        conn.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    log.info(
        "Video worker started: UTC schedule=%s, upload_enabled=%s",
        sorted(SCHEDULE_HOURS_UTC),
        UPLOAD_ENABLED,
    )

    last_checked_slot: str | None = None

    while True:
        now = datetime.now(timezone.utc)
        slot = schedule_slot(now)

        if slot and slot != last_checked_slot:
            process_slot(slot)
            last_checked_slot = slot

        if slot is None:
            last_checked_slot = None

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())