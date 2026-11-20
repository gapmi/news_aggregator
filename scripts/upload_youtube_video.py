from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import psycopg2.extras


log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKEN_FILE = Path(
    os.getenv(
        "YOUTUBE_TOKEN_FILE",
        str(PROJECT_ROOT / "secrets" / "youtube_token.json"),
    )
)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
]

DEFAULT_CATEGORY_ID = "25"
DEFAULT_PRIVACY_STATUS = "unlisted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload one rendered news video to YouTube."
    )
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["news_horizontal", "news_vertical"],
        default="news_horizontal",
    )
    parser.add_argument(
        "--privacy-status",
        choices=["private", "unlisted", "public"],
        default=DEFAULT_PRIVACY_STATUS,
    )
    parser.add_argument(
        "--made-for-kids",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()

def get_conn():
    from clustering.offline import get_conn as open_connection

    return open_connection()

def load_credentials() -> Any:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not TOKEN_FILE.is_file():
        raise RuntimeError(
            "YouTube OAuth token is missing. Run scripts.youtube_oauth first: "
            f"{TOKEN_FILE}"
        )

    credentials = Credentials.from_authorized_user_file(
        str(TOKEN_FILE),
        SCOPES,
    )

    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

        TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

        TOKEN_FILE.chmod(0o600)

    if not credentials.valid:
        raise RuntimeError(
            "YouTube OAuth token is invalid or expired. "
            "Run scripts.youtube_oauth again."
        )

    return credentials


def load_video_metadata(
    conn,
    *,
    run_id: int,
) -> dict[str, str]:
    with conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(
            """
            SELECT title, description
            FROM mistral_video_scripts
            WHERE run_id = %s
              AND status = 'success'
              AND validation_status = 'passed'
            LIMIT 1
            """,
            (run_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"No validated Mistral metadata for run_id={run_id}"
        )

    title = str(row["title"] or "").strip()
    description = str(row["description"] or "").strip()

    if not title or not description:
        raise RuntimeError(
            f"Title or description is empty for run_id={run_id}"
        )

    return {
        "title": title[:100],
        "description": description[:5000],
    }


def latest_video_pipeline_run(
    conn,
    *,
    run_id: int,
    profile: str = "news_horizontal",
) -> dict[str, Any]:
    if profile not in {"news_horizontal", "news_vertical"}:
        raise ValueError(f"Unsupported video profile: {profile}")

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
            (run_id, profile),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            "No successful video_pipeline record for "
            f"run_id={run_id}, profile={profile}"
        )

    return dict(row)


def save_upload_result(
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
              AND job_type = 'video_pipeline'
              AND meta->>'youtube_upload_state' = 'started'
              AND COALESCE(meta->>'youtube_video_id', '') = ''
            """,
            (
                json.dumps(patch),
                pipeline_run_id,
            ),
        )

        if cur.rowcount != 1:
            raise RuntimeError(
                "YouTube upload result was not saved: "
                "pipeline record missing or upload state changed; "
                "manual reconciliation required"
            )

    conn.commit()

def build_youtube_client(credentials):
    from googleapiclient.discovery import build

    return build(
        "youtube",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def upload_chunks(
    request,
    *,
    http_error_type,
    max_consecutive_errors: int = 5,
) -> dict[str, Any]:
    if max_consecutive_errors < 1:
        raise ValueError("max_consecutive_errors must be positive")

    consecutive_errors = 0

    while True:
        try:
            status, response = request.next_chunk()
        except http_error_type as exc:
            if exc.resp.status not in {500, 502, 503, 504}:
                raise

            consecutive_errors += 1

            if consecutive_errors >= max_consecutive_errors:
                raise

            delay = 2 ** consecutive_errors
            log.warning(
                "Retryable YouTube HTTP %s; retrying in %s seconds",
                exc.resp.status,
                delay,
            )
            time.sleep(delay)
            continue

        consecutive_errors = 0

        if status is not None:
            log.info(
                "YouTube upload progress: %.1f%%",
                status.progress() * 100,
            )

        if response is not None:
            return response

        if status is None:
            raise RuntimeError(
                "YouTube upload returned neither progress nor response"
            )


def resumable_upload(
    *,
    youtube,
    video_path: Path,
    metadata: dict[str, str],
    privacy_status: str,
    made_for_kids: bool,
) -> dict[str, Any]:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    body = {
        "snippet": {
            "title": metadata["title"],
            "description": metadata["description"],
            "categoryId": DEFAULT_CATEGORY_ID,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        chunksize=8 * 1024 * 1024,
        resumable=True,
        mimetype="video/mp4",
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    return upload_chunks(
        request,
        http_error_type=HttpError,
    )

def validate_upload_target(
    *,
    run_id: int,
    profile: str,
    video_path: Path,
    privacy_status: str,
) -> Path:
    if run_id < 1:
        raise ValueError("--run-id must be positive")

    if profile not in {"news_horizontal", "news_vertical"}:
        raise ValueError(f"Unsupported video profile: {profile}")

    if profile == "news_vertical" and privacy_status != "private":
        raise RuntimeError(
            "Vertical uploads require --privacy-status private"
        )

    run_root = PROJECT_ROOT / "artifacts" / f"run_{run_id}"

    if profile == "news_vertical":
        run_root = run_root / "vertical"

    expected_video_path = (
        run_root / "render" / f"final_{run_id}.mp4"
    )

    if video_path.resolve() != expected_video_path.resolve():
        raise RuntimeError(
            f"Video path does not match profile {profile}: "
            f"expected {expected_video_path}"
        )

    return expected_video_path


def acquire_upload_lock(
    conn,
    *,
    run_id: int,
    profile: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_try_advisory_lock(
                hashtext(%s),
                hashtext(%s)
            )
            """,
            ("news_youtube_upload", f"{run_id}:{profile}"),
        )
        row = cur.fetchone()

    if row is None or not row[0]:
        raise RuntimeError(
            f"Upload lock is busy: run_id={run_id}, profile={profile}"
        )

    conn.commit()


def inspect_upload_history(
    conn,
    *,
    run_id: int,
    profile: str,
) -> str | None:
    with conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(
            """
            SELECT id, meta
            FROM pipeline_runs
            WHERE job_type = 'video_pipeline'
              AND related_run_id = %s
              AND COALESCE(
                    meta->>'profile',
                    'news_horizontal'
                  ) = %s
            ORDER BY id DESC
            """,
            (run_id, profile),
        )
        rows = cur.fetchall()

    uploaded_ids: set[str] = set()

    for row in rows:
        meta = row["meta"] or {}
        video_id = str(meta.get("youtube_video_id") or "").strip()
        state = meta.get("youtube_upload_state")

        if state in {"started", "uncertain"} and not video_id:
            raise RuntimeError(
                "Unresolved YouTube upload; manual reconciliation required: "
                f"run_id={run_id}, profile={profile}, "
                f"pipeline_run_id={row['id']}"
            )

        if video_id:
            if (
                profile == "news_vertical"
                and meta.get("youtube_privacy_status") != "private"
            ):
                raise RuntimeError(
                    "Existing vertical upload is not recorded as private; "
                    "manual verification required: "
                    f"pipeline_run_id={row['id']}"
                )

            uploaded_ids.add(video_id)
        elif state == "uploaded":
            raise RuntimeError(
                "Upload state is uploaded but YouTube ID is missing: "
                f"pipeline_run_id={row['id']}"
            )

    if len(uploaded_ids) > 1:
        raise RuntimeError(
            "Multiple YouTube IDs for the same run/profile; "
            f"manual review required: {sorted(uploaded_ids)}"
        )

    return next(iter(uploaded_ids), None)


def mark_upload_started(
    conn,
    *,
    pipeline_run_id: int,
    run_id: int,
    profile: str,
    privacy_status: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET meta = COALESCE(meta, '{}'::jsonb)
                || jsonb_build_object(
                    'youtube_upload_state', 'started',
                    'youtube_upload_started_at', NOW(),
                    'youtube_requested_privacy_status', %s::text
                )
            WHERE id = %s
              AND job_type = 'video_pipeline'
              AND related_run_id = %s
              AND status = 'success'
              AND COALESCE(
                    meta->>'profile',
                    'news_horizontal'
                  ) = %s
              AND meta->>'stage' = 'rendered'
              AND COALESCE(meta->>'youtube_video_id', '') = ''
              AND COALESCE(meta->>'youtube_upload_state', '') = ''
            """,
            (
                privacy_status,
                pipeline_run_id,
                run_id,
                profile,
            ),
        )

        if cur.rowcount != 1:
            raise RuntimeError(
                "Cannot claim rendered video for upload; "
                "record changed or an upload was already attempted"
            )

    conn.commit()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    args = parse_args()

    validate_upload_target(
        run_id=args.run_id,
        profile=args.profile,
        video_path=args.video_path,
        privacy_status=args.privacy_status,
    )

    if not args.video_path.is_file():
        raise RuntimeError(f"Video file was not found: {args.video_path}")

    if args.video_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Video file is unexpectedly small: {args.video_path}"
        )

    conn = get_conn()

    try:
        acquire_upload_lock(
            conn,
            run_id=args.run_id,
            profile=args.profile,
        )

        existing_video_id = inspect_upload_history(
            conn,
            run_id=args.run_id,
            profile=args.profile,
        )

        if existing_video_id:
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "profile": args.profile,
                        "status": "already_uploaded",
                        "youtube_video_id": existing_video_id,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        pipeline_record = latest_video_pipeline_run(
            conn,
            run_id=args.run_id,
            profile=args.profile,
        )

        pipeline_meta = pipeline_record["meta"] or {}
        recorded_video_path = pipeline_meta.get("final_video_path")

        if (
            not isinstance(recorded_video_path, str)
            or not recorded_video_path.strip()
            or Path(recorded_video_path).resolve()
            != args.video_path.resolve()
        ):
            raise RuntimeError(
                "Requested video does not match the selected "
                "profile pipeline record"
            )


        metadata = load_video_metadata(
            conn,
            run_id=args.run_id,
        )

        dry_run_output = {
            "run_id": args.run_id,
            "video_path": str(args.video_path),
            "video_size_bytes": args.video_path.stat().st_size,
            "title": metadata["title"],
            "description": metadata["description"],
            "privacy_status": args.privacy_status,
            "made_for_kids": args.made_for_kids,
            "category_id": DEFAULT_CATEGORY_ID,
            "profile": args.profile,
        }

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        **dry_run_output,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        credentials = load_credentials()

        youtube = build_youtube_client(credentials)

        mark_upload_started(
            conn,
            pipeline_run_id=int(pipeline_record["id"]),
            run_id=args.run_id,
            profile=args.profile,
            privacy_status=args.privacy_status,
        )

        response = resumable_upload(
            youtube=youtube,
            video_path=args.video_path,
            metadata=metadata,
            privacy_status=args.privacy_status,
            made_for_kids=args.made_for_kids,
        )

        response_id = response.get("id")

        if not isinstance(response_id, str) or not response_id.strip():
            raise RuntimeError(
                "YouTube returned no valid video ID; "
                "upload requires manual reconciliation"
            )

        youtube_video_id = response_id.strip()

        log.info(
            "YouTube returned video ID: run_id=%s profile=%s id=%s",
            args.run_id,
            args.profile,
            youtube_video_id,
        )
        youtube_url = (
            f"https://www.youtube.com/watch?v={youtube_video_id}"
        )

        save_upload_result(
            conn,
            pipeline_run_id=int(pipeline_record["id"]),
            patch={
                "stage": "youtube_uploaded",
                "upload_mode": "youtube",
                "youtube_video_id": youtube_video_id,
                "youtube_url": youtube_url,
                "youtube_privacy_status": args.privacy_status,
                "youtube_uploaded_at": "database_now",
                "youtube_title": metadata["title"],
                "youtube_description": metadata["description"],
                "youtube_made_for_kids": args.made_for_kids,
                "youtube_upload_state": "uploaded",
            },
        )

        print(
            json.dumps(
                {
                    "status": "uploaded",
                    "run_id": args.run_id,
                    "youtube_video_id": youtube_video_id,
                    "youtube_url": youtube_url,
                    "privacy_status": args.privacy_status,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())