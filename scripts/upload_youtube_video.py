from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import psycopg2.extras
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from clustering.offline import get_conn


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


def load_credentials() -> Credentials:
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
) -> dict[str, Any]:
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

    if row is None:
        raise RuntimeError(
            f"No successful video_pipeline record for run_id={run_id}"
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
            """,
            (
                json.dumps(patch),
                pipeline_run_id,
            ),
        )

    conn.commit()


def resumable_upload(
    *,
    youtube,
    video_path: Path,
    metadata: dict[str, str],
    privacy_status: str,
    made_for_kids: bool,
) -> dict[str, Any]:
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

    response = None

    for attempt in range(1, 6):
        try:
            status, response = request.next_chunk()

            if status is not None:
                log.info(
                    "YouTube upload progress: %.1f%%",
                    status.progress() * 100,
                )

            if response is not None:
                return response

        except HttpError as exc:
            if exc.resp.status not in {500, 502, 503, 504}:
                raise

            if attempt == 5:
                raise

            delay = 2 ** attempt
            log.warning(
                "Retryable YouTube HTTP %s; retrying in %s seconds",
                exc.resp.status,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("YouTube resumable upload returned no response")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    args = parse_args()

    if args.run_id < 1:
        raise ValueError("--run-id must be positive")

    if not args.video_path.is_file():
        raise RuntimeError(f"Video file was not found: {args.video_path}")

    if args.video_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Video file is unexpectedly small: {args.video_path}"
        )

    conn = get_conn()

    try:
        pipeline_record = latest_video_pipeline_run(
            conn,
            run_id=args.run_id,
        )

        existing_video_id = (
            (pipeline_record["meta"] or {}).get("youtube_video_id")
        )

        if existing_video_id:
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": "already_uploaded",
                        "youtube_video_id": existing_video_id,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

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

        youtube = build(
            "youtube",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )

        response = resumable_upload(
            youtube=youtube,
            video_path=args.video_path,
            metadata=metadata,
            privacy_status=args.privacy_status,
            made_for_kids=args.made_for_kids,
        )

        youtube_video_id = str(response["id"])
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