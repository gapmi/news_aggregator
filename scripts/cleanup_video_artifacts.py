from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import psycopg2.extras

from clustering.offline import get_conn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete old local video artifact directories only when their "
            "video pipeline is complete and no longer needs local retention."
        )
    )
    parser.add_argument(
        "--retention-hours",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def load_cleanup_candidates(
    conn,
    *,
    retention_hours: int,
) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            WITH latest_success AS (
                SELECT related_run_id
                FROM pipeline_runs
                WHERE job_type = 'video_pipeline'
                  AND status = 'success'
                ORDER BY finished_at DESC NULLS LAST, id DESC
                LIMIT 1
            ),
            ranked AS (
                SELECT
                    p.id,
                    p.related_run_id,
                    p.status,
                    p.started_at,
                    p.finished_at,
                    p.meta,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.related_run_id
                        ORDER BY p.finished_at DESC NULLS LAST, p.id DESC
                    ) AS row_number
                FROM pipeline_runs p
                WHERE p.job_type = 'video_pipeline'
                  AND p.status IN ('success', 'failed')
                  AND p.finished_at IS NOT NULL
            )
            SELECT
                r.id,
                r.related_run_id,
                r.status,
                r.started_at,
                r.finished_at,
                r.meta
            FROM ranked r
            WHERE r.row_number = 1
              AND r.finished_at <
                    NOW() - (%s * INTERVAL '1 hour')
              AND r.related_run_id <> COALESCE(
                    (SELECT related_run_id FROM latest_success),
                    -1
                  )
            ORDER BY r.finished_at ASC
            """,
            (retention_hours,),
        )

        rows = cur.fetchall()

    return [dict(row) for row in rows]


def is_safe_run_directory(
    *,
    run_dir: Path,
    expected_run_id: int,
) -> bool:
    if not run_dir.is_dir():
        return False

    if run_dir.name != f"run_{expected_run_id}":
        return False

    keep_path = run_dir / ".keep"

    if keep_path.exists():
        return False

    return True


def main() -> int:
    args = parse_args()

    if args.retention_hours < 1:
        raise ValueError("--retention-hours must be at least 1")

    conn = get_conn()

    try:
        candidates = load_cleanup_candidates(
            conn,
            retention_hours=args.retention_hours,
        )
    finally:
        conn.close()

    actions: list[dict[str, Any]] = []

    for candidate in candidates:
        run_id = int(candidate["related_run_id"])
        run_dir = ARTIFACTS_ROOT / f"run_{run_id}"

        action = {
            "pipeline_run_id": int(candidate["id"]),
            "clustering_run_id": run_id,
            "status": candidate["status"],
            "finished_at": (
                candidate["finished_at"].isoformat()
                if candidate["finished_at"] is not None
                else None
            ),
            "directory": str(run_dir),
            "eligible": is_safe_run_directory(
                run_dir=run_dir,
                expected_run_id=run_id,
            ),
            "deleted": False,
            "reason": None,
        }

        if not action["eligible"]:
            action["reason"] = (
                "Directory does not exist, has invalid name, "
                "or contains .keep marker"
            )
            actions.append(action)
            continue

        if args.dry_run:
            action["reason"] = "dry-run"
            actions.append(action)
            continue

        shutil.rmtree(run_dir)
        action["deleted"] = True
        action["reason"] = "deleted"
        actions.append(action)

    print(
        json.dumps(
            {
                "retention_hours": args.retention_hours,
                "dry_run": args.dry_run,
                "candidate_count": len(candidates),
                "actions": actions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())