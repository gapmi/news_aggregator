from __future__ import annotations

import argparse
import json
import logging

import psycopg2.extras

from clustering.lineage import (
    DEFAULT_MIN_OVERLAP_RATIO,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_SCORE_OVERLAP_WEIGHT,
    DEFAULT_SCORE_SIM_WEIGHT,
    build_candidates,
    delete_existing_lineage,
    save_lineage,
    select_mutual_best,
)
from clustering.mistral_payload import build_mistral_video_payload
from clustering.observability import (
    PipelineContext,
    PipelineStageError,
    capture_pipeline_error,
    emit_stage_event,
)
from clustering.offline import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_LARGEST_CLUSTER_RATIO,
    DEFAULT_MAX_PER_SOURCE,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_MIN_VALID_CLUSTER_COUNT,
    DEFAULT_WINDOW_HOURS,
    get_conn,
    run_clustering,
)
from clustering.radial_map import build_and_save_radial_maps_for_run


log = logging.getLogger(__name__)

DEFAULT_VIDEO_TARGET_DURATION_SECONDS = 120


def start_pipeline_run(conn, job_type: str, meta: dict | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_runs (job_type, status, meta)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (
                job_type,
                "running",
                psycopg2.extras.Json(meta or {}),
            ),
        )
        row = cur.fetchone()

    conn.commit()
    return int(row[0])


def finish_pipeline_run(
    conn,
    pipeline_run_id: int,
    status: str,
    related_run_id: int | None = None,
    error: str | None = None,
    meta: dict | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET
                status = %s,
                finished_at = NOW(),
                related_run_id = %s,
                error = %s,
                meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb
            WHERE id = %s
            """,
            (
                status,
                related_run_id,
                error,
                json.dumps(meta or {}),
                pipeline_run_id,
            ),
        )

    conn.commit()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)
    parser.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
    )
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)

    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=DEFAULT_MIN_OVERLAP_RATIO,
        help=(
            "Reserved for backward compatibility; overlap contributes "
            "to lineage score but is not a hard filter."
        ),
    )
    parser.add_argument("--sim-weight", type=float, default=DEFAULT_SCORE_SIM_WEIGHT)
    parser.add_argument(
        "--overlap-weight",
        type=float,
        default=DEFAULT_SCORE_OVERLAP_WEIGHT,
    )

    parser.add_argument("--skip-lineage", action="store_true")
    parser.add_argument("--dry-run-lineage", action="store_true")
    parser.add_argument(
        "--min-valid-cluster-count",
        type=int,
        default=DEFAULT_MIN_VALID_CLUSTER_COUNT,
    )
    parser.add_argument(
        "--max-largest-cluster-ratio",
        type=float,
        default=DEFAULT_MAX_LARGEST_CLUSTER_RATIO,
    )
    parser.add_argument("--skip-quality-gate", action="store_true")

    parser.add_argument(
        "--video-target-duration-seconds",
        type=int,
        default=DEFAULT_VIDEO_TARGET_DURATION_SECONDS,
        help="Target duration included in the Mistral video payload.",
    )
    parser.add_argument(
        "--skip-mistral-payload",
        action="store_true",
        help="Do not build the Mistral video payload after a successful pipeline.",
    )

    return parser.parse_args()


def get_previous_success_run_id(conn, current_run_id: int) -> int | None:
    """
    Возвращает единственного допустимого родителя для lineage:
    ближайший завершённый run с меньшим id.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id
            FROM clustering_runs
            WHERE id < %s
              AND status IN ('success', 'completed', 'degraded')
              AND finished_at IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (current_run_id,),
        )
        row = cur.fetchone()

    return int(row["id"]) if row else None


def rebuild_lineage_for_new_run(
    conn,
    current_run_id: int,
    min_similarity: float,
    _legacy_min_overlap_ratio: float,
    sim_weight: float,
    overlap_weight: float,
    dry_run: bool = False,
) -> int:
    """
    Создаёт lineage только между соседними completed runs.

    Важно: функция не выполняет commit. Commit остаётся на уровне main(),
    чтобы lineage и radial maps формировали одну атомарную pipeline-транзакцию.
    """
    parent_run_id = get_previous_success_run_id(conn, current_run_id)

    if parent_run_id is None:
        log.info(
            "No previous completed run found; lineage skipped for child_run_id=%s",
            current_run_id,
        )
        return 0

    log.info(
        "Building strict adjacent lineage parent_run_id=%s child_run_id=%s",
        parent_run_id,
        current_run_id,
    )

    candidates = build_candidates(
        conn=conn,
        parent_run_id=parent_run_id,
        child_run_id=current_run_id,
        min_similarity=min_similarity,
        _legacy_min_overlap_ratio=_legacy_min_overlap_ratio,
        sim_weight=sim_weight,
        overlap_weight=overlap_weight,
    )
    matches = select_mutual_best(candidates)

    log.info(
        "Lineage candidates=%s final_matches=%s parent_run_id=%s child_run_id=%s",
        len(candidates),
        len(matches),
        parent_run_id,
        current_run_id,
    )

    if dry_run:
        for row in matches[:20]:
            log.info(
                (
                    "DRY RUN MATCH parent=%s child=%s sim=%.4f "
                    "overlap=%.4f overlap_count=%s score=%.4f"
                ),
                row["parent_cluster_id"],
                row["child_cluster_id"],
                row["centroid_similarity"],
                row["article_overlap_ratio"],
                row["article_overlap_count"],
                row["score"],
            )
        return len(matches)

    delete_existing_lineage(
        conn,
        parent_run_id=parent_run_id,
        child_run_id=current_run_id,
    )
    inserted = save_lineage(conn, matches)

    log.info(
        "Saved strict adjacent lineage rows=%s parent_run_id=%s child_run_id=%s",
        inserted,
        parent_run_id,
        current_run_id,
    )

    return inserted


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
    )
    log.setLevel(logging.INFO)

    args = parse_args()

    if abs((args.sim_weight + args.overlap_weight) - 1.0) > 1e-9:
        raise ValueError("sim-weight + overlap-weight must equal 1.0")

    if args.video_target_duration_seconds < 30:
        raise ValueError("video-target-duration-seconds must be at least 30")

    conn = get_conn()
    pipeline_run_id = start_pipeline_run(
        conn,
        job_type="pipeline",
        meta={
            "window_hours": args.window_hours,
            "limit": args.limit,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "max_per_source": args.max_per_source,
            "min_similarity": args.min_similarity,
            "min_overlap_ratio": args.min_overlap_ratio,
            "overlap_hard_filter": False,
            "sim_weight": args.sim_weight,
            "overlap_weight": args.overlap_weight,
            "skip_lineage": args.skip_lineage,
            "dry_run_lineage": args.dry_run_lineage,
            "min_valid_cluster_count": args.min_valid_cluster_count,
            "max_largest_cluster_ratio": args.max_largest_cluster_ratio,
            "skip_quality_gate": args.skip_quality_gate,
            "video_target_duration_seconds": args.video_target_duration_seconds,
            "skip_mistral_payload": args.skip_mistral_payload,
        },
    )

    pipeline_ctx = PipelineContext(
        run_id=None,
        stage="pipeline",
        attempt=1,
    )

    emit_stage_event(
        "INFO",
        pipeline_ctx,
        "pipeline_started",
        job_type="pipeline",
        pipeline_run_id=pipeline_run_id,
        window_hours=args.window_hours,
        limit=args.limit,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        max_per_source=args.max_per_source,
        min_similarity=args.min_similarity,
        min_overlap_ratio=args.min_overlap_ratio,
        sim_weight=args.sim_weight,
        overlap_weight=args.overlap_weight,
        skip_lineage=args.skip_lineage,
        dry_run_lineage=args.dry_run_lineage,
        min_valid_cluster_count=args.min_valid_cluster_count,
        max_largest_cluster_ratio=args.max_largest_cluster_ratio,
        skip_quality_gate=args.skip_quality_gate,
        video_target_duration_seconds=args.video_target_duration_seconds,
        skip_mistral_payload=args.skip_mistral_payload,
    )

    try:
        clustering_ctx = PipelineContext(
            run_id=None,
            stage="clustering",
            attempt=1,
        )

        emit_stage_event(
            "INFO",
            clustering_ctx,
            "clustering_started",
            pipeline_run_id=pipeline_run_id,
            window_hours=args.window_hours,
            limit=args.limit,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            min_valid_cluster_count=args.min_valid_cluster_count,
            max_largest_cluster_ratio=args.max_largest_cluster_ratio,
            max_per_source=args.max_per_source,
            skip_quality_gate=args.skip_quality_gate,
        )

        try:
            run_id = run_clustering(
                window_hours=args.window_hours,
                limit=args.limit,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                min_valid_cluster_count=args.min_valid_cluster_count,
                max_largest_cluster_ratio=args.max_largest_cluster_ratio,
                max_per_source=args.max_per_source,
                skip_quality_gate=args.skip_quality_gate,
            )
        except ValueError as exc:
            wrapped = PipelineStageError(
                "CLUSTERING_VALIDATION_FAILED",
                str(exc),
                error_type=type(exc).__name__,
                retryable=False,
                extra={"pipeline_run_id": pipeline_run_id},
            )
            wrapped.__cause__ = exc
            capture_pipeline_error(clustering_ctx, wrapped)
            raise
        except Exception as exc:
            wrapped = PipelineStageError(
                "CLUSTERING_FAILED",
                str(exc),
                error_type=type(exc).__name__,
                retryable=False,
                extra={"pipeline_run_id": pipeline_run_id},
            )
            wrapped.__cause__ = exc
            capture_pipeline_error(clustering_ctx, wrapped)
            raise

        clustering_ctx.run_id = run_id

        emit_stage_event(
            "INFO",
            clustering_ctx,
            "clustering_finished",
            pipeline_run_id=pipeline_run_id,
            produced_run_id=run_id,
        )

        if run_id is None:
            emit_stage_event(
                "INFO",
                pipeline_ctx,
                "pipeline_no_run",
                pipeline_run_id=pipeline_run_id,
            )
            finish_pipeline_run(
                conn,
                pipeline_run_id,
                status="no_run",
                related_run_id=None,
                meta={
                    "lineage_inserted": 0,
                    "mistral_payload_built": False,
                },
            )
            return

        if args.skip_lineage:
            emit_stage_event(
                "INFO",
                pipeline_ctx,
                "pipeline_finished",
                pipeline_run_id=pipeline_run_id,
                related_run_id=run_id,
                lineage_skipped=True,
                lineage_inserted=0,
                mistral_payload_built=False,
            )
            finish_pipeline_run(
                conn,
                pipeline_run_id,
                status="success",
                related_run_id=run_id,
                meta={
                    "lineage_skipped": True,
                    "lineage_inserted": 0,
                    "mistral_payload_built": False,
                },
            )
            return

        lineage_ctx = PipelineContext(
            run_id=run_id,
            stage="lineage",
            attempt=1,
        )

        emit_stage_event(
            "INFO",
            lineage_ctx,
            "lineage_started",
            pipeline_run_id=pipeline_run_id,
            min_similarity=args.min_similarity,
            min_overlap_ratio=args.min_overlap_ratio,
            sim_weight=args.sim_weight,
            overlap_weight=args.overlap_weight,
            dry_run=args.dry_run_lineage,
        )

        try:
            inserted = rebuild_lineage_for_new_run(
                conn=conn,
                current_run_id=run_id,
                min_similarity=args.min_similarity,
                _legacy_min_overlap_ratio=args.min_overlap_ratio,
                sim_weight=args.sim_weight,
                overlap_weight=args.overlap_weight,
                dry_run=args.dry_run_lineage,
            )
        except Exception as exc:
            capture_pipeline_error(
                lineage_ctx,
                PipelineStageError(
                    "LINEAGE_BUILD_FAILED",
                    str(exc),
                    error_type=type(exc).__name__,
                    retryable=False,
                    extra={"pipeline_run_id": pipeline_run_id},
                ),
            )
            raise

        emit_stage_event(
            "INFO",
            lineage_ctx,
            "lineage_finished",
            pipeline_run_id=pipeline_run_id,
            lineage_matches=inserted,
            dry_run=args.dry_run_lineage,
        )

        radial_ctx = PipelineContext(
            run_id=run_id,
            stage="radial_map",
            attempt=1,
        )

        emit_stage_event(
            "INFO",
            radial_ctx,
            "radial_map_started",
            pipeline_run_id=pipeline_run_id,
        )

        try:
            radial_result = build_and_save_radial_maps_for_run(conn, run_id)
        except Exception as exc:
            capture_pipeline_error(
                radial_ctx,
                PipelineStageError(
                    "RADIAL_MAP_BUILD_FAILED",
                    str(exc),
                    error_type=type(exc).__name__,
                    retryable=False,
                    extra={"pipeline_run_id": pipeline_run_id},
                ),
            )
            raise

        emit_stage_event(
            "INFO",
            radial_ctx,
            "radial_map_finished",
            pipeline_run_id=pipeline_run_id,
            radial_cluster_count=radial_result["cluster_count"],
            radial_point_count=radial_result["point_count"],
        )

        if args.dry_run_lineage:
            conn.rollback()

            emit_stage_event(
                "INFO",
                pipeline_ctx,
                "pipeline_finished",
                pipeline_run_id=pipeline_run_id,
                related_run_id=run_id,
                dry_run_lineage=True,
                lineage_matches=inserted,
                radial_cluster_count=radial_result["cluster_count"],
                radial_point_count=radial_result["point_count"],
                mistral_payload_built=False,
            )

            finish_pipeline_run(
                conn,
                pipeline_run_id,
                status="success",
                related_run_id=run_id,
                meta={
                    "dry_run_lineage": True,
                    "lineage_matches": inserted,
                    "radial_cluster_count": radial_result["cluster_count"],
                    "radial_point_count": radial_result["point_count"],
                    "mistral_payload_built": False,
                },
            )
            return

        conn.commit()

        mistral_payload = None
        mistral_payload_built = False
        mistral_payload_parent_run_id = None
        mistral_payload_topic_count = 0

        if not args.skip_mistral_payload:
            mistral_ctx = PipelineContext(
                run_id=run_id,
                stage="mistral_payload",
                attempt=1,
            )

            emit_stage_event(
                "INFO",
                mistral_ctx,
                "mistral_payload_started",
                pipeline_run_id=pipeline_run_id,
                target_duration_seconds=args.video_target_duration_seconds,
            )

            try:
                mistral_payload = build_mistral_video_payload(
                    conn=conn,
                    child_run_id=run_id,
                    target_duration_seconds=args.video_target_duration_seconds,
                )
            except Exception as exc:
                capture_pipeline_error(
                    mistral_ctx,
                    PipelineStageError(
                        "MISTRAL_PAYLOAD_BUILD_FAILED",
                        str(exc),
                        error_type=type(exc).__name__,
                        retryable=False,
                        extra={"pipeline_run_id": pipeline_run_id},
                    ),
                )
                raise

            mistral_payload_built = True
            mistral_payload_parent_run_id = (
                mistral_payload["analysis_context"].get("parent_run_id")
            )
            mistral_payload_topic_count = len(
                mistral_payload.get("editorial_topics", [])
            )

            log.info(
                (
                    "Mistral payload built: child_run_id=%s parent_run_id=%s "
                    "topics=%s target_duration_seconds=%s"
                ),
                run_id,
                mistral_payload_parent_run_id,
                mistral_payload_topic_count,
                args.video_target_duration_seconds,
            )

            emit_stage_event(
                "INFO",
                mistral_ctx,
                "mistral_payload_finished",
                pipeline_run_id=pipeline_run_id,
                parent_run_id=mistral_payload_parent_run_id,
                editorial_topic_count=mistral_payload_topic_count,
                target_duration_seconds=args.video_target_duration_seconds,
            )

        emit_stage_event(
            "INFO",
            pipeline_ctx,
            "pipeline_finished",
            pipeline_run_id=pipeline_run_id,
            related_run_id=run_id,
            lineage_inserted=inserted,
            radial_cluster_count=radial_result["cluster_count"],
            radial_point_count=radial_result["point_count"],
            mistral_payload_built=mistral_payload_built,
            mistral_payload_topic_count=mistral_payload_topic_count,
        )

        finish_pipeline_run(
            conn,
            pipeline_run_id,
            status="success",
            related_run_id=run_id,
            meta={
                "lineage_inserted": inserted,
                "radial_cluster_count": radial_result["cluster_count"],
                "radial_point_count": radial_result["point_count"],
                "mistral_payload_built": mistral_payload_built,
                "mistral_payload_parent_run_id": mistral_payload_parent_run_id,
                "mistral_payload_topic_count": mistral_payload_topic_count,
                "video_target_duration_seconds": args.video_target_duration_seconds,
            },
        )

    except Exception as exc:
        failed_run_id = locals().get("run_id")

        try:
            conn.rollback()
        except Exception:
            pass

        capture_pipeline_error(
            PipelineContext(
                run_id=failed_run_id,
                stage="pipeline",
                attempt=1,
            ),
            PipelineStageError(
                "PIPELINE_FAILED",
                str(exc),
                error_type=type(exc).__name__,
                retryable=False,
                extra={
                    "pipeline_run_id": pipeline_run_id,
                    "failed_run_id": failed_run_id,
                },
            ),
        )

        try:
            finish_pipeline_run(
                conn,
                pipeline_run_id,
                status="failed",
                related_run_id=failed_run_id,
                error=str(exc),
                meta={
                    "failed_step": "pipeline",
                    "error_code": "PIPELINE_FAILED",
                    "error_type": type(exc).__name__,
                },
            )
        except Exception:
            log.exception(
                "Failed to update pipeline_runs for pipeline_run_id=%s",
                pipeline_run_id,
            )

        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()