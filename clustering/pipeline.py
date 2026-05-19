from __future__ import annotations

import argparse
import logging

import psycopg2.extras

from clustering.offline import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_LARGEST_CLUSTER_RATIO,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_MIN_VALID_CLUSTER_COUNT,
    DEFAULT_WINDOW_HOURS,
    get_conn,
    run_clustering,
)

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


log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)

    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--min-overlap-ratio", type=float, default=DEFAULT_MIN_OVERLAP_RATIO)
    parser.add_argument("--sim-weight", type=float, default=DEFAULT_SCORE_SIM_WEIGHT)
    parser.add_argument("--overlap-weight", type=float, default=DEFAULT_SCORE_OVERLAP_WEIGHT)

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
    parser.add_argument(
        "--skip-quality-gate",
        action="store_true",
    )

    return parser.parse_args()


def get_previous_success_run_id(conn, current_run_id: int) -> int | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT prev.id
            FROM clustering_runs current
            JOIN clustering_runs prev
              ON prev.started_at < current.started_at
            WHERE current.id = %s
              AND prev.status IN ('success', 'completed')
              AND prev.finished_at IS NOT NULL
            ORDER BY prev.started_at DESC, prev.id DESC
            LIMIT 1
            """,
            (current_run_id,),
        )
        row = cur.fetchone()

    return row["id"] if row else None


def rebuild_lineage_for_new_run(
    conn,
    current_run_id: int,
    min_similarity: float,
    min_overlap_ratio: float,
    sim_weight: float,
    overlap_weight: float,
    dry_run: bool = False,
) -> int:
    parent_run_id = get_previous_success_run_id(conn, current_run_id)

    if parent_run_id is None:
        log.info("No previous successful run found for current_run_id=%s", current_run_id)
        return 0

    log.info(
        "Building lineage parent_run_id=%s child_run_id=%s",
        parent_run_id,
        current_run_id,
    )

    candidates = build_candidates(
        conn=conn,
        parent_run_id=parent_run_id,
        child_run_id=current_run_id,
        min_similarity=min_similarity,
        min_overlap_ratio=min_overlap_ratio,
        sim_weight=sim_weight,
        overlap_weight=overlap_weight,
    )

    matches = select_mutual_best(candidates)

    log.info("Lineage candidates=%s final_matches=%s", len(candidates), len(matches))

    if dry_run:
        for row in matches[:20]:
            log.info(
                "DRY RUN MATCH parent=%s child=%s sim=%.4f overlap=%.4f overlap_count=%s score=%.4f",
                row["parent_cluster_id"],
                row["child_cluster_id"],
                row["centroid_similarity"],
                row["article_overlap_ratio"],
                row["article_overlap_count"],
                row["score"],
            )
        return len(matches)

    delete_existing_lineage(conn, parent_run_id, current_run_id)
    inserted = save_lineage(conn, matches)

    log.info(
        "Saved lineage rows=%s parent_run_id=%s child_run_id=%s",
        inserted,
        parent_run_id,
        current_run_id,
    )

    return inserted


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    if abs((args.sim_weight + args.overlap_weight) - 1.0) > 1e-9:
        raise ValueError("sim-weight + overlap-weight must equal 1.0")

    log.info(
        "Starting clustering pipeline window_hours=%s limit=%s min_cluster_size=%s min_samples=%s",
        args.window_hours,
        args.limit,
        args.min_cluster_size,
        args.min_samples,
    )

    run_id = run_clustering(
        window_hours=args.window_hours,
        limit=args.limit,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        min_valid_cluster_count=args.min_valid_cluster_count,
        max_largest_cluster_ratio=args.max_largest_cluster_ratio,
        skip_quality_gate=args.skip_quality_gate,
    )

    if run_id is None:
        log.info("Pipeline finished: clustering produced no run")
        return

    if args.skip_lineage:
        log.info("Pipeline finished: run_id=%s lineage skipped", run_id)
        return

    conn = get_conn()

    try:
        inserted = rebuild_lineage_for_new_run(
            conn=conn,
            current_run_id=run_id,
            min_similarity=args.min_similarity,
            min_overlap_ratio=args.min_overlap_ratio,
            sim_weight=args.sim_weight,
            overlap_weight=args.overlap_weight,
            dry_run=args.dry_run_lineage,
        )

        if args.dry_run_lineage:
            conn.rollback()
            log.info(
                "Pipeline dry-run finished: run_id=%s lineage_matches=%s",
                run_id,
                inserted,
            )
        else:
            conn.commit()
            log.info(
                "Pipeline finished: run_id=%s lineage_inserted=%s",
                run_id,
                inserted,
            )

    except Exception:
        conn.rollback()
        log.exception("Pipeline failed during lineage step for run_id=%s", run_id)
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()