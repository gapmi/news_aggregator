from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any

import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)

DEFAULT_MIN_SIMILARITY = 0.80
DEFAULT_MIN_OVERLAP_RATIO = 0.20
DEFAULT_SCORE_SIM_WEIGHT = 0.70
DEFAULT_SCORE_OVERLAP_WEIGHT = 0.30


@dataclass
class RunPair:
    parent_run_id: int
    child_run_id: int


def get_conn():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "news_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "qg9PlWWpeffd"),
    )


def get_last_two_completed_runs(conn) -> RunPair | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, started_at, finished_at
            FROM clustering_runs
            WHERE status IN ('success', 'completed')
              AND finished_at IS NOT NULL
            ORDER BY started_at DESC
            LIMIT 2
            """
        )
        rows = cur.fetchall()

    if len(rows) < 2:
        return None

    child_run_id = rows[0]["id"]
    parent_run_id = rows[1]["id"]
    return RunPair(parent_run_id=parent_run_id, child_run_id=child_run_id)


def delete_existing_lineage(conn, parent_run_id: int, child_run_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM cluster_lineage
            WHERE parent_run_id = %s
              AND child_run_id = %s
            """,
            (parent_run_id, child_run_id),
        )


def build_candidates(
    conn,
    parent_run_id: int,
    child_run_id: int,
    min_similarity: float,
    _legacy_min_overlap_ratio: float,
    sim_weight: float,
    overlap_weight: float,
) -> list[dict[str, Any]]:
    # Kept for backward compatibility: overlap contributes to score,
    # but is no longer used as a hard filter for candidate selection.
    sql = """
    WITH parent_clusters AS (
        SELECT id, run_id, size, centroid
        FROM clusters
        WHERE run_id = %(parent_run_id)s
    ),
    child_clusters AS (
        SELECT id, run_id, size, centroid
        FROM clusters
        WHERE run_id = %(child_run_id)s
    ),
    cluster_overlaps AS (
        SELECT
            pca.cluster_id AS parent_cluster_id,
            cca.cluster_id AS child_cluster_id,
            COUNT(*)::int AS overlap_count
        FROM cluster_articles pca
        JOIN cluster_articles cca
          ON cca.article_id = pca.article_id
        WHERE pca.cluster_id IN (SELECT id FROM parent_clusters)
          AND cca.cluster_id IN (SELECT id FROM child_clusters)
        GROUP BY pca.cluster_id, cca.cluster_id
    )
    SELECT
        p.run_id AS parent_run_id,
        c.run_id AS child_run_id,
        p.id AS parent_cluster_id,
        c.id AS child_cluster_id,
        p.size AS parent_size,
        c.size AS child_size,
        1 - (p.centroid <=> c.centroid) AS centroid_similarity,
        COALESCE(o.overlap_count, 0) AS article_overlap_count,
        COALESCE(o.overlap_count, 0)::double precision
            / LEAST(p.size, c.size)::double precision AS article_overlap_ratio,
        (
            %(sim_weight)s * (1 - (p.centroid <=> c.centroid)) +
            %(overlap_weight)s * (
                COALESCE(o.overlap_count, 0)::double precision
                / LEAST(p.size, c.size)::double precision
            )
        ) AS score
    FROM parent_clusters p
    CROSS JOIN child_clusters c
    LEFT JOIN cluster_overlaps o
      ON o.parent_cluster_id = p.id
     AND o.child_cluster_id = c.id
    WHERE (1 - (p.centroid <=> c.centroid)) >= %(min_similarity)s
    ORDER BY
        parent_cluster_id,
        score DESC,
        article_overlap_count DESC,
        child_size DESC,
        child_cluster_id
    """

    params = {
        "parent_run_id": parent_run_id,
        "child_run_id": child_run_id,
        "min_similarity": min_similarity,
        "sim_weight": sim_weight,
        "overlap_weight": overlap_weight,
    }

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def select_mutual_best(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_parent: dict[int, dict[str, Any]] = {}
    for row in candidates:
        parent_id = row["parent_cluster_id"]
        current = best_by_parent.get(parent_id)
        if current is None or (
            row["score"],
            row["article_overlap_count"],
            row["child_size"],
            -row["child_cluster_id"],
        ) > (
            current["score"],
            current["article_overlap_count"],
            current["child_size"],
            -current["child_cluster_id"],
        ):
            best_by_parent[parent_id] = row

    best_by_child: dict[int, dict[str, Any]] = {}
    for row in best_by_parent.values():
        child_id = row["child_cluster_id"]
        current = best_by_child.get(child_id)
        if current is None or (
            row["score"],
            row["article_overlap_count"],
            row["parent_size"],
            -row["parent_cluster_id"],
        ) > (
            current["score"],
            current["article_overlap_count"],
            current["parent_size"],
            -current["parent_cluster_id"],
        ):
            best_by_child[child_id] = row

    return sorted(
        best_by_child.values(),
        key=lambda x: (x["parent_cluster_id"], x["child_cluster_id"]),
    )


def save_lineage(conn, matches: list[dict[str, Any]]) -> int:
    if not matches:
        return 0

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO cluster_lineage (
                parent_run_id,
                child_run_id,
                parent_cluster_id,
                child_cluster_id,
                centroid_similarity,
                article_overlap_ratio,
                article_overlap_count,
                parent_size,
                child_size,
                score,
                link_type
            ) VALUES (
                %(parent_run_id)s,
                %(child_run_id)s,
                %(parent_cluster_id)s,
                %(child_cluster_id)s,
                %(centroid_similarity)s,
                %(article_overlap_ratio)s,
                %(article_overlap_count)s,
                %(parent_size)s,
                %(child_size)s,
                %(score)s,
                'continuation'
            )
            ON CONFLICT (parent_cluster_id, child_cluster_id) DO UPDATE
            SET centroid_similarity = EXCLUDED.centroid_similarity,
                article_overlap_ratio = EXCLUDED.article_overlap_ratio,
                article_overlap_count = EXCLUDED.article_overlap_count,
                parent_size = EXCLUDED.parent_size,
                child_size = EXCLUDED.child_size,
                score = EXCLUDED.score,
                link_type = EXCLUDED.link_type,
                matched_at = now()
            """,
            matches,
            page_size=100,
        )

    return len(matches)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run-id", type=int)
    parser.add_argument("--child-run-id", type=int)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=DEFAULT_MIN_OVERLAP_RATIO,
        help=(
            "Reserved for backward compatibility; overlap contributes "
            "to score but is not a hard filter for candidates."
        ),
    )
    parser.add_argument("--sim-weight", type=float, default=DEFAULT_SCORE_SIM_WEIGHT)
    parser.add_argument("--overlap-weight", type=float, default=DEFAULT_SCORE_OVERLAP_WEIGHT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if abs((args.sim_weight + args.overlap_weight) - 1.0) > 1e-9:
        raise ValueError("sim-weight + overlap-weight must equal 1.0")

    conn = get_conn()
    conn.autocommit = False

    try:
        if args.parent_run_id and args.child_run_id:
            run_pair = RunPair(parent_run_id=args.parent_run_id, child_run_id=args.child_run_id)
        else:
            run_pair = get_last_two_completed_runs(conn)

        if not run_pair:
            log.info("Not enough completed runs to build lineage")
            conn.rollback()
            return

        log.info(
            "Building lineage parent_run_id=%s child_run_id=%s",
            run_pair.parent_run_id,
            run_pair.child_run_id,
        )

        candidates = build_candidates(
            conn=conn,
            parent_run_id=run_pair.parent_run_id,
            child_run_id=run_pair.child_run_id,
            min_similarity=args.min_similarity,
            _legacy_min_overlap_ratio=args.min_overlap_ratio,
            sim_weight=args.sim_weight,
            overlap_weight=args.overlap_weight,
        )

        matches = select_mutual_best(candidates)

        log.info("Candidates=%s, final_matches=%s", len(candidates), len(matches))

        if args.dry_run:
            conn.rollback()
            for row in matches[:20]:
                log.info(
                    "MATCH parent=%s child=%s sim=%.4f overlap=%.4f overlap_count=%s score=%.4f",
                    row["parent_cluster_id"],
                    row["child_cluster_id"],
                    row["centroid_similarity"],
                    row["article_overlap_ratio"],
                    row["article_overlap_count"],
                    row["score"],
                )
            return

        delete_existing_lineage(conn, run_pair.parent_run_id, run_pair.child_run_id)
        inserted = save_lineage(conn, matches)
        conn.commit()
        log.info("Saved %s lineage rows", inserted)

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()