"""
- читает 10 baseline-кластеров из cluster_centroids.csv;
- сохраняет representative article как candidate_rank=1;
- выбирает ещё до двух статей того же кластера по минимальной cosine distance к сохранённому cluster centroid;
- не дублирует article_id;
- записывает cluster_asset_candidates.csv;
- не пишет в PostgreSQL и не меняет существующие assets.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras


INPUT_CSV = Path(os.environ.get("ASSET_CENTROIDS_CSV", "cluster_centroids.csv"))
OUTPUT_CSV = Path(
    os.environ.get("ASSET_CANDIDATES_CSV", "cluster_asset_candidates.csv")
)
CANDIDATES_PER_CLUSTER = 3


def read_centroid_rows() -> list[dict[str, str]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    with INPUT_CSV.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Input CSV is empty: {INPUT_CSV}")

    required_fields = {
        "cluster_id",
        "run_id",
        "representative_article_id",
    }
    missing = required_fields.difference(rows[0].keys())
    if missing:
        raise ValueError(
            "Input CSV is missing required fields: "
            + ", ".join(sorted(missing))
        )

    return rows


def connect() -> psycopg2.extensions.connection:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    return psycopg2.connect(database_url)


def load_candidates(
    conn: psycopg2.extensions.connection,
    cluster_ids: list[int],
    run_id: int,
) -> list[dict[str, Any]]:
    query = """
        WITH selected_clusters AS (
            SELECT
                c.id AS cluster_id,
                c.run_id,
                c.size AS cluster_size,
                c.representative_article_id,
                c.representative_title,
                c.centroid
            FROM public.clusters AS c
            WHERE c.run_id = %(run_id)s
              AND c.id = ANY(%(cluster_ids)s)
        ),
        ranked_members AS (
            SELECT
                c.cluster_id,
                c.run_id,
                c.cluster_size,
                c.representative_article_id,
                c.representative_title,
                a.id AS article_id,
                a.title AS article_title,
                a.url AS article_url,
                a.source,
                a.published,
                a.rss_description,
                (a.id = c.representative_article_id) AS is_representative,
                CASE
                    WHEN a.embedding IS NULL OR c.centroid IS NULL THEN NULL
                    ELSE a.embedding <=> c.centroid
                END AS centroid_distance,
                row_number() OVER (
                    PARTITION BY c.cluster_id
                    ORDER BY
                        CASE
                            WHEN a.id = c.representative_article_id THEN 0
                            ELSE 1
                        END,
                        CASE
                            WHEN a.embedding IS NULL OR c.centroid IS NULL
                            THEN NULL
                            ELSE a.embedding <=> c.centroid
                        END NULLS LAST,
                        a.published DESC NULLS LAST,
                        a.id DESC
                ) AS candidate_rank
            FROM selected_clusters AS c
            JOIN public.cluster_articles AS ca
              ON ca.cluster_id = c.cluster_id
            JOIN public.articles AS a
              ON a.id = ca.article_id
        )
        SELECT
            cluster_id,
            run_id,
            cluster_size,
            representative_article_id,
            representative_title,
            candidate_rank,
            is_representative,
            article_id,
            article_title,
            article_url,
            source,
            published,
            rss_description,
            centroid_distance
        FROM ranked_members
        WHERE candidate_rank <= %(candidates_per_cluster)s
        ORDER BY cluster_id, candidate_rank
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            query,
            {
                "run_id": run_id,
                "cluster_ids": cluster_ids,
                "candidates_per_cluster": CANDIDATES_PER_CLUSTER,
            },
        )
        return [dict(row) for row in cursor.fetchall()]


def normalize_value(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def main() -> None:
    centroid_rows = read_centroid_rows()

    run_ids = {int(row["run_id"]) for row in centroid_rows}
    if len(run_ids) != 1:
        raise ValueError(
            "Input CSV must contain exactly one run_id; found: "
            + ", ".join(str(run_id) for run_id in sorted(run_ids))
        )

    run_id = next(iter(run_ids))
    cluster_ids = [int(row["cluster_id"]) for row in centroid_rows]
    expected_cluster_ids = set(cluster_ids)

    with connect() as conn:
        candidate_rows = load_candidates(
            conn=conn,
            cluster_ids=cluster_ids,
            run_id=run_id,
        )

    actual_cluster_ids = {int(row["cluster_id"]) for row in candidate_rows}
    missing_clusters = expected_cluster_ids.difference(actual_cluster_ids)
    if missing_clusters:
        raise RuntimeError(
            "No candidates returned for clusters: "
            + ", ".join(str(cluster_id) for cluster_id in sorted(missing_clusters))
        )

    candidate_rows.sort(
        key=lambda row: (
            cluster_ids.index(int(row["cluster_id"])),
            int(row["candidate_rank"]),
        )
    )

    fieldnames = [
        "cluster_id",
        "run_id",
        "cluster_size",
        "representative_article_id",
        "representative_title",
        "candidate_rank",
        "is_representative",
        "article_id",
        "article_title",
        "article_url",
        "source",
        "published",
        "rss_description",
        "centroid_distance",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in candidate_rows:
            writer.writerow(
                {
                    field: normalize_value(row.get(field))
                    for field in fieldnames
                }
            )

    by_cluster: dict[str, int] = {}
    for row in candidate_rows:
        cluster_id = str(row["cluster_id"])
        by_cluster[cluster_id] = by_cluster.get(cluster_id, 0) + 1

    print(f"Input: {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Run ID: {run_id}")
    print(f"Clusters: {len(expected_cluster_ids)}")
    print(f"Candidate rows: {len(candidate_rows)}")
    print("Candidates by cluster:", dict(sorted(by_cluster.items(), key=lambda item: int(item[0]))))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
