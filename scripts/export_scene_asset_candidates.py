import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import psycopg2.extras

from clustering.offline import get_conn


FIELDNAMES = [
    "scene_number",
    "topic_reference",
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
]


QUERY = """
WITH scene_topics AS (
    SELECT
        s.run_id,
        s.scene_number,
        topic_reference.value AS topic_reference
    FROM mistral_video_script_scenes AS s
    CROSS JOIN LATERAL jsonb_array_elements_text(
        s.topic_references
    ) AS topic_reference(value)
    WHERE s.run_id = %(run_id)s
),
scene_clusters AS (
    SELECT DISTINCT ON (st.scene_number, st.topic_reference)
        st.scene_number,
        st.topic_reference,
        c.id AS cluster_id,
        c.run_id,
        c.size AS cluster_size,
        c.representative_article_id,
        c.representative_title
    FROM scene_topics AS st
    JOIN clusters AS c
      ON c.run_id = st.run_id
    LEFT JOIN cluster_names AS cn
      ON cn.cluster_id = c.id
    WHERE st.topic_reference IN (
        cn.name_title,
        cn.name_short,
        c.representative_title
    )
    ORDER BY
        st.scene_number,
        st.topic_reference,
        c.size DESC,
        c.id
),
candidate_articles AS (
    SELECT
        sc.scene_number,
        sc.topic_reference,
        sc.cluster_id,
        sc.run_id,
        sc.cluster_size,
        sc.representative_article_id,
        sc.representative_title,
        a.id AS article_id,
        a.title AS article_title,
        a.url AS article_url,
        a.source,
        a.published,
        a.rss_description,
        CASE
            WHEN a.id = sc.representative_article_id THEN 0
            ELSE 1
        END AS representative_sort
    FROM scene_clusters AS sc
    JOIN cluster_articles AS ca
      ON ca.cluster_id = sc.cluster_id
    JOIN articles AS a
      ON a.id = ca.article_id
    WHERE a.url IS NOT NULL
      AND BTRIM(a.url) <> ''
),
ranked_candidates AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY scene_number, cluster_id
            ORDER BY
                representative_sort,
                published DESC NULLS LAST,
                article_id DESC
        ) AS candidate_rank
    FROM candidate_articles
)
SELECT
    scene_number,
    topic_reference,
    cluster_id,
    run_id,
    cluster_size,
    representative_article_id,
    representative_title,
    candidate_rank,
    (article_id = representative_article_id) AS is_representative,
    article_id,
    article_title,
    article_url,
    source,
    published,
    rss_description
FROM ranked_candidates
WHERE candidate_rank <= 3
ORDER BY
    scene_number,
    candidate_rank
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export up to three article candidates per Mistral video scene "
            "for the cluster asset collector."
        )
    )
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def main() -> int:
    args = parse_args()

    if args.run_id < 1:
        raise ValueError("--run-id must be positive")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    conn = get_conn()

    try:
        with conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(QUERY, {"run_id": args.run_id})
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError(
            f"No scene-linked asset candidates found for run_id={args.run_id}"
        )

    scene_numbers = sorted(
        {int(row["scene_number"]) for row in rows}
    )

    with args.output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: clean_value(row.get(field))
                    for field in FIELDNAMES
                }
            )

    by_scene: dict[int, int] = {}
    for row in rows:
        scene_number = int(row["scene_number"])
        by_scene[scene_number] = by_scene.get(scene_number, 0) + 1

    print(
        {
            "run_id": args.run_id,
            "output_csv": str(args.output_csv),
            "row_count": len(rows),
            "scene_numbers": scene_numbers,
            "candidates_per_scene": by_scene,
        }
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)