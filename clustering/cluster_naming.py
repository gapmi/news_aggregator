from __future__ import annotations

from collections import Counter
import re


def generate_cluster_name(conn, cluster_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.title, a.source
            FROM cluster_articles ca
            JOIN articles a ON a.id = ca.article_id
            WHERE ca.cluster_id = %s
            ORDER BY a.published DESC NULLS LAST, a.id DESC
            """,
            (cluster_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return {
            "name_title": f"Unnamed cluster {cluster_id}",
            "tags": [],
            "language_code": None,
        }

    titles = [row[0] for row in rows if row[0]]
    sources = [row[1] for row in rows if row[1]]

    tokens = []
    for title in titles:
        tokens.extend(re.findall(r"\b[\w-]{3,}\b", title.lower()))

    stopwords = {
        "the", "and", "for", "with", "from", "that", "this",
        "after", "before", "into", "over", "under", "news",
        "said", "says", "will", "have", "has",
    }

    counts = Counter(token for token in tokens if token not in stopwords)
    tags = [token.replace("-", " ").title() for token, _count in counts.most_common(7)]

    name_title = ", ".join(tags[:2]) if tags else f"Unnamed cluster {cluster_id}"

    return {
        "name_title": name_title,
        "tags": tags[:7],
        "language_code": "en",
    }


def upsert_cluster_name(conn, cluster_id: int) -> None:
    payload = generate_cluster_name(conn, cluster_id)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cluster_names (
                cluster_id,
                name_short,
                name_title,
                tags,
                language_code,
                concepts
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (cluster_id) DO UPDATE
            SET
                name_short = EXCLUDED.name_short,
                name_title = EXCLUDED.name_title,
                tags = EXCLUDED.tags,
                language_code = EXCLUDED.language_code,
                concepts = EXCLUDED.concepts
            """,
            (
                cluster_id,
                payload["name_title"][:80],
                payload["name_title"],
                payload["tags"],
                payload["language_code"],
                None,
            ),
        )


def upsert_cluster_names_for_run(conn, run_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM clusters WHERE run_id = %s ORDER BY id",
            (run_id,),
        )
        cluster_ids = [row[0] for row in cur.fetchall()]

    for cluster_id in cluster_ids:
        upsert_cluster_name(conn, cluster_id)

    return len(cluster_ids)