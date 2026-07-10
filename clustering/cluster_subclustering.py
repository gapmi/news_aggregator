from __future__ import annotations

from collections import defaultdict

import hdbscan
import numpy as np
import psycopg2.extras
from psycopg2.extras import execute_values


MIN_SUBCLUSTER_PARENT_SIZE = 10
SUBCLUSTER_MIN_CLUSTER_SIZE = 3
SUBCLUSTER_MIN_SAMPLES = 2


def _compute_centroid(vectors: np.ndarray) -> np.ndarray:
    return vectors.mean(axis=0).astype(np.float32)


def _choose_representative(article_rows: list[dict], vectors: np.ndarray, centroid: np.ndarray):
    distances = np.linalg.norm(vectors - centroid, axis=1)
    idx = int(np.argmin(distances))
    return article_rows[idx], distances


def _load_cluster_articles_for_run(conn, run_id: int) -> dict[int, list[dict]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                c.id AS cluster_id,
                a.id AS article_id,
                a.title,
                a.embedding
            FROM clusters c
            JOIN cluster_articles ca ON ca.cluster_id = c.id
            JOIN articles a ON a.id = ca.article_id
            WHERE c.run_id = %s
              AND a.embedding IS NOT NULL
            ORDER BY c.id, a.id
            """,
            (run_id,),
        )
        rows = cur.fetchall()

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["cluster_id"])].append(
            {
                "article_id": int(row["article_id"]),
                "title": row["title"],
                "embedding": row["embedding"].to_numpy().astype(np.float32) if hasattr(row["embedding"], "to_numpy") else np.array(row["embedding"], dtype=np.float32),
            }
        )
    return grouped


def _delete_existing_subclusters_for_run(conn, run_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM cluster_subclusters
            WHERE run_id = %s
            """,
            (run_id,),
        )


def save_cluster_subclusters_for_run(
    conn,
    run_id: int,
    min_parent_size: int = MIN_SUBCLUSTER_PARENT_SIZE,
    min_cluster_size: int = SUBCLUSTER_MIN_CLUSTER_SIZE,
    min_samples: int = SUBCLUSTER_MIN_SAMPLES,
) -> dict:
    cluster_articles = _load_cluster_articles_for_run(conn, run_id)
    _delete_existing_subclusters_for_run(conn, run_id)

    subcluster_rows = []
    article_rows = []
    summary = {
        "cluster_count_seen": 0,
        "cluster_count_subclustered": 0,
        "subcluster_count": 0,
        "assigned_article_count": 0,
        "unassigned_article_count": 0,
    }

    for cluster_id, items in cluster_articles.items():
        summary["cluster_count_seen"] += 1
        n = len(items)

        if n < min_parent_size:
            summary["unassigned_article_count"] += n
            continue

        X = np.vstack([item["embedding"] for item in items]).astype(np.float32)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(X)

        probabilities = getattr(clusterer, "probabilities_", None)
        outlier_scores = getattr(clusterer, "outlier_scores_", None)

        grouped = defaultdict(list)
        for idx, label in enumerate(labels.tolist()):
            if int(label) == -1:
                continue
            grouped[int(label)].append(idx)

        if not grouped:
            summary["unassigned_article_count"] += n
            continue

        summary["cluster_count_subclustered"] += 1

        sorted_labels = sorted(grouped.keys())
        local_subcluster_payloads = []

        for label in sorted_labels:
            member_indices = grouped[label]
            member_items = [items[i] for i in member_indices]
            member_vectors = np.vstack([items[i]["embedding"] for i in member_indices]).astype(np.float32)

            centroid = _compute_centroid(member_vectors)
            representative, member_distances = _choose_representative(member_items, member_vectors, centroid)

            local_subcluster_payloads.append(
                {
                    "run_id": run_id,
                    "cluster_id": cluster_id,
                    "label": int(label),
                    "size": len(member_indices),
                    "is_noise": False,
                    "representative_article_id": representative["article_id"],
                    "representative_title": representative["title"],
                    "centroid": centroid,
                    "member_indices": member_indices,
                }
            )

        inserted = []
        with conn.cursor() as cur:
            inserted = execute_values(
                cur,
                """
                INSERT INTO cluster_subclusters (
                    run_id,
                    cluster_id,
                    label,
                    size,
                    is_noise,
                    representative_article_id,
                    representative_title,
                    centroid
                )
                VALUES %s
                RETURNING id, cluster_id, label
                """,
                [
                    (
                        row["run_id"],
                        row["cluster_id"],
                        row["label"],
                        row["size"],
                        row["is_noise"],
                        row["representative_article_id"],
                        row["representative_title"],
                        row["centroid"],
                    )
                    for row in local_subcluster_payloads
                ],
                fetch=True,
            )
            inserted = cur.fetchall()

        subcluster_id_by_label = {int(label): int(subcluster_id) for subcluster_id, _cluster_id, label in inserted}

        for payload in local_subcluster_payloads:
            label = payload["label"]
            subcluster_id = subcluster_id_by_label[label]
            centroid = payload["centroid"]

            for idx in payload["member_indices"]:
                item = items[idx]
                distance = float(np.linalg.norm(item["embedding"] - centroid))
                probability = (
                    float(probabilities[idx])
                    if probabilities is not None and np.isfinite(probabilities[idx])
                    else None
                )
                outlier_score = (
                    float(outlier_scores[idx])
                    if outlier_scores is not None and np.isfinite(outlier_scores[idx])
                    else None
                )

                article_rows.append(
                    (
                        subcluster_id,
                        cluster_id,
                        item["article_id"],
                        run_id,
                        probability is not None and probability > 0.5,
                        distance,
                        probability,
                        outlier_score,
                    )
                )

        summary["subcluster_count"] += len(local_subcluster_payloads)
        summary["assigned_article_count"] += sum(len(x["member_indices"]) for x in local_subcluster_payloads)
        summary["unassigned_article_count"] += int(np.sum(labels == -1))

    if article_rows:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO cluster_subcluster_articles (
                    subcluster_id,
                    cluster_id,
                    article_id,
                    run_id,
                    is_core,
                    distance,
                    probability,
                    outlier_score
                )
                VALUES %s
                """,
                article_rows,
            )

    return summary