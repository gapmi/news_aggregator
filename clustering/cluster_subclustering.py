from __future__ import annotations

from collections import defaultdict

import hdbscan
import numpy as np
import psycopg2.extras
from psycopg2.extras import execute_values


MIN_SUBCLUSTER_PARENT_SIZE = 80
SUBCLUSTER_MIN_CLUSTER_SIZE = 12
SUBCLUSTER_MIN_SAMPLES = 6
MIN_SECTOR_ASSIGNED_ARTICLES = 12
MIN_SECTOR_PARENT_SHARE = 0.015
MIN_SECTOR_QUALITY_SCORE = 0.90
MAX_SECTORS_PER_CLUSTER = 6
SECTOR_MAX_VISIBLE_COUNT = 8
SECTOR_MIN_MEDIAN_SIZE = 8
SUBCLUSTER_SELECTION_METHOD = "leaf"
SUBCLUSTER_MIN_PROBABILITY = 0.35
SUBCLUSTER_MAX_DISTANCE_QUANTILE = 0.90
DEFAULT_TRIGGER_PARENT_SIZE = 120
DEFAULT_TRIGGER_PARENT_RATIO = 0.10


def _to_numpy_vector(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "to_numpy"):
        return value.to_numpy().astype(np.float32)
    if hasattr(value, "tolist"):
        return np.asarray(value.tolist(), dtype=np.float32)
    if hasattr(value, "to_list"):
        return np.asarray(value.to_list(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


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
                c.size AS cluster_size,
                a.id AS article_id,
                a.title,
                a.source,
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
        embedding = _to_numpy_vector(row["embedding"])
        if embedding is None:
            continue
        grouped[int(row["cluster_id"])].append(
            {
                "cluster_size": int(row["cluster_size"]),
                "article_id": int(row["article_id"]),
                "title": row["title"],
                "source": row["source"],
                "embedding": embedding,
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


def _should_subcluster_parent(
    parent_size: int,
    total_article_count: int,
    min_parent_size: int,
    trigger_parent_size: int,
    trigger_parent_ratio: float,
) -> bool:
    if parent_size < min_parent_size:
        return False
    if parent_size >= trigger_parent_size:
        return True
    if total_article_count > 0 and (parent_size / total_article_count) >= trigger_parent_ratio:
        return True
    return False


def save_cluster_subclusters_for_run(
    conn,
    run_id: int,
    min_parent_size: int = MIN_SUBCLUSTER_PARENT_SIZE,
    min_cluster_size: int = SUBCLUSTER_MIN_CLUSTER_SIZE,
    min_samples: int = SUBCLUSTER_MIN_SAMPLES,
    trigger_parent_size: int = DEFAULT_TRIGGER_PARENT_SIZE,
    trigger_parent_ratio: float = DEFAULT_TRIGGER_PARENT_RATIO,
    total_article_count: int = 0,
) -> dict:
    cluster_articles = _load_cluster_articles_for_run(conn, run_id)
    _delete_existing_subclusters_for_run(conn, run_id)

    article_rows = []
    summary = {
        "cluster_count_seen": 0,
        "cluster_count_subclustered": 0,
        "subcluster_count": 0,
        "assigned_article_count": 0,
        "unassigned_article_count": 0,
        "skipped_parent_count": 0,
    }

    for cluster_id, items in cluster_articles.items():
        summary["cluster_count_seen"] += 1
        n = len(items)
        parent_size = int(items[0]["cluster_size"]) if items else n

        if not _should_subcluster_parent(
            parent_size=parent_size,
            total_article_count=total_article_count,
            min_parent_size=min_parent_size,
            trigger_parent_size=trigger_parent_size,
            trigger_parent_ratio=trigger_parent_ratio,
        ):
            summary["skipped_parent_count"] += 1
            summary["unassigned_article_count"] += n

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE clusters
                    SET has_sectors = false,
                        visible_sector_count = 0,
                        largest_sector_ratio = NULL,
                        is_oversized = %s
                    WHERE id = %s
                    """,
                    (bool(parent_size >= 40), cluster_id),
                )
            continue

        X = np.vstack([item["embedding"] for item in items]).astype(np.float32)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=SUBCLUSTER_SELECTION_METHOD,
            prediction_data=True,
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

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE clusters
                    SET has_sectors = false,
                        visible_sector_count = 0,
                        largest_sector_ratio = NULL,
                        is_oversized = %s
                    WHERE id = %s
                    """,
                    (bool(parent_size >= 40), cluster_id),
                )
            continue

        summary["cluster_count_subclustered"] += 1
        sorted_labels = sorted(grouped.keys())
        local_subcluster_payloads = []

        for label in sorted_labels:
            member_indices = grouped[label]
            member_items = [items[i] for i in member_indices]
            member_vectors = np.vstack([items[i]["embedding"] for i in member_indices]).astype(np.float32)
            centroid = _compute_centroid(member_vectors)
            representative, _member_distances = _choose_representative(member_items, member_vectors, centroid)

            probabilities_local = []
            for idx in member_indices:
                if probabilities is not None and np.isfinite(probabilities[idx]):
                    probabilities_local.append(float(probabilities[idx]))

            mean_probability = float(np.mean(probabilities_local)) if probabilities_local else None

            source_values = {
                item.get("source")
                for item in member_items
                if item.get("source")
            }

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
                    "quality_score": mean_probability,
                    "promotion_candidate": False,
                    "promoted_cluster_id": None,
                    "source_count": len(source_values),
                }
            )

        with conn.cursor() as cur:
            execute_values(
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
                    centroid,
                    quality_score,
                    promotion_candidate,
                    promoted_cluster_id,
                    source_count
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
                        row["quality_score"],
                        row["promotion_candidate"],
                        row["promoted_cluster_id"],
                        row["source_count"],
                    )
                    for row in local_subcluster_payloads
                ],
            )
            inserted = cur.fetchall()

        subcluster_id_by_label = {
            int(returned_label): int(subcluster_id)
            for subcluster_id, _returned_cluster_id, returned_label in inserted
        }

        current_cluster_article_rows = []

        for payload in local_subcluster_payloads:
            label = payload["label"]
            subcluster_id = subcluster_id_by_label[label]
            centroid = payload["centroid"]
            raw_distances = []
            per_member = []

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
                raw_distances.append(distance)
                per_member.append((item, distance, probability, outlier_score))

            distance_limit = (
                float(np.quantile(raw_distances, SUBCLUSTER_MAX_DISTANCE_QUANTILE))
                if raw_distances
                else None
            )

            for item, distance, probability, outlier_score in per_member:
                keep = True
                if probability is not None and probability < SUBCLUSTER_MIN_PROBABILITY:
                    keep = False
                if distance_limit is not None and distance > distance_limit:
                    keep = False
                if not keep:
                    continue

                current_cluster_article_rows.append(
                    (
                        subcluster_id,
                        cluster_id,
                        item["article_id"],
                        run_id,
                        bool(probability is not None and probability >= 0.5),
                        distance,
                        probability,
                        outlier_score,
                    )
                )

        assigned_counts_by_subcluster = defaultdict(int)
        for row in current_cluster_article_rows:
            assigned_subcluster_id = row[0]
            assigned_counts_by_subcluster[assigned_subcluster_id] += 1

        visible_subclusters = []
        for payload in local_subcluster_payloads:
            label = payload["label"]
            subcluster_id = subcluster_id_by_label[label]
            assigned_count = assigned_counts_by_subcluster.get(subcluster_id, 0)

            if assigned_count < MIN_SECTOR_ASSIGNED_ARTICLES:
                continue

            if parent_size > 0 and (assigned_count / parent_size) < MIN_SECTOR_PARENT_SHARE:
                continue

            quality_score = payload["quality_score"]
            if quality_score is not None and quality_score < MIN_SECTOR_QUALITY_SCORE:
                continue

            visible_subclusters.append(
                {
                    "subcluster_id": subcluster_id,
                    "assigned_count": assigned_count,
                    "quality_score": quality_score,
                }
            )

        visible_subclusters.sort(
            key=lambda x: (x["assigned_count"], x["quality_score"] or 0.0),
            reverse=True,
        )
        visible_subclusters = visible_subclusters[:MAX_SECTORS_PER_CLUSTER]
        visible_sector_count = min(len(visible_subclusters), SECTOR_MAX_VISIBLE_COUNT)

        largest_sector_ratio = (
            max((p["assigned_count"] / parent_size) for p in visible_subclusters)
            if visible_subclusters and parent_size > 0
            else None
        )
        has_sectors = visible_sector_count >= 2

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE clusters
                SET has_sectors = %s,
                    visible_sector_count = %s,
                    largest_sector_ratio = %s,
                    is_oversized = %s
                WHERE id = %s
                """,
                (
                    has_sectors,
                    visible_sector_count,
                    largest_sector_ratio,
                    bool(parent_size >= 40),
                    cluster_id,
                ),
            )

        summary["subcluster_count"] += len(local_subcluster_payloads)
        assigned_here = len(current_cluster_article_rows)
        summary["assigned_article_count"] += assigned_here
        summary["unassigned_article_count"] += max(0, n - assigned_here)

        article_rows.extend(current_cluster_article_rows)

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
                ON CONFLICT DO NOTHING
                """,
                article_rows,
            )

    return summary