from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import hdbscan
import numpy as np
import psycopg2.extras
from psycopg2.extras import execute_values


EMERGENT_MIN_CLUSTER_SIZE = 3
EMERGENT_MIN_SAMPLES = 2
EMERGENT_SELECTION_METHOD = "leaf"
EMERGENT_MIN_PROBABILITY = 0.35
EMERGENT_MAX_DISTANCE_QUANTILE = 0.90


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


def _choose_representative(items: list[dict], vectors: np.ndarray, centroid: np.ndarray) -> dict:
    distances = np.linalg.norm(vectors - centroid, axis=1)
    return items[int(np.argmin(distances))]


def _load_noise_articles_for_run(conn, run_id: int) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                a.id AS article_id,
                a.title,
                a.source,
                a.embedding,
                a.published AS published_at
            FROM clustering_run_articles cra
            JOIN articles a ON a.id = cra.article_id
            WHERE cra.run_id = %s
              AND cra.cluster_label = -1
              AND a.embedding IS NOT NULL
            ORDER BY a.id
            """,
            (run_id,),
        )
        rows = cur.fetchall()

    items: list[dict] = []

    for row in rows:
        embedding = _to_numpy_vector(row["embedding"])
        if embedding is None:
            continue

        items.append(
            {
                "article_id": int(row["article_id"]),
                "title": row["title"],
                "source": row["source"],
                "embedding": embedding,
                "published_at": row["published_at"],
            }
        )

    return items


def _clear_emergent_topics_for_run(conn, run_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM emergent_topics
            WHERE run_id = %s
              AND status = 'emergent'
            """,
            (run_id,),
        )


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _promotion_score(
    size: int,
    source_count: int,
    mean_probability: float | None,
) -> float:
    size_component = min(size / 10.0, 1.0)
    source_component = min(source_count / 4.0, 1.0)
    quality_component = mean_probability if mean_probability is not None else 0.0

    return float(
        0.45 * size_component
        + 0.30 * source_component
        + 0.25 * quality_component
    )


def save_emergent_topics_for_run(
    conn,
    run_id: int,
    min_cluster_size: int = EMERGENT_MIN_CLUSTER_SIZE,
    min_samples: int = EMERGENT_MIN_SAMPLES,
) -> dict:
    items = _load_noise_articles_for_run(conn, run_id)

    summary = {
        "noise_article_count": len(items),
        "emergent_topic_count": 0,
        "assigned_article_count": 0,
        "unassigned_article_count": len(items),
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM emergent_topic_articles
            WHERE run_id = %s
            """,
            (run_id,),
        )

    _clear_emergent_topics_for_run(conn, run_id)

    if len(items) < min_cluster_size:
        return summary

    X = np.vstack([item["embedding"] for item in items]).astype(np.float32)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=EMERGENT_SELECTION_METHOD,
        prediction_data=True,
    )

    labels = clusterer.fit_predict(X)
    probabilities = getattr(clusterer, "probabilities_", None)

    grouped: dict[int, list[int]] = defaultdict(list)

    for idx, label in enumerate(labels.tolist()):
        if int(label) != -1:
            grouped[int(label)].append(idx)

    if not grouped:
        return summary

    topic_rows = []
    topic_members = []

    for label in sorted(grouped):
        member_indices = grouped[label]
        member_items = [items[idx] for idx in member_indices]
        member_vectors = np.vstack(
            [items[idx]["embedding"] for idx in member_indices]
        ).astype(np.float32)

        centroid = _compute_centroid(member_vectors)
        representative = _choose_representative(
            member_items,
            member_vectors,
            centroid,
        )

        raw_distances = np.linalg.norm(member_vectors - centroid, axis=1)
        distance_limit = float(
            np.quantile(raw_distances, EMERGENT_MAX_DISTANCE_QUANTILE)
        )

        accepted_indices = []
        probabilities_local = []

        for idx in member_indices:
            probability = (
                float(probabilities[idx])
                if probabilities is not None and np.isfinite(probabilities[idx])
                else None
            )

            distance = float(np.linalg.norm(items[idx]["embedding"] - centroid))

            if probability is not None and probability < EMERGENT_MIN_PROBABILITY:
                continue

            if distance > distance_limit:
                continue

            accepted_indices.append(idx)

            if probability is not None:
                probabilities_local.append(probability)

        if len(accepted_indices) < min_cluster_size:
            continue

        accepted_items = [items[idx] for idx in accepted_indices]

        source_count = len(
            {
                item["source"]
                for item in accepted_items
                if item.get("source")
            }
        )

        timestamps = [
            item["published_at"]
            for item in accepted_items
            if item.get("published_at") is not None
        ]

        first_seen_at = min(timestamps) if timestamps else None
        last_seen_at = max(timestamps) if timestamps else None

        mean_probability = _safe_mean(probabilities_local)
        promotion_score = _promotion_score(
            size=len(accepted_indices),
            source_count=source_count,
            mean_probability=mean_probability,
        )

        topic_rows.append(
            {
                "label": label,
                "row": (
                    run_id,
                    "emergent",
                    representative["article_id"],
                    representative["article_id"],
                    representative["title"],
                    len(accepted_indices),
                    source_count,
                    first_seen_at,
                    last_seen_at,
                    mean_probability,
                    mean_probability,
                    promotion_score,
                ),
            }
        )

        for idx in accepted_indices:
            probability = (
                float(probabilities[idx])
                if probabilities is not None and np.isfinite(probabilities[idx])
                else None
            )
            distance = float(np.linalg.norm(items[idx]["embedding"] - centroid))

            topic_members.append(
                {
                    "label": label,
                    "article_id": items[idx]["article_id"],
                    "probability": probability,
                    "distance": distance,
                    "is_representative": items[idx]["article_id"] == representative["article_id"],
                }
            )

    if not topic_rows:
        return summary

    inserted = []
    with conn.cursor() as cur:
        inserted = execute_values(
            cur,
            """
            INSERT INTO emergent_topics (
                run_id,
                status,
                seed_article_id,
                representative_article_id,
                title,
                size,
                source_count,
                first_seen_at,
                last_seen_at,
                mean_similarity,
                mean_confidence,
                promotion_score
            )
            VALUES %s
            RETURNING id, representative_article_id, title
            """,
            [item["row"] for item in topic_rows],
            fetch=True,
            page_size=len(topic_rows),
        )

    topic_id_by_label = {}
    for item, inserted_row in zip(topic_rows, inserted):
        topic_id_by_label[item["label"]] = int(inserted_row[0])

    membership_rows = []
    for member in topic_members:
        emergent_topic_id = topic_id_by_label[member["label"]]
        membership_rows.append(
            (
                emergent_topic_id,
                member["article_id"],
                run_id,
                member["probability"],
                member["distance"],
                member["is_representative"],
            )
        )

    if membership_rows:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO emergent_topic_articles (
                    emergent_topic_id,
                    article_id,
                    run_id,
                    probability,
                    distance,
                    is_representative
                )
                VALUES %s
                ON CONFLICT (emergent_topic_id, article_id) DO UPDATE
                SET probability = EXCLUDED.probability,
                    distance = EXCLUDED.distance,
                    is_representative = EXCLUDED.is_representative,
                    run_id = EXCLUDED.run_id
                """,
                membership_rows,
            )

    summary["emergent_topic_count"] = len(topic_rows)
    summary["assigned_article_count"] = len(membership_rows)
    summary["unassigned_article_count"] = max(
        0,
        len(items) - summary["assigned_article_count"],
    )

    return summary