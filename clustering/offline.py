import argparse
import json
import os
import hdbscan

import numpy as np
import psycopg2
import psycopg2.extras

from collections import Counter, defaultdict
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values


DEFAULT_WINDOW_HOURS = 168
DEFAULT_LIMIT = 3000
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 3

DEFAULT_MIN_VALID_CLUSTER_COUNT = 10
DEFAULT_MAX_LARGEST_CLUSTER_RATIO = 0.50
DEFAULT_MAX_PER_SOURCE = 300


def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        conn = psycopg2.connect(db_url)
        register_vector(conn)
        return conn

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "news_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "qg9PlWWpeffd"),
    )
    register_vector(conn)
    return conn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="auto", choices=["auto"])
    parser.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)
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


def parse_embedding(value):
    if value is None:
        return None

    if isinstance(value, str):
        return np.array(json.loads(value), dtype=np.float32)

    return np.array(value, dtype=np.float32)


def load_batch(
    window_hours=DEFAULT_WINDOW_HOURS,
    limit=DEFAULT_LIMIT,
    max_per_source=DEFAULT_MAX_PER_SOURCE,
):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if max_per_source and max_per_source > 0:
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT
                            id,
                            title,
                            embedding,
                            published,
                            source,
                            ROW_NUMBER() OVER (
                                PARTITION BY source
                                ORDER BY published DESC, id DESC
                            ) AS source_rank
                        FROM articles
                        WHERE embedding IS NOT NULL
                          AND published IS NOT NULL
                          AND published >= NOW() - (%s || ' hours')::interval
                    ),
                    balanced AS (
                        SELECT id, title, embedding, published, source
                        FROM ranked
                        WHERE source_rank <= %s
                    )
                    SELECT id, title, embedding, published, source
                    FROM balanced
                    ORDER BY published DESC, id DESC
                    LIMIT %s
                    """,
                    (window_hours, max_per_source, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, title, embedding, published, source
                    FROM articles
                    WHERE embedding IS NOT NULL
                      AND published IS NOT NULL
                      AND published >= NOW() - (%s || ' hours')::interval
                    ORDER BY published DESC, id DESC
                    LIMIT %s
                    """,
                    (window_hours, limit),
                )

            rows = cur.fetchall()
    finally:
        conn.close()

    print(f"rows={len(rows)}")
    if not rows:
        return None, None

    embeddings = [parse_embedding(row["embedding"]) for row in rows]
    X = np.vstack(embeddings).astype(np.float32)

    print(f"shape={X.shape}")
    print(f"dtype={X.dtype}")
    print(f"has_nan={np.isnan(X).any()}")

    return rows, X


def compute_centroid(cluster_vectors):
    return cluster_vectors.mean(axis=0).astype(np.float32)


def validate_clustering_quality(
    labels,
    total_count: int,
    min_valid_cluster_count: int = DEFAULT_MIN_VALID_CLUSTER_COUNT,
    max_largest_cluster_ratio: float = DEFAULT_MAX_LARGEST_CLUSTER_RATIO,
) -> None:
    counts = Counter(labels.tolist())

    non_noise_counts = [
        count
        for label, count in counts.items()
        if int(label) != -1
    ]

    cluster_count = len(non_noise_counts)
    largest_cluster_size = max(non_noise_counts) if non_noise_counts else 0
    largest_cluster_ratio = (
        largest_cluster_size / total_count
        if total_count > 0
        else 0.0
    )

    if cluster_count < min_valid_cluster_count:
        raise ValueError(
            "clustering quality check failed: "
            f"cluster_count={cluster_count} < {min_valid_cluster_count}"
        )

    if largest_cluster_ratio > max_largest_cluster_ratio:
        raise ValueError(
            "clustering quality check failed: "
            f"largest_cluster_ratio={largest_cluster_ratio:.4f} "
            f"> {max_largest_cluster_ratio:.4f}; "
            f"largest_cluster_size={largest_cluster_size}; "
            f"total_count={total_count}"
        )


def choose_representative(cluster_rows, cluster_vectors, centroid=None):
    if centroid is None:
        centroid = compute_centroid(cluster_vectors)
    distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
    idx = int(np.argmin(distances))
    return cluster_rows[idx]


def save_run_start(conn, window_hours, min_cluster_size, min_samples):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO clustering_runs (
                status,
                window_hours,
                min_cluster_size,
                min_samples,
                article_count,
                cluster_count,
                noise_count
            )
            VALUES (%s, %s, %s, %s, 0, 0, 0)
            RETURNING id
            """,
            ("running", window_hours, min_cluster_size, min_samples),
        )
        run_id = cur.fetchone()[0]
    return run_id


def save_clusters_and_links(conn, run_id, rows, X, labels):
    grouped = defaultdict(list)

    for idx, (row, label) in enumerate(zip(rows, labels)):
        if int(label) == -1:
            continue
        grouped