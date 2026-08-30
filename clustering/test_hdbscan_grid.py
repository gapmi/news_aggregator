# test_hdbscan_grid.py
import argparse
import json
import os
from collections import Counter

import hdbscan
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector


WINDOW_HOURS = 168
LIMIT = 3000
MAX_PER_SOURCE = 300


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


def parse_embedding(value):
    if value is None:
        return None
    if isinstance(value, str):
        return np.array(json.loads(value), dtype=np.float32)
    if hasattr(value, "to_numpy"):
        return value.to_numpy().astype(np.float32)
    if hasattr(value, "to_list"):
        return np.array(value.to_list(), dtype=np.float32)
    if hasattr(value, "tolist"):
        return np.array(value.tolist(), dtype=np.float32)
    return np.array(value, dtype=np.float32)


def l2_normalize_embeddings(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0):
        zero_count = int(np.sum(norms == 0))
        raise ValueError(
            f"Cannot normalize embeddings: found {zero_count} zero-norm vectors"
        )
    return (X / norms).astype(np.float32)


def load_batch(
    window_hours=WINDOW_HOURS,
    limit=LIMIT,
    max_per_source=MAX_PER_SOURCE,
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

    if not rows:
        return None, None

    embeddings = [parse_embedding(row["embedding"]) for row in rows]
    X = np.vstack(embeddings).astype(np.float32)

    if np.isnan(X).any():
        raise ValueError("Cannot cluster embeddings containing NaN values")
    if not np.isfinite(X).all():
        raise ValueError("Cannot cluster embeddings containing non-finite values")

    X = l2_normalize_embeddings(X)
    return rows, X


def run_clustering(
    min_cluster_size,
    min_samples,
    cluster_selection_method,
    cluster_selection_epsilon,
):
    rows, X = load_batch()
    if rows is None:
        print("No rows found for clustering window")
        return

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)

    counts = Counter(labels.tolist())
    noise_count = counts.get(-1, 0)
    non_noise_counts = [c for l, c in counts.items() if int(l) != -1]

    cluster_count = len(non_noise_counts)
    largest_cluster_size = max(non_noise_counts) if non_noise_counts else 0
    largest_cluster_ratio = largest_cluster_size / len(rows) if rows else 0.0
    noise_ratio = noise_count / len(rows) if rows else 0.0

    print(
        f"CONFIG: min_cluster_size={min_cluster_size}, "
        f"min_samples={min_samples}, "
        f"method={cluster_selection_method}, "
        f"epsilon={cluster_selection_epsilon}"
    )
    print(f"articles={len(rows)}, clusters={cluster_count}, noise={noise_count}, "
          f"noise_ratio={noise_ratio:.4f}, largest={largest_cluster_size}, "
          f"largest_ratio={largest_cluster_ratio:.4f}")
    print(f"label_counts={dict(sorted(counts.items()))}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cluster-size", type=int, required=True)
    parser.add_argument("--min-samples", type=int, required=True)
    parser.add_argument(
        "--cluster-selection-method",
        default="leaf",
        choices=["eom", "leaf"],
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    run_clustering(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )


if __name__ == "__main__":
    main()