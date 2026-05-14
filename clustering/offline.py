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
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=3)
    return parser.parse_args()


def parse_embedding(value):
    if value is None:
        return None

    if isinstance(value, str):
        return np.array(json.loads(value), dtype=np.float32)

    return np.array(value, dtype=np.float32)


def load_batch(window_hours=24, limit=500):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, embedding, published
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
        grouped[int(label)].append((row, X[idx]))

    ordered_labels = sorted(grouped.keys())
    if not ordered_labels:
        return 0

    cluster_insert_rows = []
    for label in ordered_labels:
        items = grouped[label]
        cluster_rows = [item[0] for item in items]
        cluster_vectors = np.vstack([item[1] for item in items]).astype(np.float32)

        centroid = compute_centroid(cluster_vectors)
        representative = choose_representative(
            cluster_rows,
            cluster_vectors,
            centroid=centroid,
        )

        cluster_insert_rows.append(
            (
                run_id,
                label,
                len(cluster_rows),
                representative["id"],
                representative["title"],
                centroid,
            )
        )

    cluster_id_by_label = {}
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO clusters (
                run_id,
                label,
                size,
                representative_article_id,
                representative_title,
                centroid
            )
            VALUES %s
            RETURNING id, label
            """,
            cluster_insert_rows,
        )
        for cluster_id, label in cur.fetchall():
            cluster_id_by_label[int(label)] = cluster_id

    link_rows = []
    for label in ordered_labels:
        cluster_id = cluster_id_by_label[label]
        for row, _vector in grouped[label]:
            link_rows.append((cluster_id, row["id"]))

    if link_rows:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO cluster_articles (cluster_id, article_id)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                link_rows,
            )

    return len(ordered_labels)


def update_run_success(conn, run_id, article_count, cluster_count, noise_count):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE clustering_runs
            SET finished_at = NOW(),
                status = %s,
                article_count = %s,
                cluster_count = %s,
                noise_count = %s
            WHERE id = %s
            """,
            ("success", article_count, cluster_count, noise_count, run_id),
        )


def update_run_failed(conn, run_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE clustering_runs
            SET finished_at = NOW(),
                status = %s
            WHERE id = %s
            """,
            ("failed", run_id),
        )


def main():
    args = parse_args()

    window_hours = args.window_hours
    limit = args.limit
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples

    rows, X = load_batch(window_hours=window_hours, limit=limit)
    if rows is None:
        print("No rows found for clustering window")
        return

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)

    counts = Counter(labels.tolist())
    noise_count = counts.get(-1, 0)
    cluster_count = len([k for k in counts.keys() if k != -1])

    print(f"label_counts={dict(sorted(counts.items()))}")
    print(f"noise_count={noise_count}")
    print(f"cluster_count={cluster_count}")

    conn = get_conn()
    run_id = None

    try:
        run_id = save_run_start(
            conn,
            window_hours=window_hours,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        saved_cluster_count = save_clusters_and_links(conn, run_id, rows, X, labels)

        update_run_success(
            conn,
            run_id=run_id,
            article_count=len(rows),
            cluster_count=saved_cluster_count,
            noise_count=noise_count,
        )

        conn.commit()
        print(f"saved_run_id={run_id}")
        print(f"saved_cluster_count={saved_cluster_count}")

    except Exception:
        conn.rollback()

        if run_id is not None:
            try:
                update_run_failed(conn, run_id)
                conn.commit()
            except Exception:
                conn.rollback()

        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()