import json
import os
from collections import Counter, defaultdict

import hdbscan
import numpy as np
import psycopg2
import psycopg2.extras


def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "news_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "qg9PlWWpeffd"),
    )


def parse_embedding(value):
    if value is None:
        return None

    if isinstance(value, str):
        return np.array(json.loads(value), dtype=np.float32)

    return np.array(value, dtype=np.float32)


def load_batch(limit=200):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, embedding
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    print(f"rows={len(rows)}")
    if not rows:
        return None, None

    embeddings = []
    for row in rows:
        embeddings.append(parse_embedding(row["embedding"]))
    X = np.vstack(embeddings)

    print(f"shape={X.shape}")
    print(f"dtype={X.dtype}")
    print(f"has_nan={np.isnan(X).any()}")

    return rows, X


def run_config(rows, X, name, min_cluster_size, min_samples):
    print(
        f"\n\n===== CONFIG {name} "
        f"(min_cluster_size={min_cluster_size}, min_samples={min_samples}) ====="
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)

    counts = Counter(labels.tolist())
    noise_count = counts.get(-1, 0)
    cluster_labels = [k for k in counts.keys() if k != -1]
    cluster_count = len(cluster_labels)

    print(f"label_counts={dict(sorted(counts.items()))}")
    print(f"noise_count={noise_count}")
    print(f"cluster_count={cluster_count}")

    clusters = defaultdict(list)
    for row, label in zip(rows, labels):
        clusters[label].append(row)

    for label in sorted(cluster_labels):
        print(f"\n--- Cluster {label} (size={len(clusters[label])}) ---")
        for article in clusters[label][:5]:
            print(f"- [{article['id']}] {article['title']}")

    if -1 in clusters:
        print(f"\n--- Noise (label=-1, size={len(clusters[-1])}) first 5 ---")
        for article in clusters[-1][:5]:
            print(f"- [{article['id']}] {article['title']}")


def main():
    rows, X = load_batch(limit=200)
    if rows is None:
        return

    configs = [
        ("baseline", 5, 3),
        ("more_specific", 4, 2),
        ("more_general", 8, 4),
    ]

    for name, min_cluster_size, min_samples in configs:
        run_config(rows, X, name, min_cluster_size, min_samples)


if __name__ == "__main__":
    main()