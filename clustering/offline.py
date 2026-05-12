import json
import os
from collections import Counter

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


def main():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, embedding
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY id DESC
                LIMIT 200
                """
            )
            rows = cur.fetchall()

            print(f"rows={len(rows)}")
            if not rows:
                return

            embeddings = []
            ids = []

            for row in rows:
                vec = parse_embedding(row["embedding"])
                embeddings.append(vec)
                ids.append(row["id"])

            X = np.vstack(embeddings)

            print(f"shape={X.shape}")
            print(f"dtype={X.dtype}")
            print(f"has_nan={np.isnan(X).any()}")

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=3,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(X)

            counts = Counter(labels.tolist())

            print(f"label_counts={dict(sorted(counts.items()))}")
            print(f"noise_count={counts.get(-1, 0)}")
            print(f"cluster_count={len([k for k in counts.keys() if k != -1])}")

            for i in range(min(10, len(rows))):
                print(f"id={rows[i]['id']} label={labels[i]} title={rows[i]['title']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()