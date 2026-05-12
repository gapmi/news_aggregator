import json
import os

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
                LIMIT 5
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

            print(f"first_id={ids[0]}")
            print(f"first_title={rows[0]['title']}")
            print(f"shape={X.shape}")
            print(f"dtype={X.dtype}")
            print(f"first_dim={X.shape[1]}")
            print(f"has_nan={np.isnan(X).any()}")
            print(f"first_5_values={X[0][:5].tolist()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()