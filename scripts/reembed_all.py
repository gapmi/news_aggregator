import argparse
import os
from typing import List

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def build_text(row) -> str:
    parts: List[str] = []

    title = (row.get("title") or "").strip()
    description = (row.get("description") or "").strip()
    content = (row.get("content") or "").strip()

    if title:
        parts.append(title)
    if description:
        parts.append(description)
    if content:
        parts.append(content)

    text = "\n\n".join(parts).strip()
    return text[:8000]


def load_batch(conn, batch_size: int, offset: int):
    query = """
        SELECT id, title, description, content
        FROM articles
        WHERE COALESCE(title, '') <> ''
        ORDER BY id
        LIMIT %s OFFSET %s
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (batch_size, offset))
        return cur.fetchall()


def update_embeddings(conn, ids, embeddings):
    rows = [(embeddings[i], ids[i]) for i in range(len(ids))]

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            UPDATE articles
            SET embedding = %s
            WHERE id = %s
            """,
            rows,
            page_size=100,
        )


def main():
    args = parse_args()

    model = SentenceTransformer(MODEL_NAME)
    conn = get_conn()

    total_processed = 0
    current_offset = args.offset

    try:
        while True:
            if args.limit and total_processed >= args.limit:
                break

            current_batch_size = args.batch_size
            if args.limit:
                remaining = args.limit - total_processed
                current_batch_size = min(current_batch_size, remaining)

            batch = load_batch(
                conn,
                batch_size=current_batch_size,
                offset=current_offset,
            )

            if not batch:
                break

            ids = []
            texts = []

            for row in batch:
                text = build_text(row)
                if not text:
                    continue
                ids.append(row["id"])
                texts.append(text)

            if not ids:
                current_offset += len(batch)
                continue

            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            update_embeddings(conn, ids, embeddings)
            conn.commit()

            total_processed += len(ids)
            current_offset += len(batch)

            print(
                f"processed={total_processed} "
                f"last_batch={len(ids)} "
                f"offset={current_offset}"
            )

        print(f"done total_processed={total_processed}")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()