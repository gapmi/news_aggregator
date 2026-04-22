import os
import time

import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector

from scrapers.base import Article


class PGStorage:
    def __init__(self):
        self.conn = None
        self.embedding_service = None

        while self.conn is None:
            try:
                self.conn = psycopg2.connect(
                    host=os.getenv("DB_HOST", "db"),
                    port=os.getenv("DB_PORT", "5432"),
                    dbname=os.getenv("DB_NAME", "news_db"),
                    user=os.getenv("DB_USER", "postgres"),
                    password=os.getenv("DB_PASSWORD", "qg9PlWWpeffd")
                )
                register_vector(self.conn)
                self._create_table()
                print("Database initialized successfully!")
            except Exception as e:
                print(f"Waiting for DB... error: {e}")
                time.sleep(5)

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    url TEXT UNIQUE,
                    published TIMESTAMP,
                    source TEXT,
                    embedding vector(384)
                )
            """)
        self.conn.commit()

    def _get_embedding_service(self):
        if self.embedding_service is None:
            from processors.embeddings import EmbeddingService
            self.embedding_service = EmbeddingService()
        return self.embedding_service

    def save(self, articles: list[Article]):
        saved_rows = []

        with self.conn.cursor() as cur:
            for a in articles:
                cur.execute("""
                    INSERT INTO articles (title, url, published, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE
                    SET title = EXCLUDED.title,
                        published = EXCLUDED.published,
                        source = EXCLUDED.source
                    RETURNING id, title
                """, (a.title, a.url, a.published, a.source))

                row = cur.fetchone()
                saved_rows.append({
                    "id": row[0],
                    "title": row[1],
                })

        self.conn.commit()

        payload = [
            {"id": row["id"], "title": row["title"]}
            for row in saved_rows
            if row["title"]
        ]

        if not payload:
            return

        try:
            from processors.embeddings import ArticleText

            embedding_service = self._get_embedding_service()

            article_payload = [
                ArticleText(id=row["id"], title=row["title"], content=None)
                for row in payload
            ]

            vectors = embedding_service.encode_batch(article_payload, batch_size=8)

            update_rows = [
                (vector, article.id)
                for article, vector in zip(article_payload, vectors)
            ]

            with self.conn.cursor() as cur:
                execute_batch(
                    cur,
                    "UPDATE articles SET embedding = %s WHERE id = %s",
                    update_rows,
                    page_size=50
                )

            self.conn.commit()
            print(f"Embeddings updated: {len(update_rows)}")

        except Exception as e:
            print(f"Embedding generation skipped due to error: {e}")
            self.conn.rollback()